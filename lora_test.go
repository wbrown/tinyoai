package tinyoai

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// loRAFixture writes a two-layer PEFT adapter for the repository's tiny model.
// The edit callback can corrupt metadata or one tensor before serialization.
func loRAFixture(t *testing.T, edit func(map[string]any, map[string]weightTensor)) string {
	t.Helper()
	config := map[string]any{"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 2, "lora_alpha": 1,
		"base_model_name_or_path": "tiny-fixture", "target_modules": []string{"q_proj", "down_proj"}, "bias": "none"}
	weights := make(map[string]weightTensor)
	for layer := 0; layer < 2; layer++ {
		for _, target := range []string{"self_attn.q_proj", "mlp.down_proj"} {
			input := 16
			if target == "mlp.down_proj" {
				input = 40
			}
			for i, shape := range [][]int{{2, input}, {16, 2}} {
				name := "base_model.model.model.layers." + string(rune('0'+layer)) + "." + target + []string{".lora_A.weight", ".lora_B.weight"}[i]
				data := make([]byte, shape[0]*shape[1]*4)
				for j := 0; j < len(data)/4; j++ {
					binary.LittleEndian.PutUint32(data[j*4:], math.Float32bits(float32((j+3*layer)%11-5)*0.025))
				}
				weights[name] = weightTensor{data: data, dtype: "F32", shape: shape}
			}
		}
	}
	if edit != nil {
		edit(config, weights)
	}
	header := make(map[string]tensorInfo)
	var payload []byte
	for name, tensor := range weights {
		start := len(payload)
		payload = append(payload, tensor.data...)
		header[name] = tensorInfo{Dtype: tensor.dtype, Shape: tensor.shape, Offsets: []int64{int64(start), int64(len(payload))}}
	}
	data, _ := json.Marshal(header)
	file := writeSafeTest(t, string(data), payload)
	dir := filepath.Dir(file)
	if err := os.Rename(file, filepath.Join(dir, "adapter_model.safetensors")); err != nil {
		t.Fatal(err)
	}
	data, _ = json.Marshal(config)
	if err := os.WriteFile(filepath.Join(dir, "adapter_config.json"), data, 0600); err != nil {
		t.Fatal(err)
	}
	return dir
}

// loRATestConfig returns the dimensions shared by the synthetic adapter/model.
func loRATestConfig(t *testing.T) StableLMConfig {
	t.Helper()
	data, err := os.ReadFile("testdata/stablelm/config.json")
	if err != nil {
		t.Fatal(err)
	}
	var c StableLMConfig
	if err = json.Unmarshal(data, &c); err != nil {
		t.Fatal(err)
	}
	return c
}

// TestLoRAValidation exercises shape/name/rank coverage, unsupported PEFT
// variants, finite values, cancellation, and the content identity used by logs.
func TestLoRAValidation(t *testing.T) {
	ctx := context.Background()
	cfg := loRATestConfig(t)
	dir := loRAFixture(t, nil)
	info, tensors, err := readLoRA(ctx, dir, cfg, 0.75)
	if err != nil || info.Rank != 2 || info.Alpha != 1 || info.Scale != 0.75 || info.Projections != 4 || len(tensors) != 8 || len(info.SHA256) != 64 || info.Bytes != 704 {
		t.Fatalf("valid adapter: %+v, tensors=%d, err=%v", info, len(tensors), err)
	}
	again, _, err := readLoRA(ctx, dir, cfg, 0)
	if err != nil || again.SHA256 != info.SHA256 {
		t.Fatal("scale changed content identity", err)
	}
	for name, edit := range map[string]func(map[string]any, map[string]weightTensor){
		"missing-alpha":  func(c map[string]any, _ map[string]weightTensor) { delete(c, "lora_alpha") },
		"dora":           func(c map[string]any, _ map[string]weightTensor) { c["use_dora"] = true },
		"rslora":         func(c map[string]any, _ map[string]weightTensor) { c["use_rslora"] = true },
		"rank-pattern":   func(c map[string]any, _ map[string]weightTensor) { c["rank_pattern"] = map[string]int{"q_proj": 4} },
		"transpose":      func(c map[string]any, _ map[string]weightTensor) { c["fan_in_fan_out"] = true },
		"extra-module":   func(c map[string]any, _ map[string]weightTensor) { c["modules_to_save"] = []string{"lm_head"} },
		"wrong-rank":     func(c map[string]any, _ map[string]weightTensor) { c["r"] = 3 },
		"bias":           func(c map[string]any, _ map[string]weightTensor) { c["bias"] = "all" },
		"target":         func(c map[string]any, _ map[string]weightTensor) { c["target_modules"] = []string{"lm_head"} },
		"unknown-option": func(c map[string]any, _ map[string]weightTensor) { c["future_variant"] = true },
		"missing": func(_ map[string]any, w map[string]weightTensor) {
			for k := range w {
				delete(w, k)
				break
			}
		},
		"wrong-name": func(_ map[string]any, w map[string]weightTensor) {
			for k, v := range w {
				delete(w, k)
				w["wrong.name"] = v
				break
			}
		},
		"nan": func(_ map[string]any, w map[string]weightTensor) {
			for _, v := range w {
				binary.LittleEndian.PutUint32(v.data, math.Float32bits(float32(math.NaN())))
				break
			}
		},
		"overflow": func(_ map[string]any, w map[string]weightTensor) {
			for _, v := range w {
				binary.LittleEndian.PutUint32(v.data, math.Float32bits(70000))
				break
			}
		},
	} {
		t.Run(name, func(t *testing.T) {
			if _, _, err := readLoRA(ctx, loRAFixture(t, edit), cfg, 1); err == nil {
				t.Fatal("accepted invalid adapter")
			}
		})
	}
	for _, scale := range []float64{math.NaN(), math.Inf(1), 200000} {
		if _, _, err := readLoRA(ctx, dir, cfg, scale); err == nil {
			t.Fatal("accepted invalid strength", scale)
		}
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if _, _, err := readLoRA(cancelled, dir, cfg, 1); err != context.Canceled {
		t.Fatal("ignored cancellation", err)
	}
}

// TestLoRAUnsupportedBackend ensures ordinary builds expose a useful error
// without attempting to read a checkpoint or blocking a cancelled caller.
func TestLoRAUnsupportedBackend(t *testing.T) {
	m := &MLX{gate: make(chan struct{}, 1)}
	m.gate <- struct{}{}
	if _, err := m.LoadLoRA(context.Background(), "missing", 1); err == nil {
		t.Fatal("accepted unsupported backend")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	<-m.gate
	if _, err := m.LoadLoRA(ctx, "missing", 1); err != context.Canceled {
		t.Fatal("cancelled caller waited for gate", err)
	}
}
