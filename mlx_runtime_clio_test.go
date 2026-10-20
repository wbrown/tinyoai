//go:build mlx && darwin && arm64 && cgo

package tinyoai

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	mx "github.com/wbrown/tinyoai/internal/mlx"
)

// Requires real Clio and an existing compatible adapter. The copy below widens
// the adapter for an F32 inference check; it performs no optimization or training.
func TestClioRuntimeInference(t *testing.T) {
	modelDir, adapterDir := os.Getenv("TINYOAI_CLIO_MODEL"), os.Getenv("TINYOAI_CLIO_ADAPTER")
	if modelDir == "" || adapterDir == "" {
		t.Skip("requires actual Clio weights and a compatible adapter")
	}
	m, err := LoadMLXNative(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ctx := context.Background()
	_, tensors, err := readLoRA(ctx, adapterDir, m.config, 1)
	if err != nil {
		t.Fatal(err)
	}
	header := make(map[string]tensorInfo)
	var payload []byte
	for i, tensor := range tensors {
		start := len(payload)
		for _, value := range tensor.values {
			payload = binary.LittleEndian.AppendUint32(payload, math.Float32bits(value))
		}
		suffix := []string{".lora_A.weight", ".lora_B.weight"}[i%2]
		header["base_model.model."+tensor.name+suffix] = tensorInfo{
			Dtype: "F32", Shape: tensor.shape, Offsets: []int64{int64(start), int64(len(payload))},
		}
	}
	raw, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	file := writeSafeTest(t, string(raw), payload)
	dir := filepath.Dir(file)
	if err = os.Rename(file, filepath.Join(dir, "adapter_model.safetensors")); err != nil {
		t.Fatal(err)
	}
	config, err := os.ReadFile(filepath.Join(adapterDir, "adapter_config.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(dir, "adapter_config.json"), config, 0600); err != nil {
		t.Fatal(err)
	}
	prompt := "The horse was black. Mara left its bridle on the gate and listened to the rain."
	ids := m.tokenizer.encode(prompt)
	forward := func(model *MLX) []float32 {
		t.Helper()
		row, e := model.native.Forward(ctx, 0, ids, true)
		if e != nil {
			t.Fatal(e)
		}
		for _, v := range row {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatal("non-finite inference")
			}
		}
		return row
	}
	base := forward(m)
	info, err := m.LoadLoRA(ctx, dir, 1)
	if err != nil || info.Dtype != "F32" {
		t.Fatal(info, err)
	}
	for _, pair := range m.native.(*nativeMLX).lora.pairs {
		if err = mx.Run(func() {
			if pair.a.Dtype() != mx.Float32 || pair.b.Dtype() != mx.Float32 {
				t.Error("native adapter was narrowed")
			}
		}); err != nil {
			t.Fatal(err)
		}
	}
	adapted := forward(m)
	if reflect.DeepEqual(base, adapted) {
		t.Fatal("adapter did not affect inference")
	}
	session, err := m.Fork(ctx)
	if err != nil {
		t.Fatal(err)
	}
	fork := session.(*MLX)
	defer fork.Close()
	if !reflect.DeepEqual(adapted, forward(fork)) {
		t.Fatal("fork changed adapter output")
	}
	if _, err = m.SetLoRAScale(ctx, 0); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(base, forward(m)) {
		t.Fatal("zero scale did not restore base")
	}
	if err = m.Close(); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(adapted, forward(fork)) {
		t.Fatal("parent close changed child output")
	}
	var plans []CachePlan
	opts := GenerateOptions{MaxTokens: 2, Logprobs: 65, OnCachePlan: func(p CachePlan) error {
		plans = append(plans, p)
		return nil
	}}
	if _, err = fork.Prefill(prompt, opts); err != nil {
		t.Fatal(err)
	}
	result, err := fork.Generate(prompt, opts)
	if err != nil || len(plans) != 2 || plans[1].ReusedTokens != result.CachedPromptTokens {
		t.Fatal(plans, result, err)
	}
	for _, record := range result.Logprobs {
		if len(record.Top) != 65 {
			t.Fatal("lost requested alternatives", len(record.Top))
		}
	}
	if err = fork.Close(); err != nil {
		t.Fatal(err)
	}
	t.Logf("actual Clio: %d prompt tokens, %d generated tokens, %d F32 projections; base restoration, fork lifetime and cache callback passed",
		len(ids), result.CompletionTokens, info.Projections)
}
