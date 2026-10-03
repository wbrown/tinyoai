package tinyoai

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
)

// LoRAInfo identifies the adapter attached to a generation session. An empty
// SHA256 means no adapter. Scale zero bypasses a loaded adapter without freeing
// its tensors. The zero value describes an unadapted session.
type LoRAInfo struct {
	// SHA256 hashes the configuration bytes followed by the weight-file bytes.
	SHA256 string `json:"sha256,omitempty"`
	// BaseModel is the adapter's declared base checkpoint, not a verified identity.
	BaseModel string `json:"base_model,omitempty"`
	// Rank is the shared inner dimension of each low-rank matrix pair.
	Rank int `json:"rank,omitempty"`
	// Alpha is the adapter's normalization constant; each correction uses Alpha/Rank.
	Alpha float64 `json:"alpha,omitempty"`
	// Scale is the requested strength, before multiplying by Alpha/Rank.
	Scale float64 `json:"scale"`
	// Projections counts adapted base matrices across all layers, one per A/B pair.
	Projections int `json:"projections,omitempty"`
	// Bytes counts retained FP16 tensor storage, excluding overhead and workspaces.
	// Forks can share these bytes; summing their reports would count storage twice.
	Bytes uint64 `json:"bytes,omitempty"`
}

// LoRAController changes one session's adapter while retaining its base weights.
// Successful mutations invalidate that session's KV and saved probabilities.
// Adapters must have been trained against the caller's base checkpoint.
// Calls serialize with inference and can cancel while waiting for the session.
type LoRAController interface {
	// LoadLoRA replaces the adapter from dir at the requested strength. An empty
	// dir unloads it. A failed load preserves the previous adapter and cache.
	LoadLoRA(ctx context.Context, dir string, scale float64) (LoRAInfo, error)
	// SetLoRAScale changes strength without rereading tensors. Zero bypasses the
	// correction; negative strengths are allowed. No loaded adapter is an error.
	SetLoRAScale(ctx context.Context, scale float64) (LoRAInfo, error)
	// LoRAInfo reports the resident adapter after earlier session work finishes.
	LoRAInfo(ctx context.Context) (LoRAInfo, error)
}

// mlxLoRAForwarder owns native adapter state behind the MLX session gate.
// Successful mutations clear native KV before the caller drops its Go records.
type mlxLoRAForwarder interface {
	// loadLoRA evaluates replacement tensors before committing; an empty dir unloads.
	loadLoRA(ctx context.Context, dir string, scale float64) (LoRAInfo, error)
	// setLoRAScale validates the multiplier and invalidates KV before changing it.
	setLoRAScale(scale float64) (LoRAInfo, error)
	// loRAInfo returns metadata without changing adapter tensors or cached state.
	loRAInfo() LoRAInfo
}

// LoadLoRA atomically replaces this session's adapter from a PEFT directory.
// An empty directory unloads it. The native StableLM backend supports ordinary
// inference LoRA on attention and MLP projections; other backends return an
// error. Failed loads preserve the current adapter and cache. Cancellation
// waits for any admitted native evaluation to finish before releasing handles.
func (m *MLX) LoadLoRA(ctx context.Context, dir string, scale float64) (LoRAInfo, error) {
	return m.withLoRA(ctx, true, func(ctx context.Context, n mlxLoRAForwarder) (LoRAInfo, error) {
		return n.loadLoRA(ctx, dir, scale)
	})
}

// SetLoRAScale changes the loaded adapter's strength without rereading weights.
// Zero bypasses the correction; negative strengths are allowed. The effective
// alpha/rank multiplier must be finite and representable in FP16.
func (m *MLX) SetLoRAScale(ctx context.Context, scale float64) (LoRAInfo, error) {
	return m.withLoRA(ctx, true, func(_ context.Context, n mlxLoRAForwarder) (LoRAInfo, error) {
		return n.setLoRAScale(scale)
	})
}

// LoRAInfo returns a snapshot after any active generation or adapter change.
func (m *MLX) LoRAInfo(ctx context.Context) (LoRAInfo, error) {
	return m.withLoRA(ctx, false, func(_ context.Context, n mlxLoRAForwarder) (LoRAInfo, error) {
		return n.loRAInfo(), nil
	})
}

// withLoRA serializes adapter access with generation, forks, branches, and Close.
// The forwarder commits native cache invalidation before Go drops its history.
func (m *MLX) withLoRA(ctx context.Context, mutate bool, fn func(context.Context, mlxLoRAForwarder) (LoRAInfo, error)) (LoRAInfo, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	select {
	case <-ctx.Done():
		return LoRAInfo{}, ctx.Err()
	case <-m.gate:
	}
	defer func() { m.gate <- struct{}{} }()
	if err := ctx.Err(); err != nil {
		return LoRAInfo{}, err
	}
	if m.closed {
		return LoRAInfo{}, fmt.Errorf("MLX backend is closed")
	}
	n, ok := m.native.(mlxLoRAForwarder)
	if !ok {
		return LoRAInfo{}, fmt.Errorf("LoRA requires the native StableLM MLX backend")
	}
	info, err := fn(ctx, n)
	if err == nil && mutate {
		m.tokens, m.probRows, m.probCount = nil, nil, 0
	}
	return info, err
}

type loRAConfig struct {
	Type      string   `json:"peft_type"`
	Task      string   `json:"task_type"`
	BaseModel string   `json:"base_model_name_or_path"`
	Rank      int      `json:"r"`
	Alpha     float64  `json:"lora_alpha"`
	Targets   []string `json:"target_modules"`
	Bias      string   `json:"bias"`
}

type loRATensor struct {
	name   string
	shape  []int
	values []float32
}

// parseLoRAConfig accepts PEFT's ordinary inference form. Training metadata
// does not alter evaluation; nonempty variant options are rejected rather
// than silently applying the wrong scaling or missing extra trained weights.
func parseLoRAConfig(data []byte) (loRAConfig, error) {
	var c loRAConfig
	if err := json.Unmarshal(data, &c); err != nil {
		return c, err
	}
	if c.Type != "LORA" || (c.Task != "" && c.Task != "CAUSAL_LM") || c.Rank < 1 ||
		c.Alpha < 0 || math.IsNaN(c.Alpha) || math.IsInf(c.Alpha, 0) ||
		(c.Bias != "" && c.Bias != "none") || len(c.Targets) == 0 {
		return c, fmt.Errorf("invalid or unsupported ordinary LoRA configuration")
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return c, err
	}
	if alpha, ok := fields["lora_alpha"]; !ok || string(alpha) == "null" {
		return c, fmt.Errorf("LoRA must specify lora_alpha")
	}
	for name, value := range fields {
		switch name {
		case "peft_type", "task_type", "base_model_name_or_path", "r", "lora_alpha", "target_modules", "bias",
			"peft_version", "revision", "inference_mode", "init_lora_weights", "lora_dropout", "auto_mapping":
			// These are consumed above or have no effect in inference mode.
		case "fan_in_fan_out", "use_dora", "use_rslora", "lora_bias", "use_qalora", "use_bdlora", "ensure_weight_tying",
			"rank_pattern", "alpha_pattern", "modules_to_save", "layers_to_transform", "layers_pattern", "exclude_modules",
			"target_parameters", "trainable_token_indices", "layer_replication", "alora_invocation_tokens",
			"arrow_config", "corda_config", "eva_config", "loftq_config", "lora_ga_config", "megatron_config",
			"monteclora_config", "velora_config":
			var v any
			if err := json.Unmarshal(value, &v); err != nil {
				return c, err
			}
			empty := v == nil || v == false
			switch v := v.(type) {
			case map[string]any:
				empty = len(v) == 0
			case []any:
				empty = len(v) == 0
			}
			if !empty {
				return c, fmt.Errorf("unsupported LoRA option %s", name)
			}
		case "qalora_group_size", "megatron_core":
			// Their corresponding variants must be disabled above.
		default:
			return c, fmt.Errorf("unknown LoRA option %s", name)
		}
	}
	seen := make(map[string]bool)
	for _, target := range c.Targets {
		switch target {
		case "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj":
		default:
			return c, fmt.Errorf("unsupported LoRA target %q", target)
		}
		if seen[target] {
			return c, fmt.Errorf("duplicate LoRA target %q", target)
		}
		seen[target] = true
	}
	return c, nil
}

// loRAScale checks the scalar before it reaches a half-precision graph.
func loRAScale(info LoRAInfo, scale float64) (float32, error) {
	factor := scale * info.Alpha / float64(info.Rank)
	if math.IsNaN(scale) || math.IsInf(scale, 0) || math.IsNaN(factor) || math.IsInf(factor, 0) || math.Abs(factor) > 65504 {
		return 0, fmt.Errorf("LoRA scale must have a finite FP16 alpha/rank multiplier")
	}
	return float32(factor), nil
}

// readLoRA validates the complete tensor set against this architecture and
// copies finite, FP16-representable values from one open file descriptor.
// Tensor names use PEFT's base_model.model prefix; partial layer selections,
// extra tensors, and mismatched ranks fail before any native state changes.
func readLoRA(ctx context.Context, dir string, model StableLMConfig, scale float64) (LoRAInfo, []loRATensor, error) {
	data, err := os.ReadFile(filepath.Join(dir, "adapter_config.json"))
	if err != nil {
		return LoRAInfo{}, nil, err
	}
	c, err := parseLoRAConfig(data)
	if err != nil {
		return LoRAInfo{}, nil, err
	}
	info := LoRAInfo{BaseModel: c.BaseModel, Rank: c.Rank, Alpha: c.Alpha, Scale: scale, Projections: len(c.Targets) * model.NumHiddenLayers}
	if _, err := loRAScale(info, scale); err != nil {
		return LoRAInfo{}, nil, err
	}
	f, err := openSafeFile(filepath.Join(dir, "adapter_model.safetensors"))
	if err != nil {
		return LoRAInfo{}, nil, err
	}
	defer f.f.Close()
	if len(f.tensors) != 2*info.Projections {
		return LoRAInfo{}, nil, fmt.Errorf("LoRA has %d tensors, expected %d", len(f.tensors), 2*info.Projections)
	}
	store := tensorStore{byName: make(map[string]*safeFile)}
	for name := range f.tensors {
		store.byName[name] = f
	}
	var tensors []loRATensor
	for layer := 0; layer < model.NumHiddenLayers; layer++ {
		for _, target := range c.Targets {
			if err := ctx.Err(); err != nil {
				return LoRAInfo{}, nil, err
			}
			group, input, output := "self_attn", model.HiddenSize, model.HiddenSize
			switch target {
			case "k_proj", "v_proj":
				output = model.NumKeyValueHeads * (model.HiddenSize / model.NumAttentionHeads)
			case "gate_proj", "up_proj":
				group, output = "mlp", model.IntermediateSize
			case "down_proj":
				group, input = "mlp", model.IntermediateSize
			}
			if c.Rank > min(input, output) {
				return LoRAInfo{}, nil, fmt.Errorf("LoRA rank %d exceeds %s dimensions", c.Rank, target)
			}
			name := fmt.Sprintf("model.layers.%d.%s.%s", layer, group, target)
			for i, shape := range [][]int{{c.Rank, input}, {output, c.Rank}} {
				suffix := []string{".lora_A.weight", ".lora_B.weight"}[i]
				w, err := store.load("base_model.model."+name+suffix, shape...)
				if err != nil {
					return LoRAInfo{}, nil, err
				}
				values := make([]float32, shape[0]*shape[1])
				for j := range values {
					v := w.at(j)
					if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.Abs(float64(v)) > 65504 {
						return LoRAInfo{}, nil, fmt.Errorf("LoRA tensor %s%s has a non-finite or overflowing value", name, suffix)
					}
					values[j] = v
				}
				info.Bytes += uint64(len(values)) * 2
				tensors = append(tensors, loRATensor{name: name, shape: shape, values: values})
			}
		}
	}
	hash := sha256.New()
	hash.Write(data)
	if _, err = io.Copy(hash, io.NewSectionReader(f.f, 0, math.MaxInt64)); err != nil {
		return LoRAInfo{}, nil, err
	}
	info.SHA256 = fmt.Sprintf("%x", hash.Sum(nil))
	return info, tensors, ctx.Err()
}
