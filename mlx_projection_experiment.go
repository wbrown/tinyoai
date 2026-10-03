//go:build mlx && darwin && arm64 && cgo

package tinyoai

import (
	"fmt"
	mx "github.com/wbrown/tinyoai/internal/mlx"
)

// packProjections joins output rows without requantizing weights. The original
// named matrices become views of the packed storage, so branches and unfused
// forwards remain available without keeping a second copy of the weights.
// Packing is an opt-in experiment and is excluded from inference timings.
func (n *nativeMLX) packProjections() error {
	if n.projectionsPacked {
		return nil
	}
	if n.lora != nil {
		return fmt.Errorf("pack base projections before loading a LoRA")
	}
	return mx.Run(func() {
		for i := range n.layers {
			l := &n.layers[i]
			prefix := fmt.Sprintf("model.layers.%d.", i)
			l.qkv = n.packMatrices(prefix+"self_attn.qkv", []string{prefix + "self_attn.q_proj", prefix + "self_attn.k_proj", prefix + "self_attn.v_proj"})
			l.gateUp = n.packMatrices(prefix+"mlp.gate_up", []string{prefix + "mlp.gate_proj", prefix + "mlp.up_proj"})
			head := n.config.HiddenSize / n.config.NumAttentionHeads
			l.q = n.matrix(prefix+"self_attn.q_proj", n.config.HiddenSize, n.config.HiddenSize)
			l.k = n.matrix(prefix+"self_attn.k_proj", n.config.NumKeyValueHeads*head, n.config.HiddenSize)
			l.v = n.matrix(prefix+"self_attn.v_proj", n.config.NumKeyValueHeads*head, n.config.HiddenSize)
			l.gate = n.matrix(prefix+"mlp.gate_proj", n.config.IntermediateSize, n.config.HiddenSize)
			l.up = n.matrix(prefix+"mlp.up_proj", n.config.IntermediateSize, n.config.HiddenSize)
		}
		n.projectionsPacked = true
		mx.ClearCache()
	})
}

// packMatrices concatenates matching matrix components along output rows and
// replaces original weights with retained views of that storage. It requires
// the MLX lock and preserves quantized bits without requantization.
func (n *nativeMLX) packMatrices(name string, parts []string) mlxMatrix {
	var packed [3]mx.Array
	suffixes := []string{".weight"}
	if n.bits != 16 {
		suffixes = append(suffixes, ".scales", ".biases")
	}
	for j, suffix := range suffixes {
		func() {
			a := &mx.Arena{Context: n.ctx}
			defer a.Free()
			joined := n.weights[parts[0]+suffix]
			for _, p := range parts[1:] {
				joined = a.Concat(joined, n.weights[p+suffix], 0)
			}
			mx.Eval(joined)
			packed[j] = joined.Retain()
			n.weights[name+suffix] = packed[j]
			pos := 0
			for _, p := range parts {
				old := n.weights[p+suffix]
				end := pos + old.Shape()[0]
				view := a.Slice(joined, 0, pos, end)
				mx.Eval(view)
				n.weights[p+suffix] = view.Retain()
				old.Free()
				pos = end
			}
		}()
	}
	return mlxMatrix{packed[0], packed[1], packed[2]}
}
