//go:build mlx && darwin && arm64 && cgo

package tinyoai

import (
	"context"
	"fmt"

	mx "github.com/wbrown/tinyoai/internal/mlx"
)

type mlxLoRAPair struct{ a, b mx.Array }

// mlxLoRA owns only adapter handles. Matrix keys borrow the session's base
// weight handles; forks remap those keys and retain independent adapter handles.
type mlxLoRA struct {
	info   LoRAInfo
	factor float32
	pairs  map[mx.Array]mlxLoRAPair
}

// free releases owned adapter handles under the MLX lock, including partial loads.
func (l *mlxLoRA) free() {
	if l == nil {
		return
	}
	for _, pair := range l.pairs {
		pair.a.Free()
		pair.b.Free()
	}
	l.pairs = nil
}

// activeLoRA reports whether a projection needs the low-rank correction.
func (n *nativeMLX) activeLoRA() bool { return n.lora != nil && n.lora.factor != 0 }

// loRAInfo returns metadata while the owning MLX session holds its gate.
func (n *nativeMLX) loRAInfo() LoRAInfo {
	if n.lora == nil {
		return LoRAInfo{}
	}
	return n.lora.info
}

// loadLoRA prepares all tensors before replacing a session's adapter. Errors
// and cancellation before the commit release the new tensors and preserve the
// original adapter, KV, and probabilities. Base weights are never modified.
func (n *nativeMLX) loadLoRA(ctx context.Context, dir string, scale float64) (LoRAInfo, error) {
	var next *mlxLoRA
	var tensors []loRATensor
	if dir != "" {
		info, data, err := readLoRA(ctx, dir, n.config, scale)
		if err != nil {
			return LoRAInfo{}, err
		}
		factor, err := loRAScale(info, scale)
		if err != nil {
			return LoRAInfo{}, err
		}
		next, tensors = &mlxLoRA{info: info, factor: factor, pairs: make(map[mx.Array]mlxLoRAPair)}, data
	}
	committed := false
	err := mx.Run(func() {
		defer func() {
			if !committed {
				next.free()
			}
		}()
		if ctx.Err() != nil {
			return
		}
		a := &mx.Arena{Context: n.ctx}
		defer a.Free()
		var all []mx.Array
		for i := 0; i < len(tensors); i += 2 {
			key, ok := n.weights[tensors[i].name+".weight"]
			if !ok {
				mx.Fail("missing base projection %s", tensors[i].name)
			}
			pair := mlxLoRAPair{}
			pair.a = a.Cast(a.FloatArray(tensors[i].values, tensors[i].shape...), mx.Float16).Retain()
			next.pairs[key] = pair
			pair.b = a.Cast(a.FloatArray(tensors[i+1].values, tensors[i+1].shape...), mx.Float16).Retain()
			next.pairs[key] = pair
			all = append(all, pair.a, pair.b)
		}
		if len(all) > 0 {
			mx.Eval(all...)
		}
		if ctx.Err() != nil {
			return
		}
		n.reset()
		n.lora.free()
		n.lora, committed = next, true
	})
	if err != nil {
		return LoRAInfo{}, err
	}
	if !committed {
		return LoRAInfo{}, ctx.Err()
	}
	return n.loRAInfo(), nil
}

// setLoRAScale retains loaded tensors but invalidates activations computed
// with the old strength. The caller holds the session gate.
func (n *nativeMLX) setLoRAScale(scale float64) (LoRAInfo, error) {
	if n.lora == nil {
		return LoRAInfo{}, fmt.Errorf("no LoRA is loaded")
	}
	factor, err := loRAScale(n.lora.info, scale)
	if err != nil {
		return LoRAInfo{}, err
	}
	if err = mx.Run(func() {
		n.reset()
		n.lora.info.Scale, n.lora.factor = scale, factor
	}); err != nil {
		return LoRAInfo{}, err
	}
	return n.loRAInfo(), nil
}
