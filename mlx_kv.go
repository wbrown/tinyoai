//go:build mlx && darwin && arm64 && cgo

package tinyoai

import (
	"fmt"
	mx "github.com/wbrown/tinyoai/internal/mlx"
)

// Q8 uses affine groups of 64: packed uint32 data and FP16 scale/bias pairs.
// Prefill reconstructs one layer for flash attention. Decode consumes packed KV.
type mlxKV struct{ data, scales, biases mx.Array }

// arrays returns borrowed handles for the populated dense cache or its packed
// data, scales, and biases.
func (c mlxKV) arrays() []mx.Array {
	if !c.data.Valid() {
		return nil
	}
	if c.scales.Valid() {
		return []mx.Array{c.data, c.scales, c.biases}
	}
	return []mx.Array{c.data}
}

// Free releases the handles owned by this cache under the MLX lock. It does
// not clear copies of the value; replace the cache before reusing it.
func (c mlxKV) Free() {
	for _, a := range c.arrays() {
		a.Free()
	}
}

// Bytes sums logical storage bytes, including quantization scales and biases.
func (c mlxKV) Bytes() uint64 {
	var size uint64
	for _, a := range c.arrays() {
		size += a.Bytes()
	}
	return size
}

// update replaces cache positions beginning at prefix, growing or reserving
// backing storage as needed. The receiver owns retained capacity; the returned
// arena views cover only the populated prefix and new tokens.
func (c *mlxKV) update(a *mx.Arena, x mx.Array, prefix, bits int, reserve ...int) mlxKV {
	if bits == 16 {
		return mlxKV{data: updateMLXCache(a, &c.data, x, prefix, reserve...)}
	}
	q := a.Quantize(x, 64, 8)
	data := updateMLXCache(a, &c.data, q[0], prefix, reserve...)
	scales := updateMLXCache(a, &c.scales, q[1], prefix, reserve...)
	biases := updateMLXCache(a, &c.biases, q[2], prefix, reserve...)
	return mlxKV{data, scales, biases}
}

// dense returns the dense cache handle or an arena-owned reconstruction of
// affine-Q8 values.
func (c mlxKV) dense(a *mx.Arena) mx.Array {
	if !c.scales.Valid() {
		return c.data
	}
	return a.Dequantize(c.data, c.scales, c.biases, 64, 8)
}

// quantizedDecodeAttention computes a single query against packed affine-Q8
// keys and values with precise softmax accumulation. All cached positions are
// visible, so no causal mask is needed.
func quantizedDecodeAttention(a *mx.Arena, q mx.Array, k, v mlxKV, scale float64) mx.Array {
	q = a.Scale(q, float32(scale))
	scores := a.QuantizedMatmul(q, k.data, k.scales, k.biases, 64, 8, true)
	return a.QuantizedMatmul(a.Softmax(scores), v.data, v.scales, v.biases, 64, 8, false)
}

// SetKVBits selects 8- or 16-bit KV storage. A precision change discards the
// cached prefix and idle allocations while leaving weights resident; cleared
// reports that invalidation.
func (n *nativeMLX) SetKVBits(bits int) (cleared bool, err error) {
	if bits != 8 && bits != 16 {
		return false, fmt.Errorf("KV bits must be 8 or 16")
	}
	err = mx.Run(func() {
		if n.kvBits != bits {
			n.reset()
			mx.ClearCache()
			n.kvBits = bits
			cleared = true
		}
	})
	return
}
