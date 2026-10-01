package tinyoai

import (
	"fmt"
	"math"
)

// StableLMOptions selects storage precision. Empty KVCache means float32.
// Float16 is lossy and reduces retained cache memory; computation stays float32.
type StableLMOptions struct {
	KVCache string // "f32" or "f16"
}

// LoadStableLMWithOptions loads a Clio checkpoint with the requested KV precision.
func LoadStableLMWithOptions(path string, opts StableLMOptions) (*StableLM, error) {
	if opts.KVCache != "" && opts.KVCache != "f32" && opts.KVCache != "f16" {
		return nil, fmt.Errorf("unsupported KV cache precision %q: use f32 or f16", opts.KVCache)
	}
	m, err := LoadStableLM(path)
	if err == nil {
		m.halfKV = opts.KVCache == "f16"
	}
	return m, err
}

// floatHalf rounds to nearest, ties to even, including subnormals and overflow.
func floatHalf(value float32) uint16 {
	bits := math.Float32bits(value)
	sign, magnitude := uint16(bits>>16)&0x8000, bits&0x7fffffff
	if magnitude >= 0x7f800000 {
		if magnitude > 0x7f800000 {
			return sign | 0x7e00
		}
		return sign | 0x7c00
	}
	exponent := int(magnitude>>23) - 127 + 15
	if exponent >= 31 {
		return sign | 0x7c00
	}
	if exponent <= 0 {
		if exponent < -10 {
			return sign
		}
		mantissa := magnitude&0x7fffff | 0x800000
		shift := uint(14 - exponent)
		rounded := (mantissa + (1<<(shift-1) - 1) + ((mantissa >> shift) & 1)) >> shift
		return sign | uint16(rounded)
	}
	magnitude += 0xfff + ((magnitude >> 13) & 1)
	return sign | uint16((magnitude>>13)-0x1c000)
}

// storeKV writes one key/value slice at an element offset in a layer, rounding
// to float16 only when that storage mode is enabled.
func (s *stableState) storeKV(l, offset int, key, value []float32) {
	if s.keys16 == nil {
		copy(s.keys[l][offset:], key)
		copy(s.values[l][offset:], value)
		return
	}
	for i, v := range key {
		s.keys16[l][offset+i] = floatHalf(v)
		s.values16[l][offset+i] = floatHalf(value[i])
	}
}

// prepareKV expands the valid prefix of a float16 layer into the shared
// float32 attention workspace. Float32 caches need no conversion. Reusing one
// layer workspace avoids retaining a second full-precision copy of every
// layer.
func (m *StableLM) prepareKV(l, positions int, s *stableState) {
	if s.keys16 == nil {
		return
	}
	head := m.config.HiddenSize / m.config.NumAttentionHeads
	capacity := len(s.att) / m.config.NumAttentionHeads
	parallelRows(m.config.NumKeyValueHeads, positions*m.config.HiddenSize, func(begin, end int) {
		for h := begin; h < end; h++ {
			start, end := h*capacity*head, (h*capacity+positions)*head
			for i := start; i < end; i++ {
				s.keyWorkspace[i] = halfFloat(s.keys16[l][i])
				s.valueWorkspace[i] = halfFloat(s.values16[l][i])
			}
		}
	})
}

// kvLayer returns float32 views for attention. In float16 mode the caller must
// first run prepareKV for l; the returned workspace is overwritten by the next
// prepared layer.
func (s *stableState) kvLayer(l int) ([]float32, []float32) {
	if s.keys16 != nil {
		return s.keyWorkspace, s.valueWorkspace
	}
	return s.keys[l], s.values[l]
}

// growKV replaces each layer buffer with larger head-major storage, preserving
// positions valid entries per head. Copying a flat prefix would use the old
// head stride and corrupt all but the first head.
func growKV[T float32 | uint16](layers [][]T, old, capacity, heads, head, positions int) {
	for l, data := range layers {
		grown := make([]T, capacity*heads*head)
		for h := 0; h < heads; h++ {
			copy(grown[h*capacity*head:], data[h*old*head:h*old*head+positions*head])
		}
		layers[l] = grown
	}
}
