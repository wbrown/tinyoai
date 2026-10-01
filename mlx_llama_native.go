//go:build mlx && darwin && arm64 && cgo

package tinyoai

import (
	"context"
	"fmt"
	"math"

	mx "github.com/wbrown/tinyoai/internal/mlx"
)

type llamaMLXLayer struct {
	attNorm, ffnNorm, q, k, v, o, gate, up, down mx.Array
	keys, values                                 mx.Array
}

type nativeLlamaMLX struct {
	ctx                     *mx.Context
	config                  Config
	layers                  []llamaMLXLayer
	embed, norm, real, imag mx.Array
	weights                 []mx.Array
	offset                  int
}

// newLlamaMLX uploads a legacy model's float32 weights and rotary tables to
// MLX and creates an independent request gate and KV cache. Partial native
// construction is released on failure.
func newLlamaMLX(m *Model) (*LlamaMLX, error) {
	n := &nativeLlamaMLX{config: m.config}
	err := mx.Run(func() {
		n.ctx = mx.New()
		a := &mx.Arena{Context: n.ctx}
		defer a.Free()
		weight := func(data []float32, shape ...int) mx.Array {
			v := a.FloatArray(data, shape...).Retain()
			n.weights = append(n.weights, v)
			return v
		}
		c, w := m.config, m.weights
		d, h, head := int(c.Dim), int(c.HiddenDim), int(c.Dim/c.NHeads)
		kv := head * int(c.NKvHeads)
		n.embed = weight(w.tokenEmbeddingTable, int(c.VocabSize), d)
		n.norm = weight(w.rmsFinalWeight, d)
		n.real = weight(w.freqCisReal, int(c.SeqLen), head/2)
		n.imag = weight(w.freqCisImag, int(c.SeqLen), head/2)
		for i := 0; i < int(c.NLayers); i++ {
			layerWeight := func(data []float32, rows, cols int) mx.Array {
				size := rows * cols
				return weight(data[i*size:(i+1)*size], rows, cols)
			}
			n.layers = append(n.layers, llamaMLXLayer{
				attNorm: weight(w.rmsAttWeight[i*d:(i+1)*d], d),
				ffnNorm: weight(w.rmsFfnWeight[i*d:(i+1)*d], d),
				q:       layerWeight(w.wq, d, d), k: layerWeight(w.wk, kv, d), v: layerWeight(w.wv, kv, d),
				o: layerWeight(w.wo, d, d), gate: layerWeight(w.w1, h, d),
				up: layerWeight(w.w3, h, d), down: layerWeight(w.w2, d, h),
			})
		}
		mx.Eval(n.weights...)
	})
	if err != nil {
		_ = n.Close()
		return nil, err
	}
	result := &LlamaMLX{model: m, native: n, gate: make(chan struct{}, 1), window: int(m.config.SeqLen)}
	result.gate <- struct{}{}
	return result, nil
}

// reset releases per-layer KV and clears the valid prefix length. Weight
// arrays remain resident; the caller holds the MLX lock.
func (n *nativeLlamaMLX) reset() {
	for i := range n.layers {
		l := &n.layers[i]
		l.keys.Free()
		l.values.Free()
		l.keys, l.values = mx.Array{}, mx.Array{}
	}
	n.offset = 0
}

// Reset discards cached KV under the MLX lock while retaining model weights.
func (n *nativeLlamaMLX) Reset() error { return mx.Run(n.reset) }

// Close releases KV, weights, and native streams under the MLX lock.
func (n *nativeLlamaMLX) Close() error {
	return mx.Run(func() {
		n.reset()
		for _, w := range n.weights {
			w.Free()
		}
		n.weights = nil
		n.ctx.Close()
		n.ctx = nil
	})
}

// llamaMLXRotate rotates adjacent coordinate pairs using the checkpoint's
// stored sine and cosine tables. StableLM's split-half rotary layout is
// different and cannot be substituted.
func llamaMLXRotate(a *mx.Arena, x, real, imag mx.Array) mx.Array {
	shape := x.Shape()
	pairs := a.Reshape(x, shape[0], shape[1], shape[2], shape[3]/2, 2)
	even, odd := a.Slice(pairs, 4, 0, 1), a.Slice(pairs, 4, 1, 2)
	r := a.Subtract(a.Multiply(even, real), a.Multiply(odd, imag))
	i := a.Add(a.Multiply(even, imag), a.Multiply(odd, real))
	return a.Reshape(a.Concat(r, i, 4), shape...)
}

// Forward replaces the cached suffix after prefix with tokens and returns
// either every logit row or the last row. All weights, activations, and KV use
// float32.
//
// Cancellation is checked before evaluation; an admitted chunk finishes and
// returns its work so the caller can record reusable KV. Native errors reset
// KV. The caller holds the model gate.
func (n *nativeLlamaMLX) Forward(ctx context.Context, prefix int, tokens []int, allLogits bool) (logits []float32, err error) {
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	c := n.config
	if prefix < 0 || prefix > n.offset || len(tokens) == 0 || prefix+len(tokens) > int(c.SeqLen) {
		return nil, fmt.Errorf("invalid Llama MLX prefix or context length")
	}
	for _, token := range tokens {
		if token < 0 || token >= int(c.VocabSize) {
			return nil, fmt.Errorf("invalid token %d", token)
		}
	}
	err = mx.Run(func() {
		a := &mx.Arena{Context: n.ctx}
		defer a.Free()
		length, d, heads, kvHeads := len(tokens), int(c.Dim), int(c.NHeads), int(c.NKvHeads)
		head := d / heads
		real := a.Reshape(a.Slice(n.real, 0, prefix, prefix+length), 1, 1, length, head/2, 1)
		imag := a.Reshape(a.Slice(n.imag, 0, prefix, prefix+length), 1, 1, length, head/2, 1)
		linear := func(x, weight mx.Array) mx.Array { return a.Matmul(x, a.Transpose(weight, 1, 0)) }
		x := a.Take(n.embed, a.Tokens(tokens))
		for i := range n.layers {
			l := &n.layers[i]
			h := a.RMSNorm(x, l.attNorm, 1e-5)
			q := a.Transpose(a.Reshape(linear(h, l.q), 1, length, heads, head), 0, 2, 1, 3)
			k := a.Transpose(a.Reshape(linear(h, l.k), 1, length, kvHeads, head), 0, 2, 1, 3)
			v := a.Transpose(a.Reshape(linear(h, l.v), 1, length, kvHeads, head), 0, 2, 1, 3)
			q, k = llamaMLXRotate(a, q, real, imag), llamaMLXRotate(a, k, real, imag)
			keys := updateMLXCache(a, &l.keys, k, prefix)
			values := updateMLXCache(a, &l.values, v, prefix)
			att := a.Attention(q, keys, values, 1/math.Sqrt(float64(head)), length > 1)
			x = a.Add(x, linear(a.Reshape(a.Transpose(att, 0, 2, 1, 3), 1, length, d), l.o))
			h = a.RMSNorm(x, l.ffnNorm, 1e-5)
			x = a.Add(x, linear(a.SwiGLU(linear(h, l.gate), linear(h, l.up)), l.down))
		}
		if !allLogits {
			x = a.Slice(x, 1, length-1, length)
		}
		out := a.Contiguous(linear(a.RMSNorm(x, n.norm, 1e-5), n.embed)).Retain()
		defer out.Free()
		live := []mx.Array{out}
		for _, l := range n.layers {
			live = append(live, l.keys, l.values)
		}
		a.Free()
		mx.Eval(live...)
		logits = out.Floats()
		n.offset = prefix + length
	})
	if err != nil {
		_ = n.Reset()
	}
	// Return completed work even if cancellation arrived during GPU evaluation.
	return logits, err
}
