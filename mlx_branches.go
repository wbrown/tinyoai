//go:build mlx && darwin && arm64 && cgo

package tinyoai

import (
	"context"
	"fmt"
	"math"

	mx "github.com/wbrown/tinyoai/internal/mlx"
)

// The caller holds MLX.gate throughout this session. The resident prefix is
// borrowed read-only; only the short [branch, head, step, dim] suffix is owned.
type nativeBranches struct {
	n                   *nativeMLX
	prefix, count, step int
	layers              []struct{ keys, values mlxKV }
	closed              bool
}

// NewBranches opens a batch of independent suffixes borrowing a resident
// prefix. The caller must keep the model gate for the entire session;
// grouped-query attention is not supported here.
func (n *nativeMLX) NewBranches(prefix, count int) (mlxBranches, error) {
	if prefix < 0 || prefix > n.offset || count < 1 || count > n.config.VocabSize || n.config.NumAttentionHeads != n.config.NumKeyValueHeads {
		return nil, fmt.Errorf("invalid MLX branch prefix or batch size")
	}
	b := &nativeBranches{n: n, prefix: prefix, count: count}
	b.layers = make([]struct{ keys, values mlxKV }, len(n.layers))
	return b, nil
}

// Close releases the owned suffix caches under the MLX lock. The shared
// document prefix remains intact; repeated calls are harmless.
func (b *nativeBranches) Close() error {
	return mx.Run(func() {
		for i := range b.layers {
			b.layers[i].keys.Free()
			b.layers[i].values.Free()
		}
		b.layers = nil
		b.closed = true
	})
}

// appendBranchKV appends one step to a branch-owned cache, optionally
// quantizing to affine Q8. It retains the replacement handles before releasing
// the old ones and allocates only the suffix length actually used.
func appendBranchKV(a *mx.Arena, cache *mlxKV, x mx.Array, bits int) mlxKV {
	parts := []mx.Array{x}
	if bits == 8 {
		q := a.Quantize(x, 64, 8)
		parts = q[:]
	}
	old := cache.arrays()
	for i := range parts {
		if len(old) > 0 {
			parts[i] = a.Concat(old[i], parts[i], 2)
		}
		parts[i] = parts[i].Retain()
	}
	cache.Free()
	*cache = mlxKV{data: parts[0]}
	if bits == 8 {
		cache.scales, cache.biases = parts[1], parts[2]
	}
	return *cache
}

// branchKVSlice borrows the first end positions of dense or quantized KV into
// arena-owned views.
func branchKVSlice(a *mx.Arena, c mlxKV, end int) mlxKV {
	c.data = a.Slice(c.data, 2, 0, end)
	if c.scales.Valid() {
		c.scales = a.Slice(c.scales, 2, 0, end)
		c.biases = a.Slice(c.biases, 2, 0, end)
	}
	return c
}

// branchKVMatmul multiplies against dense or affine-Q8 KV, optionally
// transposing the cache's final two axes.
func branchKVMatmul(a *mx.Arena, x mx.Array, c mlxKV, transpose bool) mx.Array {
	if c.scales.Valid() {
		return a.QuantizedMatmul(x, c.data, c.scales, c.biases, 64, 8, transpose)
	}
	w := c.data
	if transpose {
		w = a.Transpose(w, 0, 1, 3, 2)
	}
	return a.Matmul(x, w)
}

// branchAttention combines a shared prefix with branch-private suffixes under
// one softmax. Branches become query rows for the prefix multiplication,
// avoiding a copy of the long prefix for each branch.
func branchAttention(a *mx.Arena, q mx.Array, prefixK, prefixV, suffixK, suffixV mlxKV, prefix int, scale float64) mx.Array {
	q = a.Scale(q, float32(scale))
	suffixScores := branchKVMatmul(a, q, suffixK, true)
	if prefix == 0 {
		return branchKVMatmul(a, a.Softmax(suffixScores), suffixV, false)
	}
	// Treat branches as query rows for the shared prefix, avoiding broadcasting
	// (and potentially copying) the long KV tensors across the batch dimension.
	sharedQ := a.Transpose(q, 2, 1, 0, 3)
	prefixScores := a.Transpose(branchKVMatmul(a, sharedQ, branchKVSlice(a, prefixK, prefix), true), 2, 1, 0, 3)
	weights := a.Softmax(a.Concat(prefixScores, suffixScores, 3))
	wp := a.Transpose(a.Slice(weights, 3, 0, prefix), 2, 1, 0, 3)
	ws := a.Slice(weights, 3, prefix, weights.Shape()[3])
	p := a.Transpose(branchKVMatmul(a, wp, branchKVSlice(a, prefixV, prefix), false), 2, 1, 0, 3)
	s := branchKVMatmul(a, ws, suffixV, false)
	return a.Add(p, s)
}

// Forward evaluates one token per branch at the same absolute position and
// returns one full-vocabulary logit row per branch. Evaluation errors close
// the suffix caches; cancellation is observed before and after the native
// evaluation.
func (b *nativeBranches) Forward(ctx context.Context, tokens []int) (rows [][]float32, err error) {
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	n := b.n
	if b.closed || len(tokens) != b.count || b.prefix+b.step >= n.config.MaxPositionEmbeddings {
		return nil, fmt.Errorf("invalid branch batch or context length")
	}
	for _, id := range tokens {
		if id < 0 || id >= n.config.VocabSize {
			return nil, fmt.Errorf("invalid branch token")
		}
	}
	err = mx.Run(func() {
		a := &mx.Arena{Context: n.ctx}
		defer a.Free()
		c := n.config
		dim := c.HiddenSize / c.NumAttentionHeads
		ids := a.Reshape(a.Tokens(tokens), b.count, 1)
		e := n.embed
		x := a.Take(e.weight, ids)
		if n.bits != 16 {
			x = a.Dequantize(x, a.Take(e.scales, ids), a.Take(e.biases, ids), n.group, n.bits)
		}
		for i := range n.layers {
			l, s := &n.layers[i], &b.layers[i]
			h := a.Norm(x, l.norm, l.bias, c.LayerNormEps)
			shape := func(w mlxMatrix) mx.Array {
				return a.Transpose(a.Reshape(n.linear(a, h, w), b.count, 1, c.NumAttentionHeads, dim), 0, 2, 1, 3)
			}
			q, k, v := shape(l.q), shape(l.k), shape(l.v)
			q = a.Rope(q, int(float64(dim)*c.PartialRotaryFactor), b.prefix+b.step, c.RopeTheta)
			k = a.Rope(k, int(float64(dim)*c.PartialRotaryFactor), b.prefix+b.step, c.RopeTheta)
			keys := appendBranchKV(a, &s.keys, k, n.kvBits)
			values := appendBranchKV(a, &s.values, v, n.kvBits)
			attn := branchAttention(a, q, l.keys, l.values, keys, values, b.prefix, math.Sqrt(1/float64(dim)))
			r := n.linear(a, a.Reshape(a.Transpose(attn, 0, 2, 1, 3), b.count, 1, c.HiddenSize), l.o)
			mlp := n.linear(a, a.SwiGLU(n.linear(a, h, l.gate), n.linear(a, h, l.up)), l.down)
			x = a.Add(a.Add(x, r), mlp)
		}
		x = a.Norm(x, n.norm, n.bias, c.LayerNormEps)
		output := a.Contiguous(a.Cast(n.linear(a, x, n.head), mx.Float32)).Retain()
		defer output.Free()
		all := []mx.Array{output}
		for _, l := range b.layers {
			all = append(all, l.keys.arrays()...)
			all = append(all, l.values.arrays()...)
		}
		a.Free()
		mx.Eval(all...)
		flat := output.Floats()
		for i := 0; i < b.count; i++ {
			rows = append(rows, flat[i*c.VocabSize:(i+1)*c.VocabSize])
		}
		b.step++
	})
	if err != nil {
		_ = b.Close()
		return nil, err
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	return rows, nil
}
