//go:build mlx && darwin && arm64 && cgo

package tinyoai

import (
	"math"
	"time"

	mx "github.com/wbrown/tinyoai/internal/mlx"
)

// profileLayer duplicates the simple unfused forward only for diagnostics.
// Each boundary explicitly keeps the residual and other live inputs, making
// temporary lifetime comparable to the separately tested layer-batch option.
func (n *nativeMLX) profileLayer(a *mx.Arena, x mx.Array, layer, prefix, length int) mx.Array {
	start := time.Now()
	phase := func(name string, keep ...mx.Array) []mx.Array {
		kept := a.Materialize(keep...)
		n.onPrefillPhase(MLXPrefillPhase{Prefix: prefix, Tokens: length, Layer: layer, Name: name, Seconds: time.Since(start).Seconds()})
		start = time.Now()
		return kept
	}
	x = phase("input_ready", x)[0]
	l, c := &n.layers[layer], n.config
	dim := c.HiddenSize / c.NumAttentionHeads
	h := a.Norm(x, l.norm, l.bias, c.LayerNormEps)
	q, k, v := n.linear(a, h, l.q), n.linear(a, h, l.k), n.linear(a, h, l.v)
	q = a.Transpose(a.Reshape(q, 1, length, c.NumAttentionHeads, dim), 0, 2, 1, 3)
	k = a.Transpose(a.Reshape(k, 1, length, c.NumKeyValueHeads, dim), 0, 2, 1, 3)
	v = a.Transpose(a.Reshape(v, 1, length, c.NumKeyValueHeads, dim), 0, 2, 1, 3)
	q = a.Rope(q, int(float64(dim)*c.PartialRotaryFactor), prefix, c.RopeTheta)
	k = a.Rope(k, int(float64(dim)*c.PartialRotaryFactor), prefix, c.RopeTheta)
	kept := phase("norm_qkv_rope", x, h, q, k, v)
	x, h, q, k, v = kept[0], kept[1], kept[2], kept[3], kept[4]
	ck := l.keys.update(a, k, prefix, n.kvBits, n.reserveTokens)
	cv := l.values.update(a, v, prefix, n.kvBits, n.reserveTokens)
	keep := append([]mx.Array{x, h, q}, ck.arrays()...)
	keep = append(keep, cv.arrays()...)
	kept = phase("kv_quantize_store", keep...)
	x, h, q = kept[0], kept[1], kept[2]
	if n.kvBits == 8 {
		ck = mlxKV{kept[3], kept[4], kept[5]}
		cv = mlxKV{kept[6], kept[7], kept[8]}
	} else {
		ck, cv = mlxKV{data: kept[3]}, mlxKV{data: kept[4]}
	}
	k, v = ck.dense(a), cv.dense(a)
	kept = phase("kv_expand", x, h, q, k, v)
	x, h, q, k, v = kept[0], kept[1], kept[2], kept[3], kept[4]
	attn := a.Attention(q, k, v, math.Sqrt(1/float64(dim)), true)
	kept = phase("attention", x, h, attn)
	x, h, attn = kept[0], kept[1], kept[2]
	r := n.linear(a, a.Reshape(a.Transpose(attn, 0, 2, 1, 3), 1, length, c.HiddenSize), l.o)
	kept = phase("attention_output", x, h, r)
	x, h, r = kept[0], kept[1], kept[2]
	gate, up := n.linear(a, h, l.gate), n.linear(a, h, l.up)
	mlp := n.linear(a, a.SwiGLU(gate, up), l.down)
	kept = phase("mlp", x, r, mlp)
	x, r, mlp = kept[0], kept[1], kept[2]
	return phase("residual", a.Add(a.Add(x, r), mlp))[0]
}
