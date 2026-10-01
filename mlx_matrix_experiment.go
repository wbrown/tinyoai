//go:build mlx && darwin && arm64 && cgo

package tinyoai

import (
	"context"
	"fmt"
	mx "github.com/wbrown/tinyoai/internal/mlx"
)

// Matrix evaluates a projection with deterministic FP16 inputs and an optional
// one-element offset. The runner selects kernel overrides and compares outputs;
// this method owns only graph construction and evaluation.
func (b *mlxBenchmark) Matrix(ctx context.Context, name string, rows, offset int) (values []float32, err error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	n := b.n
	w, width := n.layers[0].q, n.config.HiddenSize
	switch name {
	case "attention":
	case "up":
		w = n.layers[0].up
	case "down":
		w, width = n.layers[0].down, n.config.IntermediateSize
	default:
		return nil, fmt.Errorf("unknown projection %q", name)
	}
	err = mx.Run(func() {
		a := &mx.Arena{Context: n.ctx}
		defer a.Free()
		data := make([]int, rows*width+offset)
		for i := range data {
			data[i] = (i*193+7919)%2003 - 1001
		}
		input := a.Scale(a.Cast(a.Tokens(data), mx.Float16), 0.001)
		input = a.Reshape(a.Slice(input, 1, offset, len(data)), 1, rows, width)
		mx.Eval(input)
		y := a.Contiguous(a.Cast(n.linear(a, input, w), mx.Float32))
		mx.Eval(y)
		values = y.Floats()
	})
	return
}
