//go:build goexperiment.simd && go1.27

package tinyoai

import "testing"

// BenchmarkQ8Decode compares direct SIMD byte extraction with tiled Q8_0
// decoding when the selected SIMD width supports both paths.
func BenchmarkQ8Decode(b *testing.B) {
	w := matrixFixture(2816, 2816, "Q8_0")
	x, out, scratch := make([]float32, 2816), make([]float32, 2816), make([]float32, 2816)
	for i := range x {
		x[i] = float32(i%17-8) / 9
	}
	input := prepareMatvec(x, w.dtype, scratch)
	if input.quad[0] == nil {
		b.Skip("direct byte extraction uses 128/256-bit SIMD")
	}
	for _, direct := range []bool{false, true} {
		name := "tiled"
		if direct {
			name = "direct"
		}
		b.Run(name, func(b *testing.B) {
			in := input
			if !direct {
				in.quad = [4][]float32{}
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				parallelRows(2816, 2816*2816, func(start, end int) { mulRowsSIMD(w, out, in, start, end) })
			}
		})
	}
}
