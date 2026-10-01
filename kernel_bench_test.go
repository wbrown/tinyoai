package tinyoai

import (
	"fmt"
	"testing"
)

// BenchmarkClioProjection measures representative attention and MLP
// projections with identical inputs and excluded setup. Use -cpu to sweep
// worker counts.
func BenchmarkClioProjection(b *testing.B) {
	for _, dtype := range []string{"BF16", "Q8_0"} {
		for _, shape := range [][2]int{{2816, 2816}, {7552, 2816}, {2816, 7552}} {
			rows, cols := shape[0], shape[1]
			w := matrixFixture(rows, cols, dtype)
			for _, n := range []int{1, 32, 64, 128} {
				b.Run(fmt.Sprintf("%s/%dx%d/n%d", dtype, rows, cols, n), func(b *testing.B) {
					x, out := make([]float32, n*cols), make([]float32, n*rows)
					for i := range x {
						x[i] = float32(i%17-8) / 9
					}
					stride := (n + batchWidth() - 1) / batchWidth() * batchWidth()
					packed := make([]float32, cols*stride)
					w.mulBatch(out, x, n, packed)
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						w.mulBatch(out, x, n, packed)
					}
					b.ReportMetric(float64(n)*float64(rows)*float64(cols)*2*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOP/s")
				})
			}
		}
	}
}
