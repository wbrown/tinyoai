//go:build !goexperiment.simd || !go1.27

package tinyoai

// batchWidth returns the token-stride alignment used by the scalar four-row
// prefill kernel.
func batchWidth() int { return 4 }

// batchDot4 adds four weight-row products to dst. Activations are column-major
// with stride token lanes, and w contains four contiguous k-element rows.
// Short float32 sums are promoted into the caller's float64 totals.
func batchDot4(dst []float64, x, w []float32, stride, k int) {
	for token := 0; token < stride; token++ {
		var a, b, c, d float32
		for j := 0; j < k; j++ {
			v := x[j*stride+token]
			a += v * w[j]
			b += v * w[k+j]
			c += v * w[2*k+j]
			d += v * w[3*k+j]
		}
		dst[token] += float64(a)
		dst[stride+token] += float64(b)
		dst[2*stride+token] += float64(c)
		dst[3*stride+token] += float64(d)
	}
}
