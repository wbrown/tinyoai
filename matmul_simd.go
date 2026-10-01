//go:build goexperiment.simd && go1.27

package tinyoai

import "simd"

// batchWidth returns four vectors of token lanes, the stride alignment
// required by batchDot4.
func batchWidth() int { var v simd.Float32s; return 4 * v.Len() }

// batchDot4 adds a four-row by four-vector product tile to dst. Independent
// accumulators hide multiply-add latency; matmul.go owns packing and
// traversal. The token stride must be a multiple of batchWidth.
func batchDot4(dst []float64, x, w []float32, stride, k int) {
	var z simd.Float32s
	width := z.Len()
	lanes := make([]float32, width)
	w0, w1, w2, w3 := w[:k], w[k:2*k], w[2*k:3*k], w[3*k:4*k]
	for token := 0; token < stride; token += 4 * width {
		var a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, d0, d1, d2, d3 simd.Float32s
		for j := 0; j < k; j++ {
			p := x[j*stride+token : j*stride+token+4*width]
			x0, x1, x2, x3 := simd.LoadFloat32s(p), simd.LoadFloat32s(p[width:]), simd.LoadFloat32s(p[2*width:]), simd.LoadFloat32s(p[3*width:])
			a, b, c, d := simd.BroadcastFloat32s(w0[j]), simd.BroadcastFloat32s(w1[j]), simd.BroadcastFloat32s(w2[j]), simd.BroadcastFloat32s(w3[j])
			a0 = a.MulAdd(x0, a0)
			a1 = a.MulAdd(x1, a1)
			a2 = a.MulAdd(x2, a2)
			a3 = a.MulAdd(x3, a3)
			b0 = b.MulAdd(x0, b0)
			b1 = b.MulAdd(x1, b1)
			b2 = b.MulAdd(x2, b2)
			b3 = b.MulAdd(x3, b3)
			c0 = c.MulAdd(x0, c0)
			c1 = c.MulAdd(x1, c1)
			c2 = c.MulAdd(x2, c2)
			c3 = c.MulAdd(x3, c3)
			d0 = d.MulAdd(x0, d0)
			d1 = d.MulAdd(x1, d1)
			d2 = d.MulAdd(x2, d2)
			d3 = d.MulAdd(x3, d3)
		}
		// Keeping the stores explicit avoids collections of size-agnostic SIMD
		// values, which the experimental package does not support.
		a0.Store(lanes)
		addBatchLanes(dst[token:], lanes)
		a1.Store(lanes)
		addBatchLanes(dst[token+width:], lanes)
		a2.Store(lanes)
		addBatchLanes(dst[token+2*width:], lanes)
		a3.Store(lanes)
		addBatchLanes(dst[token+3*width:], lanes)
		b0.Store(lanes)
		addBatchLanes(dst[stride+token:], lanes)
		b1.Store(lanes)
		addBatchLanes(dst[stride+token+width:], lanes)
		b2.Store(lanes)
		addBatchLanes(dst[stride+token+2*width:], lanes)
		b3.Store(lanes)
		addBatchLanes(dst[stride+token+3*width:], lanes)
		c0.Store(lanes)
		addBatchLanes(dst[2*stride+token:], lanes)
		c1.Store(lanes)
		addBatchLanes(dst[2*stride+token+width:], lanes)
		c2.Store(lanes)
		addBatchLanes(dst[2*stride+token+2*width:], lanes)
		c3.Store(lanes)
		addBatchLanes(dst[2*stride+token+3*width:], lanes)
		d0.Store(lanes)
		addBatchLanes(dst[3*stride+token:], lanes)
		d1.Store(lanes)
		addBatchLanes(dst[3*stride+token+width:], lanes)
		d2.Store(lanes)
		addBatchLanes(dst[3*stride+token+2*width:], lanes)
		d3.Store(lanes)
		addBatchLanes(dst[3*stride+token+3*width:], lanes)
	}
}

// addBatchLanes widens a completed vector reduction and adds its lanes to the
// corresponding float64 totals.
func addBatchLanes(dst []float64, lanes []float32) {
	for i, v := range lanes {
		dst[i] += float64(v)
	}
}
