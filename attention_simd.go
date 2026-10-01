//go:build goexperiment.simd && go1.27

package tinyoai

import "simd"

// attentionBatchWidth selects four queries when SIMD runs on hardware, or one
// when vectors are emulated.
func attentionBatchWidth() int {
	if simd.Emulated() {
		return 1
	}
	return 4
}

// attentionScores4 computes four query/key dot products per key load for a
// shared prefix of count positions. Query rows are d elements apart; score
// rows are stride elements apart. The caller completes the small causal
// triangle and applies softmax afterward.
func attentionScores4(q, keys, scores []float32, d, stride, head, count int) {
	var zero simd.Float32s
	width := zero.Len()
	lanes := make([]float32, width)
	scale := attentionScale(head)
	q0, q1, q2, q3 := q[:head], q[d:d+head], q[2*d:2*d+head], q[3*d:3*d+head]
	for t := 0; t < count; t++ {
		key := keys[t*head : (t+1)*head]
		var a, b, c, e simd.Float32s
		j := 0
		for ; j+width <= head; j += width {
			k := simd.LoadFloat32s(key[j : j+width])
			a = simd.LoadFloat32s(q0[j:]).MulAdd(k, a)
			b = simd.LoadFloat32s(q1[j:]).MulAdd(k, b)
			c = simd.LoadFloat32s(q2[j:]).MulAdd(k, c)
			e = simd.LoadFloat32s(q3[j:]).MulAdd(k, e)
		}
		var ta, tb, tc, te float64
		a.Store(lanes)
		for _, v := range lanes {
			ta += float64(v)
		}
		b.Store(lanes)
		for _, v := range lanes {
			tb += float64(v)
		}
		c.Store(lanes)
		for _, v := range lanes {
			tc += float64(v)
		}
		e.Store(lanes)
		for _, v := range lanes {
			te += float64(v)
		}
		for ; j < head; j++ {
			k := float64(key[j])
			ta += float64(q0[j]) * k
			tb += float64(q1[j]) * k
			tc += float64(q2[j]) * k
			te += float64(q3[j]) * k
		}
		scores[t], scores[stride+t], scores[2*stride+t], scores[3*stride+t] = float32(ta)/scale, float32(tb)/scale, float32(tc)/scale, float32(te)/scale
	}
}

// attentionValues4 writes four attention outputs, keeping output channels in
// registers while sharing each value load. Score rows must have zero weight
// beyond each query's causal prefix. The history traversal order is preserved
// within every sum.
func attentionValues4(out, values, scores []float32, d, stride, head, count int) {
	var zero simd.Float32s
	width := zero.Len()
	s0, s1, s2, s3 := scores[:count], scores[stride:stride+count], scores[2*stride:2*stride+count], scores[3*stride:3*stride+count]
	j := 0
	for ; j+4*width <= head; j += 4 * width {
		var a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, d0, d1, d2, d3 simd.Float32s
		for t := 0; t < count; t++ {
			v := values[t*head+j : t*head+j+4*width]
			v0, v1, v2, v3 := simd.LoadFloat32s(v), simd.LoadFloat32s(v[width:]), simd.LoadFloat32s(v[2*width:]), simd.LoadFloat32s(v[3*width:])
			a, b, c, e := simd.BroadcastFloat32s(s0[t]), simd.BroadcastFloat32s(s1[t]), simd.BroadcastFloat32s(s2[t]), simd.BroadcastFloat32s(s3[t])
			a0 = a.MulAdd(v0, a0)
			a1 = a.MulAdd(v1, a1)
			a2 = a.MulAdd(v2, a2)
			a3 = a.MulAdd(v3, a3)
			b0 = b.MulAdd(v0, b0)
			b1 = b.MulAdd(v1, b1)
			b2 = b.MulAdd(v2, b2)
			b3 = b.MulAdd(v3, b3)
			c0 = c.MulAdd(v0, c0)
			c1 = c.MulAdd(v1, c1)
			c2 = c.MulAdd(v2, c2)
			c3 = c.MulAdd(v3, c3)
			d0 = e.MulAdd(v0, d0)
			d1 = e.MulAdd(v1, d1)
			d2 = e.MulAdd(v2, d2)
			d3 = e.MulAdd(v3, d3)
		}
		a0.Store(out[0*d+j+0*width:])
		a1.Store(out[0*d+j+1*width:])
		a2.Store(out[0*d+j+2*width:])
		a3.Store(out[0*d+j+3*width:])
		b0.Store(out[1*d+j+0*width:])
		b1.Store(out[1*d+j+1*width:])
		b2.Store(out[1*d+j+2*width:])
		b3.Store(out[1*d+j+3*width:])
		c0.Store(out[2*d+j+0*width:])
		c1.Store(out[2*d+j+1*width:])
		c2.Store(out[2*d+j+2*width:])
		c3.Store(out[2*d+j+3*width:])
		d0.Store(out[3*d+j+0*width:])
		d1.Store(out[3*d+j+1*width:])
		d2.Store(out[3*d+j+2*width:])
		d3.Store(out[3*d+j+3*width:])
	}
	for ; j < head; j++ {
		var a, b, c, e float32
		for t := 0; t < count; t++ {
			v := values[t*head+j]
			a += s0[t] * v
			b += s1[t] * v
			c += s2[t] * v
			e += s3[t] * v
		}
		out[j], out[d+j], out[2*d+j], out[3*d+j] = a, b, c, e
	}
}
