//go:build !goexperiment.simd || !go1.27

package tinyoai

type matvecInput struct{ values []float32 }

// prepareMatvec borrows x directly for scalar multiplication; no packed
// activation scratch is needed.
func prepareMatvec(x []float32, _ string, _ []float32) matvecInput { return matvecInput{values: x} }

// mulRows evaluates the assigned matrix rows with the scalar
// float64-accumulating kernel.
func (w weightTensor) mulRows(out []float32, input matvecInput, start, end int) {
	w.mulRowsScalar(out, input.values, start, end)
}

// numericalBackend identifies the scalar reduction path in numerical and
// performance reports.
func numericalBackend() string { return "scalar-float64" }

// attentionDot returns a sequential float32 query/key dot product. The slices
// must have matching lengths.
func attentionDot(q, k []float32) float32 {
	var score float32
	for i, v := range q {
		score += v * k[i]
	}
	return score
}

// attentionAdd accumulates score*values into out in place, preserving the
// history order chosen by the caller.
func attentionAdd(out, values []float32, score float32) {
	for i := range out {
		out[i] += score * values[i]
	}
}
