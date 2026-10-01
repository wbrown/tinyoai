//go:build !goexperiment.simd || !go1.27

package tinyoai

// attentionBatchWidth returns one, selecting independent queries in the scalar
// attention dispatcher.
func attentionBatchWidth() int { return 1 }

// attentionScores4 is an unreachable guard in scalar builds.
// attentionBatchWidth prevents the shared dispatcher from selecting this
// SIMD-only operation.
func attentionScores4(q, keys, scores []float32, d, stride, head, count int) {
	panic("four-query SIMD attention unavailable")
}

// attentionValues4 is an unreachable guard in scalar builds; scalar attention
// accumulates one query at a time.
func attentionValues4(out, values, scores []float32, d, stride, head, count int) {
	panic("four-query SIMD attention unavailable")
}
