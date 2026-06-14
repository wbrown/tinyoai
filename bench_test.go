package tinyoai

import (
	"fmt"
	"testing"
)

// Benchmarks for the embedded stories260K engine. Run with:
//
//	go test -bench=. -benchmem -run='^$'
//
// Each reports a "tok/s" custom metric in addition to ns/op. The engine is
// pure-Go scalar code (no SIMD/Metal), so this measures single-core throughput.

// BenchmarkGenerate measures end-to-end generation throughput: prompt encoding,
// the per-token forward pass, and sampling, plus the per-call run-state
// allocation. Greedy decoding (temperature 0) keeps the work deterministic so
// each iteration does identical work.
func BenchmarkGenerate(b *testing.B) {
	m, err := Default()
	if err != nil {
		b.Fatalf("Default: %v", err)
	}
	const maxTokens = 256

	b.ReportAllocs()
	b.ResetTimer()

	produced := 0
	for i := 0; i < b.N; i++ {
		res, err := m.Generate("Once upon a time", GenerateOptions{
			MaxTokens:   maxTokens,
			Temperature: 0,
		})
		if err != nil {
			b.Fatalf("Generate: %v", err)
		}
		produced += res.CompletionTokens
	}
	b.StopTimer()

	b.ReportMetric(float64(produced)/b.Elapsed().Seconds(), "tok/s")
}

// BenchmarkForwardPass measures a single transformer forward pass (one decoded
// token) at several context lengths, isolating the matmul-heavy hot path. The
// KV cache is prefilled to each position first, so the attention loop reads real
// keys/values and the cost reflects that context length. tok/s here is the raw
// decode rate (forward passes per second) at that position.
func BenchmarkForwardPass(b *testing.B) {
	m, err := Default()
	if err != nil {
		b.Fatalf("Default: %v", err)
	}

	for _, pos := range []int32{0, 64, 256, 511} {
		b.Run(fmt.Sprintf("pos=%d", pos), func(b *testing.B) {
			state := m.newRunState()
			// Prefill the cache for positions [0, pos) so the attention loop at
			// pos reads populated keys/values.
			for p := int32(0); p < pos; p++ {
				m.transformer(1, p, state)
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				m.transformer(1, pos, state)
			}
			b.StopTimer()

			b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "tok/s")
		})
	}
}
