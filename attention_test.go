package tinyoai

import (
	"fmt"
	"math"
	"testing"
)

// batchAttentionFixture adds deterministic query rows and independent score
// scratch to an attention cache fixture.
func batchAttentionFixture(capacity, n, kvHeads int) (*StableLM, *stableState, *stableBatch) {
	m, s := attentionFixture(capacity, kvHeads)
	d := m.config.HiddenSize
	b := &stableBatch{q: make([]float32, n*d), attOut: make([]float32, n*d), att: make([]float32, n*len(s.att))}
	for i := range b.q {
		b.q[i] = float32(math.Sin(float64(i) * .31))
	}
	return m, s, b
}

// TestBatchedAttention compares tiled and one-token attention at ordinary and
// final context positions, poisoning unused KV to detect causal-boundary and
// stride errors.
func TestBatchedAttention(t *testing.T) {
	for _, tc := range []struct{ pos, n, kvHeads int }{{0, 4, 22}, {7, 7, 22}, {8187, 5, 2}, {8188, 4, 22}} {
		t.Run(fmt.Sprint(tc), func(t *testing.T) {
			m, s, b := batchAttentionFixture(8192, tc.n, tc.kvHeads)
			d, capacity := m.config.HiddenSize, 8192
			// Positions outside this batch must never be read, including the
			// partially filled final group of query rows.
			for _, cache := range [][]float32{s.keys[0], s.values[0]} {
				for h := 0; h < tc.kvHeads; h++ {
					for j := (h*capacity + tc.pos + tc.n) * 128; j < (h+1)*capacity*128; j++ {
						cache[j] = float32(math.NaN())
					}
				}
			}
			want := make([]float32, len(b.attOut))
			for j := 0; j < tc.n; j++ {
				s.q = b.q[j*d : (j+1)*d]
				s.attOut = want[j*d : (j+1)*d]
				m.attend(0, tc.pos+j, s)
			}
			m.attendBatch(0, tc.pos, tc.n, s, b)
			for i, v := range want {
				if delta := math.Abs(float64(v - b.attOut[i])); math.IsNaN(delta) || delta > 2e-6 {
					t.Fatalf("element %d: got %g, want %g", i, b.attOut[i], v)
				}
			}
		})
	}
}

// BenchmarkPrefillAttention compares tiled attention with independent
// query-row tasks at short and full context lengths.
func BenchmarkPrefillAttention(b *testing.B) {
	for _, length := range []int{512, 8192} {
		const n = 128
		m, s, batch := batchAttentionFixture(length, n, 22)
		for _, tiled := range []bool{false, true} {
			b.Run(fmt.Sprintf("positions%d/tiled%t", length, tiled), func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if tiled {
						m.attendBatch(0, length-n, n, s, batch)
					} else {
						d := m.config.HiddenSize
						parallelRows(n*m.config.NumAttentionHeads, n*d*length, func(begin, end int) {
							view := *s
							for task := begin; task < end; task++ {
								h, j := task/n, task%n
								view.q, view.attOut = batch.q[j*d:(j+1)*d], batch.attOut[j*d:(j+1)*d]
								view.att = batch.att[j*len(s.att) : (j+1)*len(s.att)]
								m.attendHeads(0, length-n+j, &view, h, h+1)
							}
						})
					}
				}
			})
		}
	}
}
