package tinyoai

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"testing"
)

// TestNumericalBackend records the selected scalar or SIMD implementation in
// the test log.
func TestNumericalBackend(t *testing.T) { t.Log(numericalBackend()) }

// attentionFixture builds deterministic queries and head-major KV for full or
// grouped-query attention without loading weights.
func attentionFixture(positions, kvHeads int) (*StableLM, *stableState) {
	const heads, head = 22, 128
	m := &StableLM{config: StableLMConfig{HiddenSize: heads * head, NumAttentionHeads: heads, NumKeyValueHeads: kvHeads}}
	s := &stableState{
		q: make([]float32, heads*head), attOut: make([]float32, heads*head), att: make([]float32, positions*heads),
		keys: [][]float32{make([]float32, positions*kvHeads*head)}, values: [][]float32{make([]float32, positions*kvHeads*head)},
	}
	rng := rand.New(rand.NewSource(831))
	for _, a := range [][]float32{s.q, s.keys[0], s.values[0]} {
		for i := range a {
			a[i] = 2*rng.Float32() - 1
		}
	}
	return m, s
}

// TestAttentionHeadParallelism requires bitwise-identical attention when the
// worker count changes, including grouped KV at the last context position.
func TestAttentionHeadParallelism(t *testing.T) {
	previous := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(previous)
	// Full MHA and grouped KV heads, including the final context position.
	for _, c := range []struct{ positions, kvHeads int }{{513, 22}, {8192, 2}} {
		m, s := attentionFixture(c.positions, c.kvHeads)
		runtime.GOMAXPROCS(1)
		m.attend(0, c.positions-1, s)
		want := append([]float32(nil), s.attOut...)
		runtime.GOMAXPROCS(16)
		m.attend(0, c.positions-1, s)
		for i, v := range want {
			if math.Float32bits(s.attOut[i]) != math.Float32bits(v) {
				t.Fatalf("%d positions/%d KV heads, element %d: parallel %g, serial %g", c.positions, c.kvHeads, i, s.attOut[i], v)
			}
		}
	}
}

// BenchmarkAttentionLayer measures full-context attention with one and eight
// Go execution slots.
func BenchmarkAttentionLayer(b *testing.B) {
	m, s := attentionFixture(8192, 22)
	for _, workers := range []int{1, 8} {
		b.Run(fmt.Sprintf("workers-%d", workers), func(b *testing.B) {
			previous := runtime.GOMAXPROCS(workers)
			defer runtime.GOMAXPROCS(previous)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				m.attend(0, 8191, s)
			}
		})
	}
}

var dotBenchmarkResult float32

// BenchmarkAttentionDot measures the 128-feature query/key inner product
// without fixture allocation in the timed loop.
func BenchmarkAttentionDot(b *testing.B) {
	q, k := make([]float32, 128), make([]float32, 128)
	for i := range q {
		q[i], k[i] = float32(i%17-8)/9, float32(i%23-11)/12
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dotBenchmarkResult = attentionDot(q, k)
	}
}

// BenchmarkAttentionAdd measures the weighted value accumulation for a
// 128-feature head.
func BenchmarkAttentionAdd(b *testing.B) {
	out, values := make([]float32, 128), make([]float32, 128)
	for i := range values {
		values[i] = float32(i%17-8) / 9
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		attentionAdd(out, values, 0.000001)
	}
}

// TestAttentionKernels compares dot products and weighted sums with float64
// arithmetic across vector widths and scalar tails.
func TestAttentionKernels(t *testing.T) {
	rng := rand.New(rand.NewSource(731))
	for _, n := range []int{1, 3, 4, 7, 15, 16, 17, 127, 128, 129} {
		q, k, out := make([]float32, n), make([]float32, n), make([]float32, n)
		var want float64
		for i := range q {
			q[i], k[i] = 2*rng.Float32()-1, 2*rng.Float32()-1
			want += float64(q[i]) * float64(k[i])
		}
		if got := attentionDot(q, k); math.Abs(float64(got)-want) > 2e-6*math.Max(1, math.Abs(want)) {
			t.Fatalf("dot length %d: %g, want %g", n, got, want)
		}
		for step := 0; step < 13; step++ {
			attentionAdd(out, k, 0.03125)
		}
		for i, got := range out {
			want := float64(k[i]) * 0.03125 * 13
			if math.Abs(float64(got)-want) > 1e-6 {
				t.Fatalf("add length %d element %d: %g, want %g", n, i, got, want)
			}
		}
	}
}

// TestMatvecLengths checks packed BF16 and F32 projections against scalar
// results across alignment boundaries, vector tails, and real model widths.
// Poisoned reusable scratch must give the same result as fresh scratch.
func TestMatvecLengths(t *testing.T) {
	// Odd row widths change the packed BF16 alignment of every other row.
	// Cover vector tails, block boundaries, and both real Clio inner sizes.
	lengths := []int{1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 255, 256, 257, 2815, 2816, 7552}
	rng := rand.New(rand.NewSource(42))
	scratch := make([]float32, 7552)
	for _, dtype := range []string{"BF16", "F32"} {
		for _, cols := range lengths {
			rows := 9
			if cols == 2816 {
				rows = 400 // Also exercise shared scratch with parallel row workers.
			}
			w := weightTensor{dtype: dtype, shape: []int{rows, cols}, data: make([]byte, rows*cols*dtypeWidth(dtype))}
			for i := 0; i < rows*cols; i++ {
				bits := math.Float32bits(2*rng.Float32() - 1)
				if dtype == "BF16" {
					binary.LittleEndian.PutUint16(w.data[i*2:], uint16(bits>>16))
				} else {
					binary.LittleEndian.PutUint32(w.data[i*4:], bits)
				}
			}
			x := make([]float32, cols)
			for i := range x {
				x[i] = 2*rng.Float32() - 1
			}
			got, want := make([]float32, rows), make([]float32, rows)
			w.mul(got, x)
			for i := range scratch {
				scratch[i] = float32(math.NaN())
			}
			reused := make([]float32, rows)
			w.mulScratch(reused, x, scratch)
			for i := range got {
				if math.Float32bits(reused[i]) != math.Float32bits(got[i]) {
					t.Fatalf("%s %dx%d row %d: reused scratch %g, fresh %g", dtype, rows, cols, i, reused[i], got[i])
				}
			}
			w.mulRowsScalar(want, x, 0, rows)
			for i := range got {
				err := math.Abs(float64(got[i]) - float64(want[i]))
				if math.IsNaN(err) || err > 2e-5*math.Max(1, math.Abs(float64(want[i]))) {
					t.Fatalf("%s %dx%d row %d: got %g, scalar %g", dtype, rows, cols, i, got[i], want[i])
				}
			}
		}
	}
}
