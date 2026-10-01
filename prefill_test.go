package tinyoai

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

// matrixFixture creates deterministic row-major F32, BF16, or Q8_0 projection
// weights for kernel tests.
func matrixFixture(rows, cols int, dtype string) weightTensor {
	nbytes := rows * cols * 2
	if dtype == "F32" {
		nbytes = rows * cols * 4
	}
	if dtype == "Q8_0" {
		nbytes = rows * cols / 32 * 34
	}
	w := weightTensor{shape: []int{rows, cols}, dtype: dtype, data: make([]byte, nbytes)}
	rng := rand.New(rand.NewSource(621))
	if dtype == "Q8_0" {
		for i := 0; i < nbytes; i += 34 {
			binary.LittleEndian.PutUint16(w.data[i:], 0x2400) // exactly 1/64
			for j := 0; j < 32; j++ {
				w.data[i+2+j] = byte(rng.Intn(256))
			}
		}
	} else {
		for i := 0; i < rows*cols; i++ {
			bits := math.Float32bits(rng.Float32() - .5)
			if dtype == "F32" {
				binary.LittleEndian.PutUint32(w.data[i*4:], bits)
			} else {
				binary.LittleEndian.PutUint16(w.data[i*2:], uint16(bits>>16))
			}
		}
	}
	return w
}

type cancelAfterChecks struct {
	context.Context
	remaining int
}

// Err injects cancellation after a fixed number of context checks, making
// partial-prefill tests deterministic.
func (c *cancelAfterChecks) Err() error {
	c.remaining--
	if c.remaining <= 0 {
		return context.Canceled
	}
	return nil
}

// TestPrefixCacheInterrupted verifies that interrupted prefill and decode can
// resume with the same greedy, seeded, and stop-limited results as cold
// inference.
func TestPrefixCacheInterrupted(t *testing.T) {
	m, err := LoadStableLM("testdata/stablelm")
	if err != nil {
		t.Fatal(err)
	}
	m.config.MaxPositionEmbeddings = 384
	prompt := "Hello" + strings.Repeat("1", prefillBatch+26)
	// Cancel after the first chunk, leaving partially populated layer caches.
	ctx := &cancelAfterChecks{Context: context.Background(), remaining: 6}
	if _, err := m.Generate(prompt, GenerateOptions{MaxTokens: 4, Context: ctx}); err != context.Canceled {
		t.Fatalf("prefill cancellation: %v", err)
	}
	check := func(opts GenerateOptions) {
		t.Helper()
		got, err := m.Generate(prompt, opts)
		if err != nil {
			t.Fatal(err)
		}
		fresh, err := LoadStableLM("testdata/stablelm")
		if err != nil {
			t.Fatal(err)
		}
		fresh.config.MaxPositionEmbeddings = 384
		want, err := fresh.Generate(prompt, opts)
		if err != nil {
			t.Fatal(err)
		}
		want.CachedPromptTokens = got.CachedPromptTokens
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("cached %+v vs cold %+v", got, want)
		}
	}
	check(GenerateOptions{MaxTokens: 8})
	ctx2, cancel := context.WithCancel(context.Background())
	defer cancel()
	if _, err := m.Generate(prompt, GenerateOptions{MaxTokens: 8, Context: ctx2, OnToken: func(string) { cancel() }}); err != context.Canceled {
		t.Fatalf("decode cancellation: %v", err)
	}
	check(GenerateOptions{MaxTokens: 8})
	check(GenerateOptions{MaxTokens: 8, Temperature: .7, Seed: 91})
	check(GenerateOptions{MaxTokens: 8, Stop: []string{"a", "the"}})
}

// TestClioPrefixCache is an opt-in full-model measurement of cold, warm, and
// appended prompts. It loads weights once, checks full-vocabulary KL, and
// optionally saves timings and logits.
func TestClioPrefixCache(t *testing.T) {
	dir, ref := os.Getenv("TINYOAI_CLIO_DIR"), os.Getenv("TINYOAI_CACHE_REFERENCE")
	if dir == "" || ref == "" {
		t.Skip("set TINYOAI_CLIO_DIR and TINYOAI_CACHE_REFERENCE")
	}
	m, err := LoadStableLM(dir)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(ref, "prompt.txt"))
	if err != nil {
		t.Fatal(err)
	}
	prompt := string(data)
	rows := []map[string]any{}
	var logits [][]float32
	trace := newLogitTrace(t, os.Getenv("TINYOAI_CACHE_TRACE"), m.config.VocabSize)
	run := func(name, text string) GenerateResult {
		start := time.Now()
		got, err := m.Generate(text, GenerateOptions{MaxTokens: 1})
		if err != nil {
			t.Fatal(err)
		}
		seconds := time.Since(start).Seconds()
		cached := <-m.cache
		logits = append(logits, append([]float32(nil), cached.state.logits...))
		m.cache <- cached
		trace.append(t, logits[len(logits)-1])
		t.Logf("%s: %.4fs, %d prompt, %d cached", name, seconds, got.PromptTokens, got.CachedPromptTokens)
		rows = append(rows, map[string]any{"case": name, "seconds": seconds, "prompt_tokens": got.PromptTokens, "cached_tokens": got.CachedPromptTokens, "text": got.Text})
		return got
	}
	cold, warm := run("cold", prompt), run("warm", prompt)
	if warm.CachedPromptTokens != cold.PromptTokens-1 {
		t.Fatal("incomplete cache hit")
	}
	cold.CachedPromptTokens = warm.CachedPromptTokens
	if !reflect.DeepEqual(cold, warm) {
		t.Fatalf("cold %+v vs warm %+v", cold, warm)
	}
	appended := run("append", prompt+"\nThe next record began:")
	saved := <-m.cache
	saved.tokens = nil // Force a cold comparison while reusing the allocated buffers.
	m.cache <- saved
	uncached := run("append_cold", prompt+"\nThe next record began:")
	uncached.CachedPromptTokens = appended.CachedPromptTokens
	if !reflect.DeepEqual(uncached, appended) {
		t.Fatalf("append %+v vs cold %+v", appended, uncached)
	}
	kl := map[string]float64{"warm": distributionKL(logits[0], logits[1]), "append": distributionKL(logits[3], logits[2])}
	for name, value := range kl {
		if math.IsNaN(value) || value < 0 || value > 1e-7 {
			t.Fatalf("%s cache KL = %g", name, value)
		}
	}
	t.Logf("full-vocabulary cache KL: %v", kl)
	if file := os.Getenv("TINYOAI_CACHE_REPORT"); file != "" {
		data, err := json.MarshalIndent(map[string]any{"model": dir, "backend": numericalBackend(), "runs": rows, "cache_kl": kl, "logits_file": os.Getenv("TINYOAI_CACHE_TRACE"), "memory": parityMemory()}, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(file, append(data, '\n'), 0600); err != nil {
			t.Fatal(err)
		}
	}
}

// TestBatchedProjection compares batched F32, BF16, and Q8_0 multiplication
// with a float64 oracle across row groups, packed blocks, and tails.
func TestBatchedProjection(t *testing.T) {
	for _, dtype := range []string{"BF16", "F32", "Q8_0"} {
		for _, cols := range []int{32, 255, 256, 257, 2816, 7552} {
			if dtype == "Q8_0" && cols%32 != 0 {
				continue
			}
			w := matrixFixture(7, cols, dtype)
			for _, n := range []int{1, 2, 7, 17, 63, 64, 65, 128} {
				x := make([]float32, n*cols)
				for i := range x {
					x[i] = float32(math.Sin(float64(i) * .13))
				}
				got := make([]float32, n*7)
				packed := make([]float32, cols*((n+batchWidth()-1)/batchWidth()*batchWidth()))
				w.mulBatch(got, x, n, packed)
				for token := 0; token < n; token++ {
					for row := 0; row < 7; row++ {
						var want float64
						for j := 0; j < cols; j++ {
							want += float64(x[token*cols+j]) * float64(w.at(row*cols+j))
						}
						if delta := math.Abs(float64(got[token*7+row]) - want); delta > 2e-4*math.Max(1, math.Abs(want)) {
							t.Fatalf("%s n%d k%d: %g vs %g", dtype, n, cols, got[token*7+row], want)
						}
					}
				}
			}
		}
	}
}

// TestBatchedPrefill checks batched logits and populated KV against sequential
// forwards, then decodes one more token to expose cache-layout errors. Unused
// positions are poisoned with NaNs.
func TestBatchedPrefill(t *testing.T) {
	m, err := LoadStableLM("testdata/stablelm")
	if err != nil {
		t.Fatal(err)
	}
	m.config.MaxPositionEmbeddings = 384
	for _, n := range []int{1, 2, 17, 63, 64, 65, 127, 128, 129, 257} {
		t.Run(fmt.Sprint(n), func(t *testing.T) {
			ids := make([]int, n+7)
			for i := range ids {
				ids[i] = 32 + (i*13)%250
			}
			seq, batch := m.newState(384), m.newState(384)
			// Poison unused positions to catch accidental reads past the causal boundary.
			for _, s := range []*stableState{seq, batch} {
				for l := range s.keys {
					for i := range s.keys[l] {
						s.keys[l][i] = float32(math.NaN())
						s.values[l][i] = float32(math.NaN())
					}
				}
			}
			for i, id := range ids {
				if err := m.forward(context.Background(), id, i, seq, true); err != nil {
					t.Fatal(err)
				}
				if i < 7 {
					if err := m.forward(context.Background(), id, i, batch, true); err != nil {
						t.Fatal(err)
					}
				}
			}
			if err := m.prefill(context.Background(), ids[7:], 7, batch); err != nil {
				t.Fatal(err)
			}
			compare := func(a, b []float32) {
				t.Helper()
				for i, v := range a {
					if d := math.Abs(float64(v - b[i])); math.IsNaN(d) || d > 3e-5 {
						t.Fatalf("element %d error %g", i, d)
					}
				}
			}
			compare(seq.logits, batch.logits)
			for l := range seq.keys {
				for h := 0; h < m.config.NumKeyValueHeads; h++ {
					at := h * 384 * 8
					end := at + len(ids)*8
					compare(seq.keys[l][at:end], batch.keys[l][at:end])
					compare(seq.values[l][at:end], batch.values[l][at:end])
				}
			}
			for _, s := range []*stableState{seq, batch} {
				if err := m.forward(context.Background(), 123, len(ids), s, true); err != nil {
					t.Fatal(err)
				}
			}
			compare(seq.logits, batch.logits)
		})
	}
}

// TestPrefixCache compares repeated, extended, shortened, and replaced prompts
// with cold inference and checks that a cancelled waiter cannot take the
// resident cache.
func TestPrefixCache(t *testing.T) {
	m, err := LoadStableLM("testdata/stablelm")
	if err != nil {
		t.Fatal(err)
	}
	prompts := []string{"Hello", "Hello", "Hello" + strings.Repeat("1", 20), "Hello1112", "Goodbye", "Hello"}
	var previous []int
	for _, prompt := range prompts {
		// No output token is forwarded at max_tokens=1, so cache contains the prompt exactly.
		got, err := m.Generate(prompt, GenerateOptions{MaxTokens: 1})
		if err != nil {
			t.Fatal(err)
		}
		fresh, err := LoadStableLM("testdata/stablelm")
		if err != nil {
			t.Fatal(err)
		}
		want, err := fresh.Generate(prompt, GenerateOptions{MaxTokens: 1})
		if err != nil {
			t.Fatal(err)
		}
		ids := m.Encode(prompt)
		common := 0
		for common < min(len(previous), len(ids)-1) && previous[common] == ids[common] {
			common++
		}
		want.CachedPromptTokens = common
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("%q cached=%+v fresh=%+v", prompt, got, want)
		}
		previous = ids
	}
	// Cancellation while queued must not steal or damage the owned context.
	saved := <-m.cache
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	if _, err := m.Generate("Hello", GenerateOptions{Context: ctx, MaxTokens: 1}); err != context.DeadlineExceeded {
		t.Fatalf("queued cancellation: %v", err)
	}
	m.cache <- saved
	if _, err := m.Generate("Hello", GenerateOptions{MaxTokens: 1}); err != nil {
		t.Fatal(err)
	}
}

// TestPrefixCacheCancellationAfterAcquire verifies that cancellation just
// after acquiring the gate leaves the previous prefix unchanged.
func TestPrefixCacheCancellationAfterAcquire(t *testing.T) {
	m, err := LoadStableLM("testdata/stablelm")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := m.Generate("Hello111111", GenerateOptions{MaxTokens: 1}); err != nil {
		t.Fatal(err)
	}
	saved := <-m.cache
	want := append([]int(nil), saved.tokens...)
	m.cache <- saved
	ctx := &cancelAfterChecks{Context: context.Background(), remaining: 2}
	if _, err := m.Generate("Goodbye", GenerateOptions{MaxTokens: 1, Context: ctx}); err != context.Canceled {
		t.Fatalf("cancellation: %v", err)
	}
	after := <-m.cache
	defer func() { m.cache <- after }()
	if after != saved || len(after.tokens) != len(want) {
		t.Fatal("cancelled waiter replaced or truncated cache")
	}
	for i, v := range want {
		if after.tokens[i] != v {
			t.Fatal("cancelled waiter changed cached tokens")
		}
	}
}

// BenchmarkBatchedProjection measures BF16 and Q8_0 throughput at several
// token batch sizes using reusable packing scratch.
func BenchmarkBatchedProjection(b *testing.B) {
	for _, dtype := range []string{"BF16", "Q8_0"} {
		w := matrixFixture(2816, 2816, dtype)
		for _, n := range []int{1, 16, 64} {
			b.Run(fmt.Sprintf("%s/n%d", dtype, n), func(b *testing.B) {
				x, out := make([]float32, n*2816), make([]float32, n*2816)
				for i := range x {
					x[i] = float32(i%17-8) / 9
				}
				packed := make([]float32, 2816*((n+batchWidth()-1)/batchWidth()*batchWidth()))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					w.mulBatch(out, x, n, packed)
				}
				b.ReportMetric(float64(n)*2816*2816*2*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOP/s")
			})
		}
	}
}
