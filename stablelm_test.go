package tinyoai

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"
	"unicode/utf8"
)

type referenceData struct {
	ExecutionMode string      `json:"execution_mode"`
	InputIDs      []int       `json:"input_ids"`
	Logits        [][]float32 `json:"logits"`
	Generation    struct {
		Prompt string `json:"prompt"`
		IDs    []int  `json:"ids"`
		Text   string `json:"text"`
	} `json:"generation"`
	Tokenization []struct {
		Text    string `json:"text"`
		IDs     []int  `json:"ids"`
		Decoded string `json:"decoded"`
	} `json:"tokenization"`
}

// readReference decodes a saved tokenizer, logit, and generation fixture or
// fails the calling test.
func readReference(t *testing.T, path string) referenceData {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var r referenceData
	if err := json.Unmarshal(data, &r); err != nil {
		t.Fatal(err)
	}
	return r
}

// TestStableLMReference compares sequential float32 logits with Transformers
// and checks greedy text, streamed text, and token usage against the tiny
// fixture.
func TestStableLMReference(t *testing.T) {
	m, err := LoadStableLM("testdata/stablelm")
	if err != nil {
		t.Fatal(err)
	}
	r := readReference(t, "testdata/stablelm/expected.json")
	s := m.newState(len(r.InputIDs))
	var maxError float64
	for pos, id := range r.InputIDs {
		if err := m.forward(context.Background(), id, pos, s, true); err != nil {
			t.Fatal(err)
		}
		for i, want := range r.Logits[pos] {
			delta := math.Abs(float64(s.logits[i] - want))
			maxError = math.Max(maxError, delta)
			if delta > 3e-5 {
				t.Fatalf("position %d logit %d = %.8g, want %.8g (delta %g)", pos, i, s.logits[i], want, delta)
			}
		}
	}
	t.Logf("maximum error vs Transformers float32: %.3g", maxError)
	var streamed strings.Builder
	res, err := m.Generate(r.Generation.Prompt, GenerateOptions{MaxTokens: len(r.Generation.IDs), OnToken: func(p string) { streamed.WriteString(p) }})
	if err != nil {
		t.Fatal(err)
	}
	if res.Text != r.Generation.Text || streamed.String() != res.Text {
		t.Fatalf("generation = %q / stream %q, want %q", res.Text, streamed.String(), r.Generation.Text)
	}
	if res.PromptTokens != len(m.Encode(r.Generation.Prompt)) || res.CompletionTokens != len(r.Generation.IDs) {
		t.Fatalf("incorrect usage: %+v", res)
	}
}

// TestNerdstashReference compares encoding and incremental decoding with the
// saved tokenizer reference cases.
func TestNerdstashReference(t *testing.T) {
	m, err := LoadStableLM("testdata/stablelm")
	if err != nil {
		t.Fatal(err)
	}
	r := readReference(t, "testdata/stablelm/expected.json")
	for _, c := range r.Tokenization {
		t.Run(c.Text, func(t *testing.T) {
			ids := m.Encode(c.Text)
			if !reflect.DeepEqual(ids, c.IDs) {
				t.Fatalf("encode %q = %v, want %v", c.Text, ids, c.IDs)
			}
			d := nerdstashDecoder{tokenizer: m.tokenizer}
			var decoded strings.Builder
			for _, id := range ids {
				decoded.WriteString(d.add(id))
			}
			decoded.WriteString(d.flush())
			if decoded.String() != c.Decoded {
				t.Fatalf("decode = %q, want %q", decoded.String(), c.Decoded)
			}
		})
	}
}

// TestStableLMGenerationLimits checks cancellation, oversized prompts, invalid
// options, and EOS accounting independently of the toy model's language
// distribution.
func TestStableLMGenerationLimits(t *testing.T) {
	m, err := LoadStableLM("testdata/stablelm")
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := m.Generate("", GenerateOptions{Context: ctx}); err != context.Canceled {
		t.Fatalf("canceled generate: %v", err)
	}
	if _, err := m.Generate(strings.Repeat("1", 64), GenerateOptions{MaxTokens: 1}); err == nil {
		t.Fatal("accepted oversized prompt")
	}
	for _, opts := range []GenerateOptions{{MaxTokens: -1}, {Temperature: -1}, {Temperature: math.NaN()}, {Temperature: math.Inf(1)}} {
		if _, err := m.Generate("", opts); err == nil {
			t.Fatal("accepted invalid options")
		}
	}
	// Force an EOS after a single sampled token without coupling the assertion
	// to the toy model's random language distribution.
	clear(m.head.data)
	m.config.EOSTokenID = 0
	r, err := m.Generate("", GenerateOptions{MaxTokens: 8})
	if err != nil || r.FinishReason != "stop" || r.Text != "" || r.CompletionTokens != 1 || r.PromptTokens != 1 {
		t.Fatalf("EOS handling: %+v, %v", r, err)
	}
}

// TestStableLMConcurrentGeneration verifies that concurrent seeded requests
// sharing a model produce the same completion while serializing access to its
// cache.
func TestStableLMConcurrentGeneration(t *testing.T) {
	m, err := LoadStableLM("testdata/stablelm")
	if err != nil {
		t.Fatal(err)
	}
	want, err := m.Generate("Hello", GenerateOptions{MaxTokens: 4, Seed: 7, Temperature: 0.8})
	if err != nil {
		t.Fatal(err)
	}
	want.CachedPromptTokens = want.PromptTokens - 1
	for i := 0; i < 4; i++ {
		t.Run("request", func(t *testing.T) {
			t.Parallel()
			got, err := m.Generate("Hello", GenerateOptions{MaxTokens: 4, Seed: 7, Temperature: 0.8})
			if err != nil || !reflect.DeepEqual(got, want) {
				t.Errorf("%+v, %v; want %+v", got, err, want)
			}
		})
	}
}

// TestStableRotatePartialSplitHalf checks the split-half rotary equation and
// verifies that features outside the rotary subspace are unchanged.
func TestStableRotatePartialSplitHalf(t *testing.T) {
	x := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	stableRotate(x, 8, 4, 1, 10000)
	want := []float64{math.Cos(1) - 3*math.Sin(1), 2*math.Cos(.01) - 4*math.Sin(.01), 3*math.Cos(1) + math.Sin(1), 4*math.Cos(.01) + 2*math.Sin(.01), 5, 6, 7, 8}
	for i := range x {
		if math.Abs(float64(x[i])-want[i]) > 1e-6 {
			t.Fatalf("rotation[%d] = %g, want %g", i, x[i], want[i])
		}
	}
}

// TestStableRotateContextBoundary compares rotary output with saved reference
// values at high absolute positions.
func TestStableRotateContextBoundary(t *testing.T) {
	data, err := os.ReadFile("testdata/stablelm/rope.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		Input []float32 `json:"input"`
		Cases []struct {
			Position int       `json:"position"`
			Output   []float32 `json:"output"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(data, &ref); err != nil {
		t.Fatal(err)
	}
	for _, c := range ref.Cases {
		x := append([]float32(nil), ref.Input...)
		stableRotate(x, 128, 32, c.Position, 10000)
		for i, v := range x {
			if d := math.Abs(float64(v - c.Output[i])); d > 5e-7 {
				t.Fatalf("position %d component %d: delta %g", c.Position, i, d)
			}
		}
	}
}

// TestCompletionOutput checks withholding of partial stop markers,
// earliest-match selection, and identical streamed and collected text.
func TestCompletionOutput(t *testing.T) {
	for _, c := range []struct {
		pieces, stops []string
		want          string
		stopped       bool
	}{
		{[]string{"hello E", "N", "D ignored"}, []string{"END"}, "hello ", true},
		{[]string{"hello E", "N", "ough"}, []string{"END"}, "hello ENough", false},
		{[]string{"hello EN"}, []string{"END"}, "hello EN", false},
		{[]string{"abcdef"}, []string{"de", "bc"}, "a", true},
	} {
		var stream strings.Builder
		o := completionOutput{stops: c.stops, onToken: func(p string) { stream.WriteString(p) }}
		for _, p := range c.pieces {
			o.add(p, false)
		}
		o.add("", true)
		if o.text.String() != c.want || stream.String() != c.want || o.stopped != c.stopped {
			t.Fatalf("output %q / %q / %v, want %+v", o.text.String(), stream.String(), o.stopped, c)
		}
	}
}

// TestNerdstashByteFallback checks complete and malformed byte runs, including
// special tokens between fragments, and requires every emitted string to be
// valid UTF-8.
func TestNerdstashByteFallback(t *testing.T) {
	m, err := LoadStableLM("testdata/stablelm")
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		bytes []byte
		want  string
	}{
		{[]byte("🦊"), "🦊"}, {[]byte{0xf0, 0x9f}, "��"},
		{[]byte{0x61, 0xff}, "��"}, {append([]byte("🦊"), 0xff), "�����"},
	} {
		d := nerdstashDecoder{tokenizer: m.tokenizer}
		var got string
		for _, b := range c.bytes {
			got += d.add(m.tokenizer.vocab[fmt.Sprintf("<0x%02X>", b)])
			got += d.add(0) // skipped special tokens do not break byte runs
			if !utf8.ValidString(got) {
				t.Fatal("invalid UTF-8 streamed")
			}
		}
		got += d.flush()
		if got != c.want {
			t.Fatalf("decode %x = %q, want %q", c.bytes, got, c.want)
		}
	}
}

// TestStableLMRejectsUnsupportedConfig rejects architecture variants whose
// residual, attention, embedding, or rotary semantics are not implemented.
func TestStableLMRejectsUnsupportedConfig(t *testing.T) {
	m, err := LoadStableLM("testdata/stablelm")
	if err != nil {
		t.Fatal(err)
	}
	for name, mutate := range map[string]func(*StableLMConfig){
		"sequential": func(c *StableLMConfig) { c.UseParallelResidual = false },
		"bias":       func(c *StableLMConfig) { c.UseQKVBias = true },
		"qk_norm":    func(c *StableLMConfig) { c.QKLayerNorm = true },
		"tied":       func(c *StableLMConfig) { c.TieWordEmbeddings = true },
		"heads":      func(c *StableLMConfig) { c.NumAttentionHeads = 0 },
		"kv_heads":   func(c *StableLMConfig) { c.NumKeyValueHeads = 3 },
		"rope":       func(c *StableLMConfig) { c.PartialRotaryFactor = .125 },
		"scaling":    func(c *StableLMConfig) { c.RopeScaling = json.RawMessage(`{"type":"linear","factor":2}`) },
	} {
		t.Run(name, func(t *testing.T) {
			c := m.Config()
			mutate(&c)
			if c.validate() == nil {
				t.Fatal("accepted unsupported config")
			}
		})
	}
}

// TestClioCheckpoint is an opt-in full-checkpoint tokenizer and generation
// check. TINYOAI_CLIO_REFERENCE enables a logit comparison against the CPU
// float32 cached-token reference produced by scripts/clio_reference.py.
// Ordinary tests never download weights.
func TestClioCheckpoint(t *testing.T) {
	dir := os.Getenv("TINYOAI_CLIO_DIR")
	if dir == "" {
		t.Skip("set TINYOAI_CLIO_DIR to the downloaded Clio checkpoint")
	}
	m, err := LoadStableLM(dir)
	if err != nil {
		t.Fatal(err)
	}
	if ids := m.Encode("Once upon a time"); !reflect.DeepEqual(ids, []int{2, 8144, 2377, 333, 682}) {
		t.Fatalf("Clio tokenization: %v", ids)
	}
	if path := os.Getenv("TINYOAI_CLIO_REFERENCE"); path != "" {
		r := readReference(t, path)
		if r.ExecutionMode != "cpu-float32-cached-token" {
			t.Fatalf("reference must use CPU float32 cached one-token mode; regenerate with scripts/clio_reference.py")
		}
		for _, c := range r.Tokenization {
			if ids := m.Encode(c.Text); !reflect.DeepEqual(ids, c.IDs) {
				t.Fatalf("Clio encode %q = %v, want %v", c.Text, ids, c.IDs)
			}
			d := nerdstashDecoder{tokenizer: m.tokenizer}
			var decoded strings.Builder
			for _, id := range c.IDs {
				decoded.WriteString(d.add(id))
			}
			decoded.WriteString(d.flush())
			if decoded.String() != c.Decoded {
				t.Fatalf("Clio decode %q = %q, want %q", c.Text, decoded.String(), c.Decoded)
			}
		}
		state := m.newState(len(r.InputIDs))
		var maxError float64
		for pos, id := range r.InputIDs {
			if err := m.forward(context.Background(), id, pos, state, true); err != nil {
				t.Fatal(err)
			}
			for i, want := range r.Logits[pos] {
				delta := math.Abs(float64(state.logits[i] - want))
				maxError = math.Max(maxError, delta)
				if delta > 0.0005 {
					t.Fatalf("Clio position %d logit %d = %g, want %g (delta %g)", pos, i, state.logits[i], want, delta)
				}
			}
		}
		t.Logf("Clio max logit error vs float32 reference: %g", maxError)
		started := time.Now()
		res, err := m.Generate(r.Generation.Prompt, GenerateOptions{MaxTokens: len(r.Generation.IDs)})
		if err != nil {
			t.Fatal(err)
		}
		if res.Text != r.Generation.Text {
			t.Fatalf("Clio completion = %q, want %q", res.Text, r.Generation.Text)
		}
		t.Logf("Clio: %q (%d completion tokens)", res.Text, res.CompletionTokens)
		t.Logf("Clio prompt + generation: %v", time.Since(started))
	} else {
		res, err := m.Generate("Once upon a time", GenerateOptions{MaxTokens: 12})
		if err != nil {
			t.Fatal(err)
		}
		if res.Text == "" {
			t.Fatal("empty Clio completion")
		}
		t.Logf("Clio: %+v", res)
	}
}
