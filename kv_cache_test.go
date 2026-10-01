package tinyoai

import (
	"math"
	"reflect"
	"strings"
	"testing"
)

// TestHalfRounding exhaustively checks half-float round trips and midpoint
// ties, including signs, subnormals, NaNs, and the overflow boundary.
func TestHalfRounding(t *testing.T) {
	for bits := 0; bits <= 65535; bits++ {
		h := uint16(bits)
		v := halfFloat(h)
		got := floatHalf(v)
		if math.IsNaN(float64(v)) {
			if got&0x7fff != 0x7e00 {
				t.Fatalf("NaN encoding %x", got)
			}
		} else if got != h {
			t.Fatalf("round trip %x -> %g -> %x", h, v, got)
		}
	}
	// Every midpoint between positive finite half values, and its neighboring
	// float32 values, checks rounding independently of the encoding algorithm.
	for h := uint16(0); h < 0x7bff; h++ {
		mid := (halfFloat(h) + halfFloat(h+1)) / 2
		want := h + h&1
		if floatHalf(mid) != want || floatHalf(-mid) != want|0x8000 {
			t.Fatalf("tie at %x: %g -> %x, want %x", h, mid, floatHalf(mid), want)
		}
		bits := math.Float32bits(mid)
		if floatHalf(math.Float32frombits(bits-1)) != h || floatHalf(math.Float32frombits(bits+1)) != h+1 {
			t.Fatalf("midpoint neighbors at %x", h)
		}
	}
	if floatHalf(65520) != 0x7c00 || floatHalf(-65520) != 0xfc00 || floatHalf(65504) != 0x7bff {
		t.Fatal("half overflow boundary")
	}
}

// TestHalfKVPrefixReuse compares warm, extended, and replaced FP16 caches with
// cold inference and verifies that FP32 layer caches are not retained.
func TestHalfKVPrefixReuse(t *testing.T) {
	load := func() *StableLM {
		t.Helper()
		m, err := LoadStableLMWithOptions("testdata/stablelm", StableLMOptions{KVCache: "f16"})
		if err != nil {
			t.Fatal(err)
		}
		m.config.MaxPositionEmbeddings = 384
		return m
	}
	m := load()
	for _, prompt := range []string{"Hello", "Hello", "Hello" + strings.Repeat("1", 140), "Hello" + strings.Repeat("1", 155), "Goodbye"} {
		got, err := m.Generate(prompt, GenerateOptions{MaxTokens: 4})
		if err != nil {
			t.Fatal(err)
		}
		fresh := load()
		want, err := fresh.Generate(prompt, GenerateOptions{MaxTokens: 4})
		if err != nil {
			t.Fatal(err)
		}
		want.CachedPromptTokens = got.CachedPromptTokens
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("half cache reuse: got %+v, want %+v", got, want)
		}
		saved := <-m.cache
		cold := <-fresh.cache
		if kl := distributionKL(cold.state.logits, saved.state.logits); kl > 1e-8 || math.IsNaN(kl) {
			t.Fatalf("half cache logits KL %g", kl)
		}
		if len(saved.state.keys) != 0 || len(saved.state.keys16) != m.config.NumHiddenLayers {
			t.Fatal("half cache retained float32 layers")
		}
		m.cache <- saved
	}
}
