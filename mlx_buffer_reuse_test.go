//go:build mlx && darwin && arm64 && cgo

package tinyoai

import (
	"os"
	"reflect"
	"testing"
	"time"

	mx "github.com/wbrown/tinyoai/internal/mlx"
)

// This opt-in regression uses actual Clio weights and a caller-provided prompt.
// It exercises Generate, including probability collection and rewind, rather
// than directly reserving buffers or substituting a miniature CPU model.
func TestClioContextBufferReuse(t *testing.T) {
	dir, promptPath := os.Getenv("TINYOAI_MLX_DIR"), os.Getenv("TINYOAI_BUFFER_PROMPT")
	if dir == "" || promptPath == "" {
		t.Skip("set actual Clio model and buffer-regression prompt paths")
	}
	prompt, err := os.ReadFile(promptPath)
	if err != nil {
		t.Fatal(err)
	}
	m, err := LoadMLXNative(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	n := m.native.(*nativeMLX)
	productionWorkspace := n.stableKVWorkspace
	if !productionWorkspace {
		t.Fatal("production loader did not enable stable attention buffers")
	}
	checkCapacity := func(want int) {
		t.Helper()
		var wrong bool
		if err := mx.Run(func() {
			for _, layer := range n.layers {
				wrong = wrong || layer.keys.data.Shape()[2] != want || layer.values.data.Shape()[2] != want
			}
		}); err != nil {
			t.Fatal(err)
		}
		if wrong || n.reserveTokens != want {
			t.Fatalf("Generate did not reserve selected context %d (reservation %d)", want, n.reserveTokens)
		}
	}
	var baseline []logprobRow
	var text string
	opts := GenerateOptions{ContextLength: 8192, MaxTokens: 1, Temperature: 0, Seed: 42, KVBits: 8, Logprobs: 12, PromptLogprobs: true}
	for trial, stable := range []bool{false, productionWorkspace} {
		if err := mx.Run(func() { n.ctx.Synchronize(); n.reset(); mx.ClearCache() }); err != nil {
			t.Fatal(err)
		}
		m.tokens, m.probRows, m.probCount = nil, nil, 0
		// The production Generate entry point must supply the reservation.
		n.reserveTokens, n.stableKVWorkspace = 0, stable
		start := time.Now()
		cold, err := m.Generate(string(prompt), opts)
		if err != nil {
			t.Fatal(err)
		}
		elapsed := time.Since(start)
		checkCapacity(opts.ContextLength)
		if trial == 0 {
			baseline, text = append([]logprobRow(nil), m.probRows...), cold.Text
		} else if cold.Text != text || !reflect.DeepEqual(baseline, m.probRows) {
			t.Fatal("stable workspace changed output or saved prompt probabilities")
		}
		var active, idle uint64
		if err := mx.Run(func() { active, _ = mx.Memory(); idle = mx.CacheMemory() }); err != nil {
			t.Fatal(err)
		}
		t.Logf("stable=%v prompt=%d cold=%s live=%d idle=%d", stable, cold.PromptTokens, elapsed, active, idle)
		warm, err := m.Generate(string(prompt), opts)
		if err != nil {
			t.Fatal(err)
		}
		if warm.CachedPromptTokens != cold.PromptTokens-2 || warm.Text != cold.Text {
			t.Fatal("rewind failed prefix reuse or changed output")
		}
	}
	// Selecting a smaller context invalidates the larger cache. Regrowth must
	// neither expose spare positions nor reuse the abandoned generated suffix.
	short := "The lamp was still burning when Alice returned. She set her bag beside the door and listened."
	opts.ContextLength = 1024
	small, err := m.Generate(short, opts)
	if err != nil {
		t.Fatal(err)
	}
	checkCapacity(opts.ContextLength)
	if small.CachedPromptTokens != 0 {
		t.Fatal("shrinking context retained invalid prefix")
	}
	replayed, err := m.Generate(short, opts)
	if err != nil {
		t.Fatal(err)
	}
	if replayed.CachedPromptTokens != small.PromptTokens-2 || replayed.Text != small.Text {
		t.Fatal("short-context rewind changed output or failed prefix reuse")
	}
	opts.ContextLength = 8192
	restored, err := m.Generate(string(prompt), opts)
	if err != nil {
		t.Fatal(err)
	}
	checkCapacity(opts.ContextLength)
	if restored.Text != text || !reflect.DeepEqual(baseline, m.probRows) {
		t.Fatal("context shrink/regrowth changed output or saved prompt probabilities")
	}
}
