//go:build mlx && darwin && arm64 && cgo

package tinyoai

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"github.com/wbrown/tinyoai/tokenizerinfo"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// tinyStableMLXFixture converts the checked-in FP32 fixture to FP16 without
// downloading weights or introducing Python into native integration tests.
func tinyStableMLXFixture(t *testing.T) string {
	t.Helper()
	store, err := openTensorStore("testdata/stablelm")
	if err != nil {
		t.Fatal(err)
	}
	defer store.close()
	names := make([]string, 0, len(store.byName))
	for name := range store.byName {
		names = append(names, name)
	}
	sort.Strings(names)
	header := make(map[string]tensorInfo)
	var payload []byte
	for _, name := range names {
		info := store.byName[name].tensors[name]
		w, err := store.load(name, info.Shape...)
		if err != nil {
			t.Fatal(err)
		}
		count := 1
		for _, d := range info.Shape {
			count *= d
		}
		start := len(payload)
		for i := 0; i < count; i++ {
			payload = binary.LittleEndian.AppendUint16(payload, floatHalf(w.at(i)))
		}
		header[name] = tensorInfo{Dtype: "F16", Shape: info.Shape, Offsets: []int64{int64(start), int64(len(payload))}}
	}
	data, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	// Safetensors headers are padded to eight bytes for native loader alignment.
	for len(data)%8 != 0 {
		data = append(data, ' ')
	}
	path := writeSafeTest(t, string(data), payload)
	dir := filepath.Dir(path)
	for _, name := range []string{"config.json", "tokenizer.json"} {
		data, err := os.ReadFile(filepath.Join("testdata/stablelm", name))
		if err != nil {
			t.Fatal(err)
		}
		if err = os.WriteFile(filepath.Join(dir, name), data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

// TestMLXBenchmarkResume exercises the public adapter, actual Metal prefill,
// report persistence, exact resumption, and rejection of changed prompt policy.
func TestMLXBenchmarkResume(t *testing.T) {
	if os.Getenv("TINYOAI_TEST_METAL") != "1" {
		t.Skip("set TINYOAI_TEST_METAL=1")
	}
	dir := tinyStableMLXFixture(t)
	out := t.TempDir()
	meta, err := newMLX(dir)
	if err != nil {
		t.Fatal(err)
	}
	prompt := "Once upon a time"
	prepared, skips := 0, 0
	config := MLXPrefillExperiments{Cases: []MLXPrefillCase{{Name: "baseline", Context: 64, Chunk: 512, HeadBatch: 32, KVBits: 16}}, DecodeTokens: 2,
		PreparePrompt: func(_ tokenizerinfo.Spec, _ MLXPrefillCase, _ int) (MLXPrefillPrompt, error) {
			prepared++
			return MLXPrefillPrompt{Prompt: prompt, PromptTokens: len(meta.tokenizer.encode(prompt))}, nil
		},
		Progress: func(s string) {
			if strings.HasPrefix(s, "skip completed") {
				skips++
			}
		},
	}
	run := func() error { return RunMLXPrefillExperiments(context.Background(), dir, out, config) }
	if err := run(); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(out, "baseline.json")
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := run(); err != nil {
		t.Fatal(err)
	}
	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if prepared != 2 || skips != 1 || string(before) != string(after) {
		t.Fatal("identical resume did not validate inputs and preserve report")
	}
	prompt = "A different story"
	if err := run(); err == nil || !strings.Contains(err.Error(), "incompatible existing result") {
		t.Fatalf("changed prompt reused old data: %v", err)
	}
}
