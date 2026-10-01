package tinyoai

import (
	"bytes"
	"context"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
)

// ggufFixture independently writes the tiny checkpoint in GGUF's
// reversed-dimension layout with 32-byte alignment. The optional edit callback
// constructs malformed metadata or tensors for rejection tests.
func ggufFixture(t *testing.T, edit func(map[string]any, map[string]weightTensor)) string {
	t.Helper()
	m, err := LoadStableLM("testdata/stablelm")
	if err != nil {
		t.Fatal(err)
	}
	c := m.config
	meta := map[string]any{
		"general.architecture": "stablelm", "stablelm.embedding_length": uint32(c.HiddenSize),
		"stablelm.feed_forward_length": uint32(c.IntermediateSize), "stablelm.block_count": uint32(c.NumHiddenLayers),
		"stablelm.attention.head_count": uint32(c.NumAttentionHeads), "stablelm.attention.head_count_kv": uint32(c.NumKeyValueHeads),
		"stablelm.context_length": uint32(c.MaxPositionEmbeddings), "stablelm.rope.dimension_count": uint32(4),
		"stablelm.use_parallel_residual": true, "stablelm.attention.layer_norm_epsilon": c.LayerNormEps,
		"tokenizer.ggml.bos_token_id": uint32(c.BOSTokenID), "tokenizer.ggml.eos_token_id": uint32(c.EOSTokenID),
		"tokenizer.ggml.tokens": append([]string(nil), m.tokenizer.pieces...),
	}
	store, err := openTensorStore("testdata/stablelm")
	if err != nil {
		t.Fatal(err)
	}
	defer store.close()
	tensors := map[string]weightTensor{}
	for name, f := range store.byName {
		w, err := store.load(name, f.tensors[name].Shape...)
		if err != nil {
			t.Fatal(err)
		}
		tensors[ggufNames.Replace(name)] = w
	}
	if edit != nil {
		edit(meta, tensors)
	}
	var header, body bytes.Buffer
	number := func(v any) {
		if err := binary.Write(&header, binary.LittleEndian, v); err != nil {
			t.Fatal(err)
		}
	}
	str := func(s string) { number(uint64(len(s))); header.WriteString(s) }
	number(uint32(0x46554747))
	number(uint32(3))
	number(uint64(len(tensors)))
	number(uint64(len(meta)))
	keys := make([]string, 0, len(meta))
	for key := range meta {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		str(key)
		switch v := meta[key].(type) {
		case string:
			number(uint32(8))
			str(v)
		case uint32:
			number(uint32(4))
			number(v)
		case float64:
			number(uint32(12))
			number(v)
		case bool:
			number(uint32(7))
			number(v)
		case []string:
			number(uint32(9))
			number(uint32(8))
			number(uint64(len(v)))
			for _, s := range v {
				str(s)
			}
		case []int32:
			number(uint32(9))
			number(uint32(5))
			number(uint64(len(v)))
			number(v)
		default:
			t.Fatalf("unsupported fixture metadata %T", v)
		}
	}
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		w := tensors[name]
		str(name)
		number(uint32(len(w.shape)))
		for i := len(w.shape) - 1; i >= 0; i-- {
			number(uint64(w.shape[i]))
		}
		kind, ok := map[string]uint32{"F32": 0, "F16": 1, "BF16": 30, "Q8_0": 8, "Q5_K": 13, "Q6_K": 14, "Q5_1": 7}[w.dtype]
		if !ok {
			kind = 99
		}
		number(kind)
		number(uint64(body.Len()))
		body.Write(w.data)
		for body.Len()%32 != 0 {
			body.WriteByte(0)
		}
	}
	for header.Len()%32 != 0 {
		header.WriteByte(0)
	}
	header.Write(body.Bytes())
	dir := t.TempDir()
	tok, err := os.ReadFile("testdata/stablelm/tokenizer.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), tok, 0600); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "model.gguf")
	if err := os.WriteFile(path, header.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	return path
}

// TestGGUFRoundTrip requires identical configuration and logits after
// converting the tiny safetensors fixture to GGUF, allowing padded unused
// vocabulary entries.
func TestGGUFRoundTrip(t *testing.T) {
	want, err := LoadStableLM("testdata/stablelm")
	if err != nil {
		t.Fatal(err)
	}
	got, err := LoadStableLM(ggufFixture(t, func(meta map[string]any, _ map[string]weightTensor) {
		meta["tokenizer.ggml.tokens"].([]string)[341] = "[PAD341]"
		types := make([]int32, 342)
		types[341] = 5
		meta["tokenizer.ggml.token_type"] = types
	}))
	if err != nil {
		t.Fatal(err)
	}
	want.config.RopeScaling = nil // JSON null and absent metadata both mean no scaling.
	if !reflect.DeepEqual(got.config, want.config) {
		t.Fatalf("config: %+v vs %+v", got.config, want.config)
	}
	a, b := want.newState(64), got.newState(64)
	ids := want.Encode("Hello world")
	if err := want.prefill(context.Background(), ids, 0, a); err != nil {
		t.Fatal(err)
	}
	if err := got.prefill(context.Background(), ids, 0, b); err != nil {
		t.Fatal(err)
	}
	for i, v := range a.logits {
		if v != b.logits[i] {
			t.Fatalf("logit %d: %g vs %g", i, v, b.logits[i])
		}
	}
}

// TestGGUFRejects exercises unsupported architecture and quantization,
// malformed metadata, incompatible shapes, truncated payloads, and overlapping
// tensors.
func TestGGUFRejects(t *testing.T) {
	cases := []struct {
		name, message string
		edit          func(map[string]any, map[string]weightTensor)
	}{
		{"architecture", "StableLM", func(m map[string]any, _ map[string]weightTensor) { m["general.architecture"] = "llama" }},
		{"nan", "LayerNorm", func(m map[string]any, _ map[string]weightTensor) {
			m["stablelm.attention.layer_norm_epsilon"] = math.NaN()
		}},
		{"infinite", "RoPE", func(m map[string]any, _ map[string]weightTensor) { m["stablelm.rope.freq_base"] = math.Inf(1) }},
		{"alignment", "alignment", func(m map[string]any, _ map[string]weightTensor) { m["general.alignment"] = uint32(3) }},
		{"vocabulary", "vocabulary mismatch", func(m map[string]any, _ map[string]weightTensor) { m["tokenizer.ggml.tokens"].([]string)[30] = "wrong" }},
		{"missing", "tensor count", func(_ map[string]any, w map[string]weightTensor) { delete(w, "output.weight") }},
		{"shape", "shape", func(_ map[string]any, w map[string]weightTensor) {
			v := w["output.weight"]
			v.shape = []int{16, 342}
			w["output.weight"] = v
		}},
		{"unsupported", "unsupported type", func(_ map[string]any, w map[string]weightTensor) {
			v := w["output.weight"]
			v.dtype = "Q4_K"
			w["output.weight"] = v
		}},
		{"Q8 alignment", "divisible by 32", func(_ map[string]any, w map[string]weightTensor) {
			v := w["output.weight"]
			v.dtype = "Q8_0"
			w["output.weight"] = v
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := LoadStableLM(ggufFixture(t, tc.edit))
			if err == nil || !strings.Contains(err.Error(), tc.message) {
				t.Fatalf("got %v, want %s", err, tc.message)
			}
		})
	}
	t.Run("truncated", func(t *testing.T) {
		path := ggufFixture(t, nil)
		info, err := os.Stat(path)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.Truncate(path, info.Size()-100); err != nil {
			t.Fatal(err)
		}
		if _, err := LoadStableLM(path); err == nil {
			t.Fatal("accepted truncated data")
		}
	})
	t.Run("overlap", func(t *testing.T) {
		path := ggufFixture(t, nil)
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		name := "blk.0.attn_norm.bias"
		at := bytes.Index(data, []byte(name)) + len(name)
		rank := int(binary.LittleEndian.Uint32(data[at:]))
		at += 4 + 8*rank + 4
		binary.LittleEndian.PutUint64(data[at:], 0)
		if err := os.WriteFile(path, data, 0600); err != nil {
			t.Fatal(err)
		}
		if _, err := LoadStableLM(path); err == nil || !strings.Contains(err.Error(), "overlapping") {
			t.Fatalf("got %v", err)
		}
	})
}
