package tinyoai

import (
	"bytes"
	"encoding/binary"
	"os"
	"strings"
	"testing"
)

// TestLoadInvalidLlamaDimensions rejects invalid hidden sizes, grouped-query
// head counts, rotary widths, and vocabularies before allocating weights.
func TestLoadInvalidLlamaDimensions(t *testing.T) {
	valid := Config{Dim: 64, HiddenDim: 172, NLayers: 5, NHeads: 8, NKvHeads: 4, VocabSize: 512, SeqLen: 512}
	for name, change := range map[string]func(*Config){
		"hidden":          func(c *Config) { c.HiddenDim = -1 },
		"zero KV heads":   func(c *Config) { c.NKvHeads = 0 },
		"uneven GQA":      func(c *Config) { c.NKvHeads = 3 },
		"odd rotary head": func(c *Config) { c.Dim = 56 },
		"no BOS":          func(c *Config) { c.VocabSize = 1 },
	} {
		t.Run(name, func(t *testing.T) {
			c := valid
			change(&c)
			var checkpoint bytes.Buffer
			if err := binary.Write(&checkpoint, binary.LittleEndian, c); err != nil {
				t.Fatal(err)
			}
			_, err := LoadModel(&checkpoint, bytes.NewReader(nil))
			if err == nil || !strings.Contains(err.Error(), "invalid config header") {
				t.Fatalf("must reject invalid dimensions before allocating weights: %v", err)
			}
		})
	}
}

// TestEmbeddedModelLayout pins the embedded stories260K config and verifies the
// on-disk weight count matches the GQA layout exactly (header + every tensor).
// If the asset is swapped or the layout drifts, this fails with the discrepancy.
func TestEmbeddedModelLayout(t *testing.T) {
	f, err := os.Open("assets/stories260K.bin")
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer f.Close()
	var c Config
	if err := binary.Read(f, binary.LittleEndian, &c); err != nil {
		t.Fatalf("read header: %v", err)
	}
	want := Config{Dim: 64, HiddenDim: 172, NLayers: 5, NHeads: 8, NKvHeads: 4, VocabSize: 512, SeqLen: 512}
	if c != want {
		t.Fatalf("config = %+v, want %+v", c, want)
	}

	fi, _ := f.Stat()
	gotFloats := (fi.Size() - 28) / 4 // file minus the 7-int32 header

	dim, hidden, layers := int64(c.Dim), int64(c.HiddenDim), int64(c.NLayers)
	headSize := int64(c.Dim / c.NHeads)
	kvDim := headSize * int64(c.NKvHeads)
	seq := int64(c.SeqLen)
	wantFloats := int64(c.VocabSize)*dim + // token embedding
		2*layers*dim + // rms att + rms ffn
		2*layers*dim*dim + // wq, wo
		2*layers*dim*kvDim + // wk, wv (GQA)
		3*layers*hidden*dim + // w1, w2, w3
		dim + // rms final
		2*seq*headSize/2 // freq_cis real + imag
	if gotFloats != wantFloats {
		t.Errorf("on-disk weight floats = %d, computed layout = %d", gotFloats, wantFloats)
	}
}

// TestLoadDefaultModel verifies the embedded stories260K checkpoint parses with
// the legacy llama2.c layout: the header is sane and every byte is consumed
// (the trailing-byte guard in LoadModel turns a layout mismatch into an error).
func TestLoadDefaultModel(t *testing.T) {
	ckpt, err := os.Open("assets/stories260K.bin")
	if err != nil {
		t.Fatalf("open checkpoint: %v", err)
	}
	defer ckpt.Close()
	tok, err := os.Open("assets/tok512.bin")
	if err != nil {
		t.Fatalf("open tokenizer: %v", err)
	}
	defer tok.Close()

	m, err := LoadModel(ckpt, tok)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	t.Logf("config: %+v", m.Config())
	t.Logf("vocab size: %d, maxTokenLength: %d", len(m.vocab), m.maxTokenLength)

	if int(m.config.VocabSize) != len(m.vocab) {
		t.Errorf("vocab size %d != loaded vocab entries %d", m.config.VocabSize, len(m.vocab))
	}
}
