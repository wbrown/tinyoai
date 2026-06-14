package tinyoai

import (
	"encoding/binary"
	"os"
	"testing"
)

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
