package tinyoai

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
)

const cacheMagic = "TNYKV001"

type cacheHeader struct {
	Model  [32]byte `json:"model_sha256"`
	Half   bool     `json:"float16"`
	Tokens []int    `json:"tokens"`
}

// modelIdentity is called while owning m.cache. It binds saved state to the
// exact configuration and tensor bytes, independently of filenames.
func (m *StableLM) modelIdentity() [32]byte {
	if m.cacheIdentity != [32]byte{} {
		return m.cacheIdentity
	}
	h := sha256.New()
	config, _ := json.Marshal(m.config)
	h.Write(config)
	weights := []weightTensor{m.embed, m.head, m.norm, m.bias}
	for _, l := range m.layers {
		weights = append(weights, l.norm, l.bias, l.q, l.k, l.v, l.o, l.gate, l.up, l.down)
	}
	for _, w := range weights {
		meta, _ := json.Marshal(struct {
			Type  string
			Shape []int
		}{w.dtype, w.shape})
		h.Write(meta)
		h.Write(w.data)
	}
	copy(m.cacheIdentity[:], h.Sum(nil))
	return m.cacheIdentity
}

// SaveCache atomically saves the completed token prefix and its KV state.
// It waits for active generation to finish; incomplete layers are never saved.
// The file contains prompt-derived data and is created with mode 0600.
func (m *StableLM) SaveCache(path string) error {
	saved := <-m.cache
	defer func() { m.cache <- saved }()
	header := cacheHeader{Model: m.modelIdentity(), Half: m.halfKV}
	if saved != nil {
		header.Tokens = saved.tokens
	}
	data, err := json.Marshal(header)
	if err != nil {
		return err
	}
	f, err := os.CreateTemp(filepath.Dir(path), ".tinyoai-kv-*")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	defer f.Close()
	buffer := bufio.NewWriterSize(f, 256<<10)
	digest := sha256.New()
	w := io.MultiWriter(buffer, digest)
	prefix := make([]byte, len(cacheMagic)+4)
	copy(prefix, cacheMagic)
	binary.LittleEndian.PutUint32(prefix[len(cacheMagic):], uint32(len(data)))
	if _, err = w.Write(prefix); err != nil {
		return err
	}
	if _, err = w.Write(data); err != nil {
		return err
	}
	if len(header.Tokens) != 0 {
		if err = m.cachePayload(nil, w, saved); err != nil {
			return err
		}
	}
	if _, err = buffer.Write(digest.Sum(nil)); err != nil {
		return err
	}
	if err = buffer.Flush(); err != nil {
		return err
	}
	if err = f.Sync(); err != nil {
		return err
	}
	if err = f.Close(); err != nil {
		return err
	}
	return os.Rename(f.Name(), path)
}

// LoadCache verifies a snapshot before replacing the retained context. Errors
// leave the existing cache intact. The model and KV precision must match.
func (m *StableLM) LoadCache(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	saved := <-m.cache
	defer func() { m.cache <- saved }()
	buffer := bufio.NewReaderSize(f, 256<<10)
	digest := sha256.New()
	r := io.TeeReader(buffer, digest)
	prefix := make([]byte, len(cacheMagic)+4)
	if _, err = io.ReadFull(r, prefix); err != nil {
		return err
	}
	if string(prefix[:len(cacheMagic)]) != cacheMagic {
		return fmt.Errorf("unsupported KV cache format")
	}
	n := binary.LittleEndian.Uint32(prefix[len(cacheMagic):])
	if n > 1<<20 {
		return fmt.Errorf("KV cache header too large")
	}
	data := make([]byte, int(n))
	if _, err = io.ReadFull(r, data); err != nil {
		return err
	}
	var header cacheHeader
	if err = json.Unmarshal(data, &header); err != nil {
		return err
	}
	if len(header.Tokens) > m.config.MaxPositionEmbeddings {
		return fmt.Errorf("KV cache exceeds model context")
	}
	for _, id := range header.Tokens {
		if id < 0 || id >= m.config.VocabSize {
			return fmt.Errorf("invalid KV cache token %d", id)
		}
	}
	if header.Model != m.modelIdentity() {
		return fmt.Errorf("KV cache model does not match checkpoint")
	}
	if header.Half != m.halfKV {
		return fmt.Errorf("KV cache precision does not match model options")
	}
	elementBytes := uint64(4)
	if header.Half {
		elementBytes = 2
	}
	c := m.config
	values := uint64(c.NumHiddenLayers) * 2 * uint64(c.NumKeyValueHeads) * uint64(c.HiddenSize/c.NumAttentionHeads) * uint64(len(header.Tokens))
	stat, err := f.Stat()
	if err != nil {
		return err
	}
	expected := uint64(len(prefix)) + uint64(n) + values*elementBytes + sha256.Size
	if uint64(stat.Size()) != expected {
		return fmt.Errorf("KV cache size %d, expected %d", stat.Size(), expected)
	}
	restored := &stableContext{}
	restored.resize(m, max(1, len(header.Tokens)))
	restored.tokens = header.Tokens
	if err = m.cachePayload(r, nil, restored); err != nil {
		return err
	}
	checksum := make([]byte, sha256.Size)
	if _, err = io.ReadFull(buffer, checksum); err != nil {
		return err
	}
	if !bytes.Equal(checksum, digest.Sum(nil)) {
		return fmt.Errorf("KV cache checksum mismatch")
	}
	saved = restored
	return nil
}

// cachePayload reads or writes populated KV positions in
// layer/head/key-or-value order using a bounded buffer. Exactly one of r and w
// is nonnil. Capacity padding is omitted, so snapshots do not depend on the
// resident head stride.
func (m *StableLM) cachePayload(r io.Reader, w io.Writer, saved *stableContext) error {
	if len(saved.tokens) == 0 {
		return nil
	}
	s := saved.state
	c := m.config
	head, capacity := c.HiddenSize/c.NumAttentionHeads, len(s.att)/c.NumAttentionHeads
	buffer := make([]byte, 64<<10)
	for l := 0; l < c.NumHiddenLayers; l++ {
		for h := 0; h < c.NumKeyValueHeads; h++ {
			start, end := h*capacity*head, (h*capacity+len(saved.tokens))*head
			for kind := 0; kind < 2; kind++ {
				var full []float32
				var half []uint16
				if m.halfKV {
					if kind == 0 {
						half = s.keys16[l][start:end]
					} else {
						half = s.values16[l][start:end]
					}
				} else {
					if kind == 0 {
						full = s.keys[l][start:end]
					} else {
						full = s.values[l][start:end]
					}
				}
				if err := cacheValues(r, w, full, half, buffer); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

// cacheValues transfers one KV slice as little-endian float32 or raw float16
// bits. A nonnil r selects decoding; otherwise w receives encoded values.
// Buffer must hold at least one element and is reused for the entire slice.
func cacheValues(r io.Reader, w io.Writer, full []float32, half []uint16, buffer []byte) error {
	size, count := 4, len(full)
	if half != nil {
		size, count = 2, len(half)
	}
	for offset := 0; offset < count; {
		n := min(count-offset, len(buffer)/size)
		data := buffer[:n*size]
		if r != nil {
			if _, err := io.ReadFull(r, data); err != nil {
				return err
			}
		}
		for i := 0; i < n; i++ {
			at := data[i*size:]
			if r != nil {
				if half != nil {
					half[offset+i] = binary.LittleEndian.Uint16(at)
				} else {
					full[offset+i] = math.Float32frombits(binary.LittleEndian.Uint32(at))
				}
			} else {
				if half != nil {
					binary.LittleEndian.PutUint16(at, half[offset+i])
				} else {
					binary.LittleEndian.PutUint32(at, math.Float32bits(full[offset+i]))
				}
			}
		}
		if w != nil {
			if _, err := w.Write(data); err != nil {
				return err
			}
		}
		offset += n
	}
	return nil
}
