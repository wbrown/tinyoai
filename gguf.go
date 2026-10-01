package tinyoai

// A narrow GGUF v3 reader for StableLM and Clio's published Q8/Q6/Q5 variants.
// The original tokenizer.json is a sidecar because Clio's published GGUF
// tokenizer has different added-token whitespace semantics.

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

type ggufTensor struct {
	shape        []int
	dtype        string
	offset, size int64
}

type ggufReader struct {
	r   io.Reader
	err error
}

// number reads a little-endian scalar or fixed layout into v unless an earlier
// read failed. The first I/O error is retained in r.err for the enclosing
// parser.
func (r *ggufReader) number(v any) {
	if r.err == nil {
		r.err = binary.Read(r.r, binary.LittleEndian, v)
	}
}

// u32 reads a little-endian uint32, returning zero if the reader has already
// failed. The caller checks r.err after the enclosing record.
func (r *ggufReader) u32() (n uint32) { r.number(&n); return }

// u64 reads a little-endian uint64, retaining read failures in r.err.
func (r *ggufReader) u64() (n uint64) { r.number(&n); return }

// str reads a length-prefixed GGUF string, rejecting lengths above the header
// parser's 16 MiB limit before allocating.
func (r *ggufReader) str() string {
	n := r.u64()
	if n > 16<<20 {
		r.err = fmt.Errorf("GGUF string too large")
		return ""
	}
	if r.err != nil {
		return ""
	}
	b := make([]byte, int(n))
	_, r.err = io.ReadFull(r.r, b)
	return string(b)
}

// value decodes a GGUF metadata value, normalizing integer and floating types
// to 64-bit Go values. Arrays are bounded and cannot nest; malformed or
// unsupported values set r.err.
func (r *ggufReader) value(kind uint32) any {
	switch kind {
	case 0:
		var v uint8
		r.number(&v)
		return uint64(v)
	case 1:
		var v int8
		r.number(&v)
		return int64(v)
	case 2:
		var v uint16
		r.number(&v)
		return uint64(v)
	case 3:
		var v int16
		r.number(&v)
		return int64(v)
	case 4:
		return uint64(r.u32())
	case 5:
		var v int32
		r.number(&v)
		return int64(v)
	case 6:
		var v float32
		r.number(&v)
		return float64(v)
	case 7:
		var v uint8
		r.number(&v)
		if v > 1 {
			r.err = fmt.Errorf("invalid GGUF boolean")
		}
		return v == 1
	case 8:
		return r.str()
	case 9:
		kind, n := r.u32(), r.u64()
		if kind == 9 || n > 1<<20 {
			r.err = fmt.Errorf("unsupported GGUF array")
			return nil
		}
		if r.err != nil {
			return nil
		}
		values := make([]any, int(n))
		for i := range values {
			values[i] = r.value(kind)
			if r.err != nil {
				break
			}
		}
		return values
	case 10:
		return r.u64()
	case 11:
		var v int64
		r.number(&v)
		return v
	case 12:
		var v float64
		r.number(&v)
		return v
	default:
		r.err = fmt.Errorf("unsupported GGUF metadata type %d", kind)
		return nil
	}
}

// readGGUF validates a little-endian GGUF v3 header and returns metadata plus
// tensor locations. Shapes are converted from GGUF dimension order to
// row-major engine order, and offsets become absolute file positions. It
// checks alignment, bounds, overlap, and supported quantization without
// loading weights.
func readGGUF(f *os.File) (map[string]any, map[string]ggufTensor, error) {
	stat, err := f.Stat()
	if err != nil {
		return nil, nil, err
	}
	section := io.NewSectionReader(f, 0, min(stat.Size(), 64<<20))
	r := ggufReader{r: section}
	if r.u32() != 0x46554747 || r.u32() != 3 {
		return nil, nil, fmt.Errorf("expected little-endian GGUF v3")
	}
	nTensors, nMeta := r.u64(), r.u64()
	if nTensors == 0 || nTensors > 10000 || nMeta > 65536 {
		return nil, nil, fmt.Errorf("invalid GGUF counts")
	}
	meta := make(map[string]any)
	for i := uint64(0); i < nMeta && r.err == nil; i++ {
		key := r.str()
		if _, exists := meta[key]; exists {
			return nil, nil, fmt.Errorf("duplicate GGUF key %q", key)
		}
		meta[key] = r.value(r.u32())
	}
	tensors := make(map[string]ggufTensor)
	for i := uint64(0); i < nTensors && r.err == nil; i++ {
		name, rank := r.str(), r.u32()
		if rank == 0 || rank > 4 {
			return nil, nil, fmt.Errorf("invalid GGUF tensor rank")
		}
		t := ggufTensor{shape: make([]int, int(rank))}
		n := int64(1)
		for j := range t.shape {
			d := r.u64()
			if d == 0 || d > uint64(math.MaxInt64/n)/4 || d > uint64(int(^uint(0)>>1)) {
				return nil, nil, fmt.Errorf("invalid GGUF tensor dimensions")
			}
			n *= int64(d)
			t.shape[len(t.shape)-1-j] = int(d) // GGUF stores the contiguous dimension first.
		}
		switch kind := r.u32(); kind {
		case 0:
			t.dtype, t.size = "F32", n*4
		case 1:
			t.dtype, t.size = "F16", n*2
		case 30:
			t.dtype, t.size = "BF16", n*2
		case 8:
			if rank != 2 || t.shape[1]%32 != 0 {
				return nil, nil, fmt.Errorf("Q8_0 rows must be divisible by 32")
			}
			t.dtype, t.size = "Q8_0", n/32*34
		case 7:
			if rank != 2 || t.shape[1]%32 != 0 {
				return nil, nil, fmt.Errorf("Q5_1 rows must be divisible by 32")
			}
			t.dtype, t.size = "Q5_1", n/32*24
		case 13, 14:
			if rank != 2 || t.shape[1]%256 != 0 {
				return nil, nil, fmt.Errorf("K-quant rows must be divisible by 256")
			}
			t.dtype = "Q5_K"
			if kind == 14 {
				t.dtype = "Q6_K"
			}
			t.size = n / 256 * int64(kBlockBytes(t.dtype))
		default:
			return nil, nil, fmt.Errorf("GGUF tensor %s: unsupported type %d", name, kind)
		}
		offset := r.u64()
		if offset > math.MaxInt64 || t.size > int64(int(^uint(0)>>1)) {
			return nil, nil, fmt.Errorf("GGUF tensor too large")
		}
		t.offset = int64(offset)
		if _, exists := tensors[name]; exists {
			return nil, nil, fmt.Errorf("duplicate GGUF tensor %q", name)
		}
		tensors[name] = t
	}
	if r.err != nil {
		return nil, nil, fmt.Errorf("GGUF header: %w", r.err)
	}
	align := uint64(32)
	if v, exists := meta["general.alignment"]; exists {
		var ok bool
		align, ok = v.(uint64)
		if !ok {
			align = 0
		}
	}
	if align == 0 || align > 4096 || align&(align-1) != 0 {
		return nil, nil, fmt.Errorf("invalid GGUF alignment")
	}
	pos, _ := section.Seek(0, io.SeekCurrent)
	start := (pos + int64(align) - 1) & -int64(align)
	ranges := make([]ggufTensor, 0, len(tensors))
	for name, t := range tensors {
		if t.offset%int64(align) != 0 || start > stat.Size() || t.offset > stat.Size()-start || t.size > stat.Size()-start-t.offset {
			return nil, nil, fmt.Errorf("GGUF tensor %s outside file or misaligned", name)
		}
		t.offset += start
		tensors[name] = t
		ranges = append(ranges, t)
	}
	sort.Slice(ranges, func(i, j int) bool { return ranges[i].offset < ranges[j].offset })
	for i := 1; i < len(ranges); i++ {
		if ranges[i].offset < ranges[i-1].offset+ranges[i-1].size {
			return nil, nil, fmt.Errorf("overlapping GGUF tensors")
		}
	}
	return meta, tensors, nil
}

var ggufNames = strings.NewReplacer(
	"model.embed_tokens", "token_embd", "lm_head", "output", "model.norm", "output_norm",
	"model.layers.", "blk.", "input_layernorm", "attn_norm",
	"self_attn.q_proj", "attn_q", "self_attn.k_proj", "attn_k", "self_attn.v_proj", "attn_v", "self_attn.o_proj", "attn_output",
	"mlp.gate_proj", "ffn_gate", "mlp.up_proj", "ffn_up", "mlp.down_proj", "ffn_down",
)

// loadStableGGUF translates a validated GGUF StableLM checkpoint into the
// common weight loader. The original tokenizer.json beside the file is
// required and checked against the GGUF vocabulary, preserving added-token
// whitespace behavior.
func loadStableGGUF(path string) (*StableLM, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	meta, tensors, err := readGGUF(f)
	if err != nil {
		return nil, err
	}
	if meta["general.architecture"] != "stablelm" || (meta["general.type"] != nil && meta["general.type"] != "model") {
		return nil, fmt.Errorf("GGUF must contain a StableLM model")
	}
	var metaErr error
	integer := func(key string) int {
		n, ok := meta[key].(uint64)
		if !ok || n > uint64(int(^uint(0)>>1)) {
			metaErr = fmt.Errorf("missing or invalid GGUF integer %s", key)
			return 0
		}
		return int(n)
	}
	c := StableLMConfig{ModelType: "stablelm", HiddenAct: "silu", RopeTheta: 10000,
		HiddenSize: integer("stablelm.embedding_length"), IntermediateSize: integer("stablelm.feed_forward_length"), NumHiddenLayers: integer("stablelm.block_count"),
		NumAttentionHeads: integer("stablelm.attention.head_count"), NumKeyValueHeads: integer("stablelm.attention.head_count_kv"), MaxPositionEmbeddings: integer("stablelm.context_length"),
		BOSTokenID: integer("tokenizer.ggml.bos_token_id"), EOSTokenID: integer("tokenizer.ggml.eos_token_id")}
	c.UseParallelResidual, _ = meta["stablelm.use_parallel_residual"].(bool)
	c.LayerNormEps, _ = meta["stablelm.attention.layer_norm_epsilon"].(float64)
	if theta, exists := meta["stablelm.rope.freq_base"]; exists {
		var ok bool
		c.RopeTheta, ok = theta.(float64)
		if !ok {
			return nil, fmt.Errorf("invalid GGUF RoPE frequency base")
		}
	}
	if c.HiddenSize > 0 && c.NumAttentionHeads > 0 {
		c.PartialRotaryFactor = float64(integer("stablelm.rope.dimension_count")) * float64(c.NumAttentionHeads) / float64(c.HiddenSize)
	}
	vocab, ok := meta["tokenizer.ggml.tokens"].([]any)
	if !ok {
		return nil, fmt.Errorf("GGUF missing vocabulary")
	}
	c.VocabSize = len(vocab)
	if metaErr != nil {
		return nil, metaErr
	}
	if err := c.validate(); err != nil {
		return nil, err
	}
	// Two embedding/output matrices, two final norm vectors, and nine tensors
	// per layer. Check before allocating layers from untrusted metadata.
	if len(tensors) < 4 || (len(tensors)-4)%9 != 0 || (len(tensors)-4)/9 != c.NumHiddenLayers {
		return nil, fmt.Errorf("unexpected GGUF tensor count for %d StableLM layers", c.NumHiddenLayers)
	}
	if scaling, ok := meta["stablelm.rope.scaling.type"]; ok && scaling != "none" {
		return nil, fmt.Errorf("GGUF RoPE scaling is not supported")
	}
	// Reject variants whose extra tensors would otherwise be silently ignored.
	for name := range tensors {
		if strings.Contains(name, "attn_q_norm") || strings.Contains(name, "attn_k_norm") || strings.Contains(name, "ffn_norm") || (strings.HasSuffix(name, ".bias") && !strings.Contains(name, "norm")) {
			return nil, fmt.Errorf("unsupported StableLM tensor %s", name)
		}
	}
	tok, err := loadNerdstash(filepath.Join(filepath.Dir(path), "tokenizer.json"), c.VocabSize, c.BOSTokenID)
	if err != nil {
		return nil, fmt.Errorf("GGUF requires the original tokenizer.json beside it: %w", err)
	}
	tokenTypes, _ := meta["tokenizer.ggml.token_type"].([]any)
	for i, value := range vocab {
		piece, ok := value.(string)
		// GGUF pads an unused output row that has no tokenizer entry. Keep
		// the original empty decoding behavior rather than adding a new token.
		if ok && tok.pieces[i] == "" && piece == fmt.Sprintf("[PAD%d]", i) && len(tokenTypes) == len(vocab) && (tokenTypes[i] == int64(5) || tokenTypes[i] == uint64(5)) {
			continue
		}
		if !ok || (piece != tok.pieces[i] && strings.ReplaceAll(piece, " ", "▁") != tok.pieces[i]) {
			return nil, fmt.Errorf("GGUF/tokenizer vocabulary mismatch at %d", i)
		}
	}
	return loadStableWeights(c, tok, func(name string, shape ...int) (weightTensor, error) {
		t, ok := tensors[ggufNames.Replace(name)]
		if !ok {
			return weightTensor{}, fmt.Errorf("missing GGUF tensor %s", name)
		}
		if len(shape) != len(t.shape) {
			return weightTensor{}, fmt.Errorf("GGUF tensor %s has wrong rank", name)
		}
		for i := range shape {
			if shape[i] != t.shape[i] {
				return weightTensor{}, fmt.Errorf("GGUF tensor %s: shape %v, want %v", name, t.shape, shape)
			}
		}
		w := weightTensor{dtype: t.dtype, shape: shape, data: make([]byte, int(t.size))}
		_, err := f.ReadAt(w.data, t.offset)
		return w, err
	})
}
