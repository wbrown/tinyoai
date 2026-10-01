package tinyoai

import (
	"os"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/wbrown/tinyoai/tokenizer"
	"github.com/wbrown/tinyoai/tokenizerinfo"
)

// Inference adds decoding and probability reporting to the shared encoder.
type nerdstashTokenizer struct {
	core      *tokenizer.Tokenizer
	vocab     map[string]int
	pieces    []string
	special   map[int]bool
	contextID string // Immutable fingerprint; computed once when loading assets.
}

// loadNerdstash reads and validates the shared encoder, then attaches
// immutable decoding views and a canonical tokenizer identity for inference.
func loadNerdstash(path string, vocabSize, bos int) (*nerdstashTokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	core, err := tokenizer.Parse(data, vocabSize, bos)
	if err != nil {
		return nil, err
	}
	id := (tokenizerinfo.Spec{JSON: data, VocabSize: vocabSize, BOS: bos}).ID()
	return &nerdstashTokenizer{core: core, vocab: core.Vocab, pieces: core.Pieces, special: core.Special, contextID: id}, nil
}

// encode returns shared-encoder token IDs with the configured BOS token
// prepended.
func (t *nerdstashTokenizer) encode(text string) []int { return t.core.Encode(text) }

// nerdstashDecoder buffers consecutive byte-fallback tokens. Hugging Face's
// ByteFallback decoder replaces *every byte* in an invalid run with U+FFFD,
// including any valid prefix; wait for the end of a run to preserve parity.
// Special tokens are skipped without interrupting the run.
type nerdstashDecoder struct {
	tokenizer *nerdstashTokenizer
	bytes     []byte
}

// add consumes one token and returns text whose byte-fallback run is complete.
// Special and unused tokens emit nothing; ordinary pieces flush pending bytes
// before replacing the space marker.
func (d *nerdstashDecoder) add(id int) string {
	t := d.tokenizer
	if t.special[id] || t.pieces[id] == "" {
		return ""
	}
	p := t.pieces[id]
	if len(p) == 6 && strings.HasPrefix(p, "<0x") && p[5] == '>' {
		if b, err := strconv.ParseUint(p[3:5], 16, 8); err == nil {
			d.bytes = append(d.bytes, byte(b))
			return ""
		}
	}
	return d.flush() + strings.ReplaceAll(p, "▁", " ")
}

// flush emits and clears the pending byte-fallback run. If the entire run is
// invalid UTF-8, each byte becomes a replacement character, matching the
// reference decoder.
func (d *nerdstashDecoder) flush() string {
	var text string
	if utf8.Valid(d.bytes) {
		text = string(d.bytes)
	} else {
		text = strings.Repeat("�", len(d.bytes))
	}
	d.bytes = d.bytes[:0]
	return text
}
