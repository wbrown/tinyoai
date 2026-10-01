// Package tokenizerinfo describes tokenizer assets independently of clients and engines.
package tokenizerinfo

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
)

// Spec contains only tokenizer assets; no model weights are needed to build context.
type Spec struct {
	// JSON contains the tokenizer pipeline and vocabulary; providers may lend
	// immutable storage.
	JSON json.RawMessage `json:"tokenizer"`
	// VocabSize includes any padding rows in the model's vocabulary.
	VocabSize int `json:"vocab_size"`
	// BOS is the token ID prepended to an encoded prompt.
	BOS int `json:"bos_token_id"`
	// ContextLength is the model position limit; it does not affect tokenizer
	// identity.
	ContextLength int `json:"context_length"`
}

// ID hashes vocabulary size, BOS, and canonical tokenizer JSON. Equivalent
// JSON formatting produces the same identity; context length is intentionally
// excluded because it does not change tokenization. Invalid JSON is hashed
// verbatim.
func (s Spec) ID() string {
	// The asset travels through JSON encoders in both Go and JavaScript. Hash
	// its canonical contents so whitespace, key order and escaping do not matter.
	var value any
	data := s.JSON
	if json.Unmarshal(data, &value) == nil {
		if canonical, err := json.Marshal(value); err == nil {
			data = canonical
		}
	}
	h := sha256.New()
	fmt.Fprintf(h, "%d:%d:", s.VocabSize, s.BOS)
	h.Write(data)
	return hex.EncodeToString(h.Sum(nil))
}
