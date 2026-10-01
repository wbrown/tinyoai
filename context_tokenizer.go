package tinyoai

import (
	"fmt"

	"github.com/wbrown/tinyoai/tokenizerinfo"
)

// ContextTokenizer returns the StableLM tokenizer description and model
// context limit without evaluating weights. JSON borrows immutable tokenizer
// storage and must not be modified.
func (m *StableLM) ContextTokenizer() tokenizerinfo.Spec {
	return tokenizerinfo.Spec{JSON: m.tokenizer.core.Source(), VocabSize: m.config.VocabSize, BOS: m.config.BOSTokenID, ContextLength: m.config.MaxPositionEmbeddings}
}

// ContextTokenizer returns the MLX tokenizer description and model context
// limit without GPU evaluation. JSON borrows immutable tokenizer storage and
// must not be modified.
func (m *MLX) ContextTokenizer() tokenizerinfo.Spec {
	return tokenizerinfo.Spec{JSON: m.tokenizer.core.Source(), VocabSize: m.config.VocabSize, BOS: m.config.BOSTokenID, ContextLength: m.config.MaxPositionEmbeddings}
}

// validateContextTokenizer rejects a supplied tokenizer identity that differs
// from the loaded model. An empty identity opts out of this client-preparation
// check.
func validateContextTokenizer(opts GenerateOptions, id string) error {
	if opts.TokenizerID != "" && opts.TokenizerID != id {
		return fmt.Errorf("the supplied tokenizer identity differs from this model")
	}
	return nil
}

// TokenizerProvider exposes tokenizer assets and the model's context limit.
type TokenizerProvider interface {
	// ContextTokenizer returns tokenizer assets and limits; callers must not
	// mutate storage borrowed from the provider.
	ContextTokenizer() tokenizerinfo.Spec
}
