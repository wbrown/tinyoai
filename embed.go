package tinyoai

import (
	"bytes"
	_ "embed"
	"sync"
)

// The embedded default model is Andrej Karpathy's stories260K — a ~260K-param
// Llama 2 (GQA: 8 query heads, 4 kv heads) trained on TinyStories and published
// as a unit-test model (github.com/karpathy/tinyllamas, MIT). Weights ~1MB,
// tokenizer ~6KB. The legacy GQA layout is verified by TestLoadDefaultModel,
// which fails if the on-disk weights are not consumed exactly.

//go:embed assets/stories260K.bin
var defaultCheckpoint []byte

//go:embed assets/tok512.bin
var defaultTokenizer []byte

var (
	defaultOnce  sync.Once
	defaultModel *Model
	defaultErr   error
)

// Default returns the embedded stories260K model, loading it once on first use.
func Default() (*Model, error) {
	defaultOnce.Do(func() {
		defaultModel, defaultErr = LoadModel(
			bytes.NewReader(defaultCheckpoint),
			bytes.NewReader(defaultTokenizer),
		)
	})
	return defaultModel, defaultErr
}
