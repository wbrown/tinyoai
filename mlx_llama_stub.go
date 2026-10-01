//go:build !mlx || !darwin || !arm64 || !cgo

package tinyoai

import "fmt"

// newLlamaMLX reports that this build has no native MLX backend.
func newLlamaMLX(*Model) (*LlamaMLX, error) {
	return nil, fmt.Errorf("Llama MLX requires Apple silicon, cgo and -tags mlx; use Default for pure Go inference")
}
