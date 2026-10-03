package tinyoai

import (
	"context"
	"fmt"
	"io"
)

// GenerationSession owns an independent generation cache. Close releases that
// cache and its references to shared weights without closing its parent.
type GenerationSession interface {
	Generator
	io.Closer
}

// Forker creates a generation session from the currently resident token prefix.
// Implementations may share immutable weights and use copy-on-write KV storage.
type Forker interface {
	// Fork snapshots the retained prefix after earlier session work finishes.
	// The caller owns the child and must close it. Parent and child then generate
	// independently; closing either leaves the other usable. Waiting can cancel.
	Fork(ctx context.Context) (GenerationSession, error)
}

// mlxForker retains an independently owned native state under the model gate.
type mlxForker interface {
	// Fork retains independent native handles while sharing tensor storage.
	// The caller holds the parent's session gate and owns the returned forwarder.
	Fork() (mlxForwarder, error)
}

// Fork snapshots the resident native cache and probability records. The child
// shares weight storage, has its own request gate, and can outlive its parent.
// KV initially shares storage; updating a shared layer can copy its full cache.
// Unsupported native architectures and the Python backend return an error.
func (m *MLX) Fork(ctx context.Context) (GenerationSession, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-m.gate:
	}
	defer func() { m.gate <- struct{}{} }()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if m.closed {
		return nil, fmt.Errorf("MLX backend is closed")
	}
	backend, ok := m.native.(mlxForker)
	if !ok {
		return nil, fmt.Errorf("this backend does not support independent generation sessions")
	}
	native, err := backend.Fork()
	if err != nil {
		return nil, err
	}
	child := &MLX{dir: m.dir, config: m.config, tokenizer: m.tokenizer,
		gate: make(chan struct{}, 1), native: native, prefillChunk: m.prefillChunk,
		tokens: append([]int(nil), m.tokens...), probCount: m.probCount,
		probRows: append([]logprobRow(nil), m.probRows...)}
	// Rows are replaced during generation, but own their slices as well so a
	// future in-place probability update cannot modify the parent's records.
	for i := range child.probRows {
		child.probRows[i].IDs = append([]int(nil), m.probRows[i].IDs...)
		child.probRows[i].Values = append([]float64(nil), m.probRows[i].Values...)
	}
	child.gate <- struct{}{}
	return child, nil
}
