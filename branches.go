package tinyoai

import (
	"context"
	"fmt"
)

// BranchRequest identifies a retained token prefix and saved alternatives to
// expand. Prefix is a cache fingerprint from TokenLogprob, not a credential.
type BranchRequest struct {
	// Position is the number of resident prefix tokens before the candidate.
	Position int
	// Prefix must match the fingerprint saved with the candidate's TokenLogprob.
	Prefix string
	// Tokens lists seed token IDs from that saved distribution, in branch order.
	Tokens []int
}

// Brancher is an optional capability. An open session holds the model's gate;
// callers must Close it before generating or opening another session.
type Brancher interface {
	// OpenBranches acquires a session against a saved prefix; callers must close
	// it to release the model gate.
	OpenBranches(context.Context, BranchRequest) (TokenBranches, error)
}

// TokenBranches advances independent suffixes against a read-only shared prefix.
// Each Forward accepts one token per branch and returns full-vocabulary logits.
// Use a session from one goroutine. Close is idempotent and releases its gate.
type TokenBranches interface {
	// Forward evaluates one token per branch using a non-nil step context. The
	// first call supplies the requested seeds.
	Forward(context.Context, []int) ([][]float32, error)
	// InitialLogprobs returns a copy of seed scores in branch order.
	InitialLogprobs() []float64
	// Remaining returns the fixed maximum suffix depth at session creation, not
	// a decrementing counter.
	Remaining() int
	// TokenPiece returns raw token bytes and false for invalid or special IDs. A
	// piece need not be valid UTF-8 in isolation.
	TokenPiece(int) (string, bool)
	// EncodeSuffix encodes text without BOS.
	EncodeSuffix(string) []int
	// Close releases suffix storage and the model gate; repeated calls are
	// harmless.
	Close() error
}

type mlxBranches interface {
	// Forward evaluates one token per branch and returns full-vocabulary rows.
	Forward(context.Context, []int) ([][]float32, error)
	// Close releases branch-owned suffix arrays, leaving shared prefix KV intact.
	Close() error
}
type mlxBrancher interface {
	// NewBranches borrows a valid prefix for a fixed number of suffixes while
	// the caller keeps the model gate.
	NewBranches(int, int) (mlxBranches, error)
}

type mlxTokenBranches struct {
	model     *MLX
	native    mlxBranches
	ctx       context.Context
	scores    []float64
	remaining int
	closed    bool
}

// OpenBranches validates saved alternatives against a resident prefix and
// holds the model gate until the returned session is closed. It rejects stale
// fingerprints and unsupported backends. A nil context means Background;
// cancellation while waiting releases no resources to the caller.
func (m *MLX) OpenBranches(ctx context.Context, req BranchRequest) (TokenBranches, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-m.gate:
	}
	release := true
	defer func() {
		if release {
			m.gate <- struct{}{}
		}
	}()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	pos := req.Position
	if m.closed || m.native == nil || pos < 1 || pos > len(m.tokens) || pos > len(m.probRows) || prefixFingerprint(m.tokens[:pos]) != req.Prefix {
		return nil, fmt.Errorf("the requested token prefix is no longer resident")
	}
	brancher, ok := m.native.(mlxBrancher)
	if !ok {
		return nil, fmt.Errorf("this backend does not support token branches")
	}
	row := m.probRows[pos-1]
	if len(req.Tokens) == 0 || len(req.Tokens) > len(row.IDs) {
		return nil, fmt.Errorf("branch seeds must come from the saved distribution")
	}
	scores := make([]float64, len(req.Tokens))
	for i, id := range req.Tokens {
		if id < 0 || id >= m.config.VocabSize || m.tokenizer.special[id] {
			return nil, fmt.Errorf("invalid branch token")
		}
		found := false
		for j, candidate := range row.IDs {
			if candidate == id {
				scores[i], found = row.Values[j], true
				break
			}
		}
		if !found {
			return nil, fmt.Errorf("token is no longer in the saved distribution")
		}
	}
	native, err := brancher.NewBranches(pos, len(req.Tokens))
	if err != nil {
		return nil, err
	}
	release = false
	return &mlxTokenBranches{model: m, native: native, ctx: ctx, scores: scores, remaining: m.config.MaxPositionEmbeddings - pos}, nil
}

// Forward advances every branch by one token after checking the opening
// context. The supplied step context must be non-nil. The session remains
// owned by the caller until Close, including after an error.
func (b *mlxTokenBranches) Forward(ctx context.Context, tokens []int) ([][]float32, error) {
	if b.closed {
		return nil, fmt.Errorf("token branches are closed")
	}
	if err := b.ctx.Err(); err != nil {
		return nil, err
	}
	return b.native.Forward(ctx, tokens)
}

// InitialLogprobs returns a copy of the saved seed log probabilities in branch
// order.
func (b *mlxTokenBranches) InitialLogprobs() []float64 { return append([]float64(nil), b.scores...) }

// Remaining returns the maximum suffix depth available when the session
// opened. It is a fixed bound, not a counter decremented by Forward.
func (b *mlxTokenBranches) Remaining() int { return b.remaining }

// TokenPiece returns the raw bytes of a nonspecial vocabulary token.
// Byte-fallback pieces may not be valid UTF-8 in isolation.
func (b *mlxTokenBranches) TokenPiece(id int) (string, bool) {
	if id < 0 || id >= b.model.config.VocabSize || b.model.tokenizer.special[id] {
		return "", false
	}
	return b.model.tokenizer.tokenBytes(id), true
}

// EncodeSuffix tokenizes continuation text without prepending BOS.
func (b *mlxTokenBranches) EncodeSuffix(text string) []int { return b.model.tokenizer.encode(text)[1:] }

// Close releases branch storage and the model gate exactly once, allowing
// generation or another branch session to proceed.
func (b *mlxTokenBranches) Close() error {
	if b.closed {
		return nil
	}
	b.closed = true
	defer func() { b.model.gate <- struct{}{} }()
	return b.native.Close()
}
