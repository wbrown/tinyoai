package tinyoai

import (
	"context"
	"errors"
	"reflect"
	"testing"
	"time"
)

type branchForwarder struct {
	contextForwarder
	closed int
	fail   bool
}
type fakeTokenBranches struct {
	owner *branchForwarder
	count int
}

// NewBranches creates a fake branch batch or injects an allocation failure to
// test gate cleanup.
func (f *branchForwarder) NewBranches(_ int, count int) (mlxBranches, error) {
	if f.fail {
		return nil, errors.New("branch allocation")
	}
	return &fakeTokenBranches{owner: f, count: count}, nil
}

// Forward returns one zero-logit row per fake branch unless the step context
// is cancelled.
func (b *fakeTokenBranches) Forward(ctx context.Context, ids []int) ([][]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	rows := make([][]float32, b.count)
	for i := range rows {
		rows[i] = make([]float32, b.owner.vocab)
	}
	return rows, nil
}

// Close counts native-session releases so the public wrapper's idempotence can
// be checked.
func (b *fakeTokenBranches) Close() error { b.owner.closed++; return nil }

// TestPublicBranchesOwnGateAndPreservePrefix verifies stale-prefix rejection,
// exclusive session ownership, cancellation, copied seed scores, and
// exactly-once release without changing document KV metadata.
func TestPublicBranchesOwnGateAndPreservePrefix(t *testing.T) {
	m, err := newMLX("testdata/stablelm")
	if err != nil {
		t.Fatal(err)
	}
	f := &branchForwarder{contextForwarder: contextForwarder{vocab: m.config.VocabSize}}
	m.native = f
	m.tokens = m.tokenizer.encode("The cat")
	id := m.tokenizer.encode("a")[1]
	m.probRows = make([]logprobRow, len(m.tokens))
	m.probRows[len(m.tokens)-1] = logprobRow{IDs: []int{id}, Values: []float64{-1.25}}
	req := BranchRequest{Position: len(m.tokens), Prefix: prefixFingerprint(m.tokens), Tokens: []int{id}}
	before := append([]int(nil), m.tokens...)
	rows := append([]logprobRow(nil), m.probRows...)
	// Native setup errors and stale identities must release the semaphore.
	for _, fail := range []bool{true, false} {
		f.fail = fail
		bad := req
		if !fail {
			bad.Prefix = "stale"
		}
		if _, err = m.OpenBranches(context.Background(), bad); err == nil {
			t.Fatal("invalid open accepted")
		}
		if len(m.gate) != 1 {
			t.Fatal("error retained model gate")
		}
	}
	b, err := m.OpenBranches(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if b.Remaining() != m.config.MaxPositionEmbeddings-len(m.tokens) || b.InitialLogprobs()[0] != -1.25 {
		t.Fatal("lost branch metadata")
	}
	scores := b.InitialLogprobs()
	scores[0] = 0
	if b.InitialLogprobs()[0] != -1.25 {
		t.Fatal("caller mutated saved scores")
	}
	if _, ok := b.TokenPiece(-1); ok {
		t.Fatal("invalid token accepted")
	}
	if _, err = b.Forward(context.Background(), req.Tokens); err != nil {
		t.Fatal(err)
	}
	queued, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if _, err = m.OpenBranches(queued, req); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatal("queued caller failed to cancel", err)
	}
	if err = b.Close(); err != nil {
		t.Fatal(err)
	}
	if err = b.Close(); err != nil || f.closed != 1 || len(m.gate) != 1 {
		t.Fatal("close did not release exactly once")
	}
	if _, err = b.Forward(context.Background(), req.Tokens); err == nil {
		t.Fatal("forward after close")
	}
	if !reflect.DeepEqual(m.tokens, before) || !reflect.DeepEqual(m.probRows, rows) {
		t.Fatal("branch altered document state")
	}
	canceled, stop := context.WithCancel(context.Background())
	b, err = m.OpenBranches(canceled, req)
	if err != nil {
		t.Fatal(err)
	}
	stop()
	if _, err = b.Forward(context.Background(), req.Tokens); !errors.Is(err, context.Canceled) {
		t.Fatal("session ignored original cancellation", err)
	}
	_ = b.Close()
}
