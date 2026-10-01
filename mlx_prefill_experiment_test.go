//go:build mlx && darwin && arm64 && cgo

package tinyoai

import (
	"context"
	mx "github.com/wbrown/tinyoai/internal/mlx"
	"github.com/wbrown/tinyoai/internal/mlxbench"
	"math"
	"os"
	"reflect"
	"strings"
	"testing"
)

// TestMLXLayerScheduling requires exact logits across evaluation-boundary and
// profiling experiments through cache growth, rewind, and decode with both KV
// precisions.
func TestMLXLayerScheduling(t *testing.T) {
	dir := os.Getenv("TINYOAI_MLX_DIR")
	if dir == "" {
		t.Skip("set TINYOAI_MLX_DIR")
	}
	m, err := LoadMLXNative(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	n := m.native.(*nativeMLX)
	ids := m.tokenizer.encode(strings.Repeat("Mara watched the lighthouse as the rain crossed the empty road. ", 60))
	for _, bits := range []int{8, 16} {
		if _, err := n.SetKVBits(bits); err != nil {
			t.Fatal(err)
		}
		var baseline [][]float32
		for _, batch := range []int{0, 1, 4, 7, 14, -1, -2} {
			if err := mx.Run(n.reset); err != nil {
				t.Fatal(err)
			}
			n.layerBatch, n.onPrefillPhase = max(batch, 0), nil
			n.outputRoot = batch == -2
			phaseCount := 0
			if batch == -1 {
				n.onPrefillPhase = func(MLXPrefillPhase) { phaseCount++ }
			}
			for step, span := range [][2]int{{0, 256}, {256, 257}, {100, 17}, {117, 1}} {
				z, err := n.Forward(context.Background(), span[0], ids[span[0]:span[0]+span[1]], true)
				if err != nil {
					t.Fatal(err)
				}
				if batch == 0 {
					baseline = append(baseline, z)
				} else if !reflect.DeepEqual(z, baseline[step]) {
					t.Fatalf("KV%d batch%d step%d changed logits (KL %.12f)", bits, batch, step, mlxbench.KL(baseline[step], z))
				}
			}
			if batch == -1 && phaseCount != 3*len(n.layers)*8 {
				t.Fatalf("got %d phase records", phaseCount)
			}
		}
		n.onPrefillPhase = nil
	}
}

// The cancellation arrives just after Forward's admission check. Its already
// admitted chunk must return complete probabilities along with completed KV.
type cancelAfterAdmission struct {
	context.Context
	cancel context.CancelFunc
}

// TestMLXBlockedTopIndices compares hierarchical selection with the
// full-vocabulary MLX operation, including tied cutoffs and several shortlist
// and block sizes.
func TestMLXBlockedTopIndices(t *testing.T) {
	if os.Getenv("TINYOAI_MLX_DIR") == "" {
		t.Skip("requires Metal")
	}
	err := mx.Run(func() {
		ctx := mx.New()
		defer ctx.Close()
		a := &mx.Arena{Context: ctx}
		defer a.Free()
		values := make([]int, 4*65536)
		for i := range values {
			values[i] = (i*193 + 7919) % 100003
		}
		// Include tied values, including a whole row tied at the cutoff.
		for i := 65536; i < 2*65536; i++ {
			values[i] = i % 29
		}
		for i := 2 * 65536; i < 3*65536; i++ {
			values[i] = 0
		}
		x := a.Reshape(a.Cast(a.Tokens(values), mx.Float32), 1, 4, 65536)
		for _, count := range []int{1, 12, 64} {
			want := a.Contiguous(a.Cast(a.TopIndices(x, count), mx.Float32))
			mx.Eval(want)
			for _, block := range []int{256, 512, 1024} {
				got := a.Contiguous(a.Cast(a.BlockTopIndices(x, count, block), mx.Float32))
				mx.Eval(got)
				if !reflect.DeepEqual(want.Floats(), got.Floats()) {
					t.Errorf("different selected IDs/order for count %d block %d", count, block)
				}
			}
		}
	})
	if err != nil {
		t.Fatal(err)
	}
}

// TestMLXPackedProjections verifies that packed views leave unfused output
// unchanged and that fused projections preserve greedy choice with small
// full-vocabulary divergence.
func TestMLXPackedProjections(t *testing.T) {
	dir := os.Getenv("TINYOAI_MLX_DIR")
	if dir == "" {
		t.Skip("requires Metal")
	}
	m, err := LoadMLXNative(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	n := m.native.(*nativeMLX)
	if _, err = n.SetKVBits(8); err != nil {
		t.Fatal(err)
	}
	ids := m.tokenizer.encode("Mara carried the letter through the rain. When she reached the lighthouse, she looked back at the empty road. The lamp was still burning upstairs, although nobody should have been home.")
	before, err := n.Forward(context.Background(), 0, ids, true)
	if err != nil {
		t.Fatal(err)
	}
	if err = n.packProjections(); err != nil {
		t.Fatal(err)
	}
	views, err := n.Forward(context.Background(), 0, ids, true)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(before, views) {
		t.Fatal("packing changed unfused arithmetic")
	}
	n.fusedProjections = true
	fused, err := n.Forward(context.Background(), 0, ids, true)
	if err != nil {
		t.Fatal(err)
	}
	kl := mlxbench.KL(before, fused)
	if math.IsNaN(kl) || kl > 0.0001 || mlxbench.Argmax(before) != mlxbench.Argmax(fused) {
		t.Fatalf("packed projection mismatch KL %.12f", kl)
	}
	t.Logf("packed vs separate KL %.12f", kl)
}

// TestMLXReservedKVRewind compares reserved and growing Q8 storage through
// writes and rewinds, requiring identical populated values and retained
// reserved capacity.
func TestMLXReservedKVRewind(t *testing.T) {
	if os.Getenv("TINYOAI_MLX_DIR") == "" {
		t.Skip("requires Metal")
	}
	err := mx.Run(func() {
		ctx := mx.New()
		defer ctx.Close()
		var baseline, reserved mlxKV
		defer func() { baseline.Free(); reserved.Free() }()
		for _, span := range [][2]int{{0, 257}, {257, 400}, {19, 31}, {50, 700}, {749, 1}} {
			func() {
				a := &mx.Arena{Context: ctx}
				defer a.Free()
				prefix, length := span[0], span[1]
				values := make([]int, 2*length*128)
				for i := range values {
					values[i] = (i*37+prefix*29)%1001 - 500
				}
				x := a.Reshape(a.Cast(a.Tokens(values), mx.Float16), 1, 2, length, 128)
				p := baseline.update(a, x, prefix, 8)
				q := reserved.update(a, x, prefix, 8, 1024)
				want := a.Contiguous(a.Cast(p.dense(a), mx.Float32))
				got := a.Contiguous(a.Cast(q.dense(a), mx.Float32))
				mx.Eval(want, got)
				if !reflect.DeepEqual(want.Floats(), got.Floats()) {
					t.Errorf("reserved stride mismatch at %v", span)
				}
				if reserved.data.Shape()[2] != 1024 {
					t.Error("reservation not retained")
				}
			}()
		}
	})
	if err != nil {
		t.Fatal(err)
	}
}

// Err returns the current context state and immediately cancels it, placing
// cancellation just after the forward admission check.
func (c cancelAfterAdmission) Err() error {
	err := c.Context.Err()
	c.cancel()
	return err
}

// TestMLXPrefillMidChunkCancellation verifies that an admitted chunk returns
// complete KV, probability rows, and final logits even when cancellation
// arrives during its evaluation.
func TestMLXPrefillMidChunkCancellation(t *testing.T) {
	dir := os.Getenv("TINYOAI_MLX_DIR")
	if dir == "" {
		t.Skip("set TINYOAI_MLX_DIR")
	}
	m, err := LoadMLXNative(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	n := m.native.(*nativeMLX)
	ids := m.tokenizer.encode("The letter arrived on a rainy Tuesday. Mara opened it and read the first line.")
	targets := append(append([]int{}, ids[1:]...), 0)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	z, rows, err := n.ForwardWithLogprobs(cancelAfterAdmission{ctx, cancel}, 0, ids, targets, 12)
	if err != nil || ctx.Err() == nil || len(rows) != len(ids) || len(z) != m.config.VocabSize || n.offset != len(ids) {
		t.Fatalf("incomplete admitted chunk: rows=%d tokens=%d logits=%d offset=%d err=%v", len(rows), len(ids), len(z), n.offset, err)
	}
}
