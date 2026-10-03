//go:build mlx && darwin && arm64 && cgo

package tinyoai

import (
	"context"
	"math"
	"os"
	"reflect"
	"testing"

	mx "github.com/wbrown/tinyoai/internal/mlx"
)

// TestMLXLoRALifecycle uses real Metal with tiny checked-in base weights. It
// verifies atomic failure, cache invalidation, shared tensor ownership, fork
// isolation, packed-projection fallback, scale zero, and unloading.
func TestMLXLoRALifecycle(t *testing.T) {
	if os.Getenv("TINYOAI_TEST_METAL") != "1" {
		t.Skip("set TINYOAI_TEST_METAL=1")
	}
	ctx := context.Background()
	m, err := LoadMLXNative(tinyStableMLXFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	n := m.native.(*nativeMLX)
	prompt := "Once upon a time"
	ids := m.tokenizer.encode(prompt)
	forward := func(model *MLX) []float32 {
		t.Helper()
		row, e := model.native.Forward(ctx, 0, ids, true)
		if e != nil {
			t.Fatal(e)
		}
		return row
	}
	base := forward(m)
	if err = n.packProjections(); err != nil {
		t.Fatal(err)
	}
	n.fusedProjections = true
	packedBase := forward(m)
	adapter := loRAFixture(t, nil)
	info, err := m.LoadLoRA(ctx, adapter, 1)
	if err != nil || info.Projections != 4 {
		t.Fatal(info, err)
	}
	adapted := forward(m)
	if reflect.DeepEqual(base, adapted) {
		t.Fatal("adapter had no effect")
	}
	// Active adapters must reach the original per-projection paths even when
	// the base was packed and the experiment requested fused projections.
	n.fusedProjections = false
	if !reflect.DeepEqual(adapted, forward(m)) {
		t.Fatal("fused path lost correction")
	}
	if _, err = m.Prefill(prompt, GenerateOptions{MaxTokens: 2, Logprobs: 3, PromptLogprobs: true}); err != nil {
		t.Fatal(err)
	}
	beforeTokens, beforeRows, beforeOffset := len(m.tokens), len(m.probRows), n.offset
	if beforeTokens == 0 || beforeRows == 0 {
		t.Fatal("prefill did not populate state")
	}
	if _, err = m.LoadLoRA(ctx, t.TempDir(), 1); err == nil {
		t.Fatal("accepted missing adapter")
	}
	if len(m.tokens) != beforeTokens || len(m.probRows) != beforeRows || n.offset != beforeOffset || n.loRAInfo() != info {
		t.Fatal("failed load mutated session")
	}
	if _, err = m.SetLoRAScale(ctx, math.Inf(1)); err == nil || n.loRAInfo() != info {
		t.Fatal("invalid scale changed adapter")
	}
	var before, after uint64
	if err = mx.Run(func() { before, _ = mx.Memory() }); err != nil {
		t.Fatal(err)
	}
	fork, err := m.Fork(ctx)
	if err != nil {
		t.Fatal(err)
	}
	child := fork.(*MLX)
	defer child.Close()
	if err = mx.Run(func() { after, _ = mx.Memory() }); err != nil {
		t.Fatal(err)
	}
	if before != after {
		t.Fatalf("fork copied adapter or base tensor storage: %d bytes", after-before)
	}
	if !reflect.DeepEqual(adapted, forward(child)) {
		t.Fatal("fork changed adapter logits")
	}
	if _, err = m.SetLoRAScale(ctx, 0); err != nil {
		t.Fatal(err)
	}
	if len(m.tokens) != 0 || len(m.probRows) != 0 || m.probCount != 0 || n.offset != 0 || n.layers[0].keys.data.Valid() {
		t.Fatal("scale retained stale cache")
	}
	if !reflect.DeepEqual(base, forward(m)) {
		t.Fatal("zero strength differs from base")
	}
	n.fusedProjections = true
	if !reflect.DeepEqual(packedBase, forward(m)) {
		t.Fatal("zero strength changed packed base")
	}
	if !reflect.DeepEqual(adapted, forward(child)) {
		t.Fatal("parent scale changed fork")
	}
	if _, err = m.LoadLoRA(ctx, adapter, -1); err != nil {
		t.Fatal(err)
	}
	if reflect.DeepEqual(adapted, forward(m)) {
		t.Fatal("negative strength ignored")
	}
	if _, err = m.LoadLoRA(ctx, "", 0); err != nil {
		t.Fatal(err)
	}
	if n.lora != nil || !reflect.DeepEqual(packedBase, forward(m)) {
		t.Fatal("unload failed to restore base")
	}
	if _, err = m.SetLoRAScale(ctx, 1); err == nil {
		t.Fatal("scaled missing adapter")
	}
	if err = m.Close(); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(adapted, forward(child)) {
		t.Fatal("parent close freed child adapter")
	}
	if _, err = child.Generate(prompt, GenerateOptions{MaxTokens: 2}); err != nil {
		t.Fatal(err)
	}
	if err = child.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err = child.LoadLoRA(ctx, adapter, 1); err == nil {
		t.Fatal("loaded closed session")
	}
	var active uint64
	if err = mx.Run(func() { active, _ = mx.Memory() }); err != nil {
		t.Fatal(err)
	}
	if active != 0 {
		t.Fatalf("leaked %d active tensor bytes", active)
	}
}
