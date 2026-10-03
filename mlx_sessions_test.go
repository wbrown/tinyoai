//go:build mlx && darwin && arm64 && cgo

package tinyoai

import (
	"context"
	"fmt"
	"os"
	"reflect"
	"strings"
	"testing"

	mx "github.com/wbrown/tinyoai/internal/mlx"
)

// TestMLXForkIsolation checks copy-on-write KV, both ownership orders, rewind,
// and cache growth against the original full-vocabulary next-token logits.
func TestMLXForkIsolation(t *testing.T) {
	dir := os.Getenv("TINYOAI_MLX_DIR")
	if dir == "" {
		t.Skip("set TINYOAI_MLX_DIR")
	}
	for _, bits := range []int{8, 16} {
		t.Run(fmt.Sprintf("KV%d", bits), func(t *testing.T) {
			m, err := LoadMLXNative(dir)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			prompt := "[ Title: The lighthouse ]\nMara found a sealed letter beside the lamp."
			opts := GenerateOptions{MaxTokens: 2, Temperature: 0, KVBits: bits, Logprobs: 12, PromptLogprobs: true}
			if _, err = m.Prefill(prompt, opts); err != nil {
				t.Fatal(err)
			}
			ids := append([]int(nil), m.tokens...)
			n := m.native.(*nativeMLX)
			reference, err := n.Forward(context.Background(), len(ids)-1, ids[len(ids)-1:], true)
			if err != nil {
				t.Fatal(err)
			}
			var before, after uint64
			if err = mx.Run(func() { before, _ = mx.Memory() }); err != nil {
				t.Fatal(err)
			}
			fork, err := m.Fork(context.Background())
			if err != nil {
				t.Fatal(err)
			}
			defer fork.Close()
			child := fork.(*MLX)
			if err = mx.Run(func() { after, _ = mx.Memory() }); err != nil {
				t.Fatal(err)
			}
			if after != before {
				t.Fatalf("fork allocated tensor storage: %d -> %d", before, after)
			}
			// Force cache growth beyond the original 256-position capacity.
			chatPrompt := prompt + "\n***\n[ Style: chat ]\n" + strings.Repeat("Author: Why did she wait?\nClio: She was uncertain.\n", 24) + "Clio:"
			chatOpts := opts
			chatOpts.Logprobs, chatOpts.PromptLogprobs = 0, false
			result, err := child.Generate(chatPrompt, chatOpts)
			if err != nil {
				t.Fatal(err)
			}
			if result.CachedPromptTokens < len(ids)-1 {
				t.Fatalf("fork missed prefix: %+v", result)
			}
			if !reflect.DeepEqual(m.tokens, ids) {
				t.Fatal("child changed parent token history")
			}
			actual, err := n.Forward(context.Background(), len(ids)-1, ids[len(ids)-1:], true)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(reference, actual) {
				t.Fatalf("child changed parent logits: KL %.12f", distributionKL(reference, actual))
			}
			// Parent edits must also preserve the child's retained state.
			childIDs := append([]int(nil), child.tokens...)
			cn := child.native.(*nativeMLX)
			childRef, err := cn.Forward(context.Background(), len(childIDs)-1, childIDs[len(childIDs)-1:], true)
			if err != nil {
				t.Fatal(err)
			}
			if _, err = m.Generate("A different story begins.", opts); err != nil {
				t.Fatal(err)
			}
			childGot, err := cn.Forward(context.Background(), len(childIDs)-1, childIDs[len(childIDs)-1:], true)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(childRef, childGot) {
				t.Fatal("parent edit changed child logits")
			}
			if err = m.Close(); err != nil {
				t.Fatal(err)
			}
			if _, err = child.Generate(chatPrompt, chatOpts); err != nil {
				t.Fatal("child after parent close:", err)
			}
			if err = child.Close(); err != nil {
				t.Fatal(err)
			}
			t.Logf("KV%d: fork added %d active bytes; parent and child logits unchanged; %d prompt tokens reused", bits, after-before, result.CachedPromptTokens)
		})
	}
}
