//go:build mlx && darwin && arm64 && cgo

package tinyoai

import (
	"math"
	"os"
	"testing"

	mx "github.com/wbrown/tinyoai/internal/mlx"
)

// TestMLXQuantizedCacheGrowthAndRewind compares populated Q8 cache rows with
// independently quantized incoming rows through growth and rewind, and checks
// packed storage accounting.
func TestMLXQuantizedCacheGrowthAndRewind(t *testing.T) {
	if os.Getenv("TINYOAI_MLX_DIR") == "" {
		t.Skip("requires Metal")
	}
	err := mx.Run(func() {
		ctx := mx.New()
		defer ctx.Close()
		var cache mlxKV
		defer func() { cache.Free() }()
		expected := make([][]float32, 2)
		for round, write := range [][2]int{{0, 257}, {257, 1}, {17, 5}, {22, 600}, {621, 1}} {
			func() {
				a := &mx.Arena{Context: ctx}
				defer a.Free()
				prefix, count := write[0], write[1]
				values := make([]int, 2*count*128)
				for i := range values {
					values[i] = (i*37+round*53)%997 - 498
				}
				x := a.Reshape(a.Cast(a.Tokens(values), mx.Float16), 1, 2, count, 128)
				gotCache := cache.update(a, x, prefix, 8)
				got := a.Contiguous(a.Cast(gotCache.dense(a), mx.Float32))
				// Quantize only the incoming rows as an independent reference.
				q := a.Quantize(x, 64, 8)
				newRows := a.Contiguous(a.Cast(a.Dequantize(q[0], q[1], q[2], 64, 8), mx.Float32))
				mx.Eval(got, newRows)
				incoming := newRows.Floats()
				for i, value := range incoming {
					if math.Abs(float64(value)-float64(values[i])) > 4 {
						t.Fatalf("unexpected quantization error at %d", i)
					}
				}
				for head := range expected {
					expected[head] = append(expected[head][:prefix*128], incoming[head*count*128:(head+1)*count*128]...)
				}
				actual := got.Floats()
				for head, rows := range expected {
					for i, want := range rows {
						if actual[head*len(rows)+i] != want {
							t.Fatalf("round %d head %d row value %d: stale/misaligned KV", round, head, i)
						}
					}
				}
				// 128 data bytes + two groups, each with a half scale and bias.
				wantBytes := uint64(2 * cache.data.Shape()[2] * (128 + 2*4))
				if cache.Bytes() != wantBytes {
					t.Fatalf("bytes %d, want %d", cache.Bytes(), wantBytes)
				}
			}()
		}
	})
	if err != nil {
		t.Fatal(err)
	}
}

// TestMLXNativeKVPrecisionSwitch verifies that real native precision changes
// invalidate reuse while repeated requests at the same precision preserve
// output and cached prefixes.
func TestMLXNativeKVPrecisionSwitch(t *testing.T) {
	dir := os.Getenv("TINYOAI_MLX_DIR")
	if dir == "" {
		t.Skip("requires native Clio")
	}
	m, err := LoadMLXNative(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	prompt := "The letter arrived on a rainy Tuesday. I had just set down my tea when Holmes"
	for _, bits := range []int{16, 8, 16, 8} {
		first, err := m.Generate(prompt, GenerateOptions{MaxTokens: 3, KVBits: bits})
		if err != nil {
			t.Fatal(err)
		}
		if first.CachedPromptTokens != 0 {
			t.Fatal("precision switch reused stale prefix")
		}
		again, err := m.Generate(prompt, GenerateOptions{MaxTokens: 3, KVBits: bits})
		if err != nil {
			t.Fatal(err)
		}
		if again.CachedPromptTokens != first.PromptTokens-1 || again.Text != first.Text {
			t.Fatalf("KV%d prefix reuse mismatch: %+v / %+v", bits, first, again)
		}
	}
}
