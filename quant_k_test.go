package tinyoai

import (
	"encoding/binary"
	"math"
	"os"
	"strings"
	"testing"
)

// TestKQuantReference compares decoded Q5_K, Q6_K, and Q5_1 weights bitwise
// with external reference fixtures, then checks partial blocks, projections,
// and GGUF tensor sizes.
func TestKQuantReference(t *testing.T) {
	for _, dtype := range []string{"Q5_K", "Q6_K", "Q5_1"} {
		t.Run(dtype, func(t *testing.T) {
			prefix := "testdata/gguf/" + strings.ToLower(dtype)
			raw, err := os.ReadFile(prefix + ".bin")
			if err != nil {
				t.Fatal(err)
			}
			want, err := os.ReadFile(prefix + "-f32.bin")
			if err != nil {
				t.Fatal(err)
			}
			w := weightTensor{dtype: dtype, data: raw, shape: []int{4, 2048}}
			decoded := make([]float32, len(want)/4)
			w.readValues(decoded, 0)
			for i, v := range decoded {
				if math.Float32bits(v) != binary.LittleEndian.Uint32(want[i*4:]) {
					t.Fatalf("decoded weight %d: %g", i, v)
				}
			}
			for _, span := range [][2]int{{0, 1}, {255, 3}, {257, 511}, {1000, 777}} {
				a := make([]float32, span[1])
				w.readValues(a, span[0])
				for i, v := range a {
					if v != decoded[span[0]+i] {
						t.Fatal("partial block decode")
					}
				}
			}
			for _, n := range []int{1, 7, 128} {
				x := make([]float32, n*2048)
				for i := range x {
					x[i] = float32(math.Sin(float64(i) * .17))
				}
				out := make([]float32, n*4)
				packed := make([]float32, 2048*((n+batchWidth()-1)/batchWidth()*batchWidth()))
				w.mulBatch(out, x, n, packed)
				for token := 0; token < n; token++ {
					for row := 0; row < 4; row++ {
						var expected float64
						for col := 0; col < 2048; col++ {
							expected += float64(x[token*2048+col]) * float64(decoded[row*2048+col])
						}
						if delta := math.Abs(float64(out[token*4+row]) - expected); delta > 2e-4*math.Max(1, math.Abs(expected)) {
							t.Fatalf("n%d token%d row%d: got %g, want %g", n, token, row, out[token*4+row], expected)
						}
					}
				}
			}
			f, err := os.Open(ggufFixture(t, func(_ map[string]any, tensors map[string]weightTensor) { tensors["output.weight"] = w }))
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			_, tensors, err := readGGUF(f)
			if err != nil {
				t.Fatal(err)
			}
			if got := tensors["output.weight"]; got.dtype != dtype || got.size != int64(len(raw)) {
				t.Fatalf("GGUF K tensor: %+v", got)
			}
			if dtype == "Q5_1" {
				w.shape = []int{1, 7552}
				x, out := make([]float32, 7552), make([]float32, 1)
				var expected float64
				for i := range x {
					x[i] = float32(math.Sin(float64(i) * .19))
					expected += float64(x[i]) * float64(decoded[i])
				}
				w.mul(out, x)
				if math.Abs(float64(out[0])-expected) > 2e-4*math.Max(1, math.Abs(expected)) {
					t.Fatal("Q5_1 down-projection tail")
				}
			}
		})
	}
}
