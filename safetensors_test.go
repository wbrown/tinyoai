package tinyoai

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// writeSafeTest writes a caller-supplied safetensors header and payload into a
// temporary file for loader tests.
func writeSafeTest(t *testing.T, header string, data []byte) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "model.safetensors")
	buf := make([]byte, 8)
	binary.LittleEndian.PutUint64(buf, uint64(len(header)))
	buf = append(buf, []byte(header)...)
	buf = append(buf, data...)
	if err := os.WriteFile(p, buf, 0600); err != nil {
		t.Fatal(err)
	}
	return p
}

// TestSafetensorsValidation rejects invalid shapes, byte ranges, overlaps,
// dtypes, and sizes, then checks successful loading and missing-tensor errors.
func TestSafetensorsValidation(t *testing.T) {
	for _, header := range []string{
		`{"x":{"dtype":"BF16","shape":[2],"data_offsets":[0,2]}}`,
		`{"x":{"dtype":"BF16","shape":[1],"data_offsets":[-2,0]}}`,
		`{"x":{"dtype":"BF16","shape":[1],"data_offsets":[2,4]}}`,
		`{"x":{"dtype":"I8","shape":[2],"data_offsets":[0,2]}}`,
		`{"x":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"y":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}`,
		`{"x":{"dtype":"BF16","shape":[9223372036854775807,2],"data_offsets":[0,2]}}`,
	} {
		path := writeSafeTest(t, header, []byte{0, 0})
		f, err := openSafeFile(path)
		if err == nil {
			f.f.Close()
			t.Fatalf("accepted malformed header %s", header)
		}
	}
	path := writeSafeTest(t, `{"x":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}`, []byte{0x80, 0x3f})
	s, err := openTensorStore(filepath.Dir(path))
	if err != nil {
		t.Fatal(err)
	}
	defer s.close()
	w, err := s.load("x", 1)
	if err != nil || w.at(0) != 1 {
		t.Fatalf("load: %v, %v", w, err)
	}
	if _, err := s.load("x", 2); err == nil {
		t.Fatal("accepted wrong shape")
	}
	if _, err := s.load("missing", 1); err == nil {
		t.Fatal("accepted missing tensor")
	}
}

// TestSafetensorsIndexValidation rejects shard traversal, nonexistent tensor
// names, and an empty weight index.
func TestSafetensorsIndexValidation(t *testing.T) {
	path := writeSafeTest(t, `{"x":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}`, []byte{0, 0})
	for _, weights := range []map[string]string{{"x": "../model.safetensors"}, {"missing": "model.safetensors"}, {}} {
		data, _ := json.Marshal(map[string]any{"weight_map": weights})
		if err := os.WriteFile(filepath.Join(filepath.Dir(path), "model.safetensors.index.json"), data, 0600); err != nil {
			t.Fatal(err)
		}
		s, err := openTensorStore(filepath.Dir(path))
		if err == nil {
			s.close()
			t.Fatalf("accepted invalid index %v", weights)
		}
	}
}

// TestHalfFloat checks representative IEEE binary16 values, preserving signed
// zero, subnormals, infinity, and NaN.
func TestHalfFloat(t *testing.T) {
	for _, c := range []struct {
		bits uint16
		want float32
	}{{0, 0}, {0x8000, float32(math.Copysign(0, -1))}, {0x3c00, 1}, {0xc000, -2}, {1, float32(math.Ldexp(1, -24))}, {0x7bff, 65504}, {0x7c00, float32(math.Inf(1))}} {
		if got := halfFloat(c.bits); math.Float32bits(got) != math.Float32bits(c.want) {
			t.Errorf("%04x = %g, want %g", c.bits, got, c.want)
		}
	}
	if !math.IsNaN(float64(halfFloat(0x7e00))) {
		t.Fatal("lost NaN")
	}
}

// BenchmarkBF16MatrixVector measures a model-width BF16 projection with
// reusable activation scratch and reports weight bytes processed.
func BenchmarkBF16MatrixVector(b *testing.B) {
	const dim = 2816
	w := weightTensor{data: make([]byte, dim*dim*2), dtype: "BF16", shape: []int{dim, dim}}
	for i := 0; i < dim*dim; i++ {
		binary.LittleEndian.PutUint16(w.data[i*2:], uint16(math.Float32bits(float32((i%127)-63)/64)>>16))
	}
	x, out := make([]float32, dim), make([]float32, dim)
	scratch := make([]float32, dim)
	for i := range x {
		x[i] = float32((i%31)-15) / 16
	}
	b.SetBytes(int64(len(w.data)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w.mulScratch(out, x, scratch)
	}
}

// TestWeightTensorWideAccumulation uses cancellation of large terms to detect
// loss of a small contribution, including a scalar tail after vector blocks.
func TestWeightTensorWideAccumulation(t *testing.T) {
	// The +1 is lost in a sequential float32 accumulator. Include a non-multiple
	// of four to exercise both the BF16 block loop and its tail.
	values := []float32{1 << 24, 1, -(1 << 24), 2, 3}
	for _, dtype := range []string{"BF16", "F32"} {
		w := weightTensor{dtype: dtype, shape: []int{1, len(values)}, data: make([]byte, len(values)*dtypeWidth(dtype))}
		for i, v := range values {
			if dtype == "BF16" {
				binary.LittleEndian.PutUint16(w.data[i*2:], uint16(math.Float32bits(v)>>16))
			} else {
				binary.LittleEndian.PutUint32(w.data[i*4:], math.Float32bits(v))
			}
		}
		out := make([]float32, 1)
		w.mul(out, []float32{1, 1, 1, 1, 1})
		if out[0] != 6 {
			t.Fatalf("%s reduction = %g, want 6", dtype, out[0])
		}
	}
}
