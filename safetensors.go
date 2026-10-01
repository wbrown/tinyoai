package tinyoai

// The safetensors reader retains BF16/F16 weights in their original compact
// representation. Activations use float32. The default matrix-vector kernel
// reduces in float64; an optional experimental SIMD kernel uses short float32
// blocks with float64 totals.

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
)

type tensorInfo struct {
	Dtype   string  `json:"dtype"`
	Shape   []int   `json:"shape"`
	Offsets []int64 `json:"data_offsets"`
}

type safeFile struct {
	f       *os.File
	start   int64
	tensors map[string]tensorInfo
}

// openSafeFile opens a safetensors file and validates its header, shapes, and
// complete nonoverlapping data layout before exposing any tensors. The caller
// owns the returned file descriptor; every failure closes it.
func openSafeFile(path string) (*safeFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	ok := false
	defer func() {
		if !ok {
			f.Close()
		}
	}()
	stat, err := f.Stat()
	if err != nil {
		return nil, err
	}
	var length uint64
	if err := binary.Read(f, binary.LittleEndian, &length); err != nil {
		return nil, err
	}
	if length == 0 || length > 16<<20 || int64(length) > stat.Size()-8 {
		return nil, fmt.Errorf("invalid safetensors header length %d", length)
	}
	header := make([]byte, int(length))
	if _, err := io.ReadFull(f, header); err != nil {
		return nil, err
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(header, &raw); err != nil {
		return nil, err
	}
	s := &safeFile{f: f, start: int64(length) + 8, tensors: make(map[string]tensorInfo)}
	var ranges [][2]int64
	for name, data := range raw {
		if name == "__metadata__" {
			continue
		}
		var info tensorInfo
		if err := json.Unmarshal(data, &info); err != nil {
			return nil, fmt.Errorf("tensor %s: %w", name, err)
		}
		width := dtypeWidth(info.Dtype)
		if width == 0 {
			return nil, fmt.Errorf("tensor %s: unsupported dtype %q", name, info.Dtype)
		}
		if len(info.Offsets) != 2 || info.Offsets[0] < 0 || info.Offsets[1] < info.Offsets[0] || info.Offsets[1] > stat.Size()-s.start {
			return nil, fmt.Errorf("tensor %s: invalid data offsets", name)
		}
		n := int64(width)
		for _, d := range info.Shape {
			if d <= 0 || n > math.MaxInt64/int64(d) {
				return nil, fmt.Errorf("tensor %s: invalid shape", name)
			}
			n *= int64(d)
		}
		if n != info.Offsets[1]-info.Offsets[0] {
			return nil, fmt.Errorf("tensor %s: shape does not match byte count", name)
		}
		ranges = append(ranges, [2]int64{info.Offsets[0], info.Offsets[1]})
		s.tensors[name] = info
	}
	sort.Slice(ranges, func(i, j int) bool { return ranges[i][0] < ranges[j][0] })
	var end int64
	for _, r := range ranges {
		if r[0] != end {
			return nil, fmt.Errorf("safetensors data contains gaps or overlapping tensors")
		}
		end = r[1]
	}
	if end != stat.Size()-s.start {
		return nil, fmt.Errorf("safetensors has trailing data")
	}
	ok = true
	return s, nil
}

// dtypeWidth returns bytes per scalar for supported unquantized storage
// formats, or zero for an unsupported type.
func dtypeWidth(dtype string) int {
	switch dtype {
	case "F32":
		return 4
	case "BF16", "F16":
		return 2
	}
	return 0
}

type tensorStore struct {
	files  []*safeFile
	byName map[string]*safeFile
}

// openTensorStore opens a local single-file or indexed safetensors checkpoint.
// It verifies that the index and shard contents agree exactly and rejects
// duplicate tensors or shard paths outside dir. The caller must close a
// successful store.
func openTensorStore(dir string) (*tensorStore, error) {
	s := &tensorStore{byName: make(map[string]*safeFile)}
	index, err := os.ReadFile(filepath.Join(dir, "model.safetensors.index.json"))
	var weightMap map[string]string
	names := map[string]bool{}
	if err == nil {
		var j struct {
			WeightMap map[string]string `json:"weight_map"`
		}
		if err := json.Unmarshal(index, &j); err != nil {
			return nil, err
		}
		weightMap = j.WeightMap
		if len(weightMap) == 0 {
			return nil, fmt.Errorf("empty safetensors index")
		}
		for _, name := range weightMap {
			if filepath.Base(name) != name || name == "." || name == ".." {
				return nil, fmt.Errorf("invalid shard filename %q", name)
			}
			names[name] = true
		}
	} else if os.IsNotExist(err) {
		names["model.safetensors"] = true
	} else {
		return nil, err
	}
	for name := range names {
		f, err := openSafeFile(filepath.Join(dir, name))
		if err != nil {
			s.close()
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		s.files = append(s.files, f)
		for tensor := range f.tensors {
			if s.byName[tensor] != nil {
				s.close()
				return nil, fmt.Errorf("duplicate tensor %s", tensor)
			}
			if weightMap != nil && weightMap[tensor] != name {
				s.close()
				return nil, fmt.Errorf("index disagrees with shard for %s", tensor)
			}
			s.byName[tensor] = f
		}
	}
	for name := range weightMap {
		if s.byName[name] == nil {
			s.close()
			return nil, fmt.Errorf("index tensor %s missing from shard", name)
		}
	}
	return s, nil
}

// close releases every shard descriptor owned by the store. Already-loaded
// weight buffers remain valid because they contain independent copies.
func (s *tensorStore) close() {
	for _, f := range s.files {
		f.f.Close()
	}
}

type weightTensor struct {
	data  []byte
	dtype string
	shape []int
}

// load checks the requested shape and copies the named tensor into an owned
// byte buffer. The returned tensor preserves its file precision and survives
// closing the store.
func (s *tensorStore) load(name string, shape ...int) (weightTensor, error) {
	f := s.byName[name]
	if f == nil {
		return weightTensor{}, fmt.Errorf("missing tensor %s", name)
	}
	info := f.tensors[name]
	if len(info.Shape) != len(shape) {
		return weightTensor{}, fmt.Errorf("tensor %s: shape %v, want %v", name, info.Shape, shape)
	}
	for i, d := range shape {
		if info.Shape[i] != d {
			return weightTensor{}, fmt.Errorf("tensor %s: shape %v, want %v", name, info.Shape, shape)
		}
	}
	n := info.Offsets[1] - info.Offsets[0]
	if n > int64(int(^uint(0)>>1)) {
		return weightTensor{}, fmt.Errorf("tensor %s is too large for this platform", name)
	}
	w := weightTensor{data: make([]byte, int(n)), dtype: info.Dtype, shape: shape}
	if _, err := f.f.ReadAt(w.data, f.start+info.Offsets[0]); err != nil {
		return weightTensor{}, fmt.Errorf("tensor %s: %w", name, err)
	}
	return w, nil
}

// at decodes one element of a validated tensor as float32. The index must be
// in range; hot loops should use readValues or matrix kernels to amortize
// block decoding.
func (w weightTensor) at(i int) float32 {
	switch w.dtype {
	case "BF16":
		return math.Float32frombits(uint32(binary.LittleEndian.Uint16(w.data[i*2:])) << 16)
	case "F16":
		return halfFloat(binary.LittleEndian.Uint16(w.data[i*2:]))
	case "Q8_0":
		block := w.data[i/32*34:]
		return halfFloat(binary.LittleEndian.Uint16(block)) * float32(int8(block[2+i%32]))
	case "Q5_K", "Q6_K", "Q5_1":
		var value [1]float32
		w.readValues(value[:], i)
		return value[0]
	default:
		return math.Float32frombits(binary.LittleEndian.Uint32(w.data[i*4:]))
	}
}

// halfFloat expands IEEE binary16 bits to float32, preserving signed zero,
// subnormal values, infinities, and NaN payload bits.
func halfFloat(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp, frac := uint32(h>>10)&31, uint32(h&1023)
	if exp == 0 {
		v := float32(math.Ldexp(float64(frac), -24))
		if sign != 0 {
			return -v
		}
		return v
	}
	if exp == 31 {
		return math.Float32frombits(sign | 0x7f800000 | frac<<13)
	}
	return math.Float32frombits(sign | (exp+112)<<23 | frac<<13)
}

// mul writes W*x to out, allocating any activation packing needed by the
// selected kernel. The matrix and vector dimensions must already be validated.
func (w weightTensor) mul(out, x []float32) {
	w.mulScratch(out, x, nil)
}

// mulScratch writes W*x using caller-owned activation scratch and disjoint row
// workers. Scratch may be reused after the call returns, when all workers have
// joined; it stores packed activations rather than expanded model weights.
func (w weightTensor) mulScratch(out, x, scratch []float32) {
	rows, cols := w.shape[0], w.shape[1]
	input := prepareMatvec(x, w.dtype, scratch)
	calc := func(start, end int) { w.mulRows(out, input, start, end) }
	parallelRows(rows, rows*cols, calc)
}
