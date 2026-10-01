package tinyoai

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"
)

type errorStats struct {
	max, squared  float64
	n, top1, rows int
}

// add accumulates maximum and squared logit error plus argmax agreement for
// one equal-length pair of vocabulary rows.
func (s *errorStats) add(got, want []float32) {
	for i, v := range got {
		d := float64(v) - float64(want[i])
		s.max = math.Max(s.max, math.Abs(d))
		s.squared += d * d
		s.n++
	}
	if argmax(got) == argmax(want) {
		s.top1++
	}
	s.rows++
}

// String formats accumulated maximum error, root-mean-square error, and argmax
// agreement for diagnostic logs.
func (s errorStats) String() string {
	return fmt.Sprintf("max=%.9g rms=%.9g top1=%d/%d", s.max, math.Sqrt(s.squared/float64(s.n)), s.top1, s.rows)
}

// openProbe opens one safetensors diagnostic file as a tensor store and
// registers file cleanup with the test.
func openProbe(t *testing.T, path string) *tensorStore {
	t.Helper()
	f, err := openSafeFile(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { f.f.Close() })
	s := &tensorStore{files: []*safeFile{f}, byName: map[string]*safeFile{}}
	for name := range f.tensors {
		s.byName[name] = f
	}
	return s
}

// probeTensor loads a named diagnostic tensor using its recorded dimensions or
// fails the test.
func probeTensor(t *testing.T, s *tensorStore, name string) weightTensor {
	t.Helper()
	f := s.byName[name]
	if f == nil {
		t.Fatalf("missing probe %s", name)
	}
	w, err := s.load(name, f.tensors[name].Shape...)
	if err != nil {
		t.Fatal(err)
	}
	return w
}

// tensorRow decodes one row of a two-dimensional tensor into a fresh float32
// slice.
func tensorRow(w weightTensor, row int) []float32 {
	x := make([]float32, w.shape[1])
	for i := range x {
		x[i] = w.at(row*len(x) + i)
	}
	return x
}

// TestClioNumerics is an opt-in comparison with full-prompt and cached PyTorch
// execution, followed by isolated projection and layer-normalization
// diagnostics. Generate probe files with scripts/clio_numerics.py.
func TestClioNumerics(t *testing.T) {
	dir, probes := os.Getenv("TINYOAI_CLIO_DIR"), os.Getenv("TINYOAI_NUMERICS_DIR")
	if dir == "" || probes == "" {
		t.Skip("set TINYOAI_CLIO_DIR and TINYOAI_NUMERICS_DIR")
	}
	m, err := LoadStableLM(dir)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(probes, "cases.json"))
	if err != nil {
		t.Fatal(err)
	}
	var meta struct {
		Cases []struct {
			IDs []int `json:"ids"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(data, &meta); err != nil {
		t.Fatal(err)
	}
	for i, c := range meta.Cases {
		p := openProbe(t, filepath.Join(probes, fmt.Sprintf("case-%d.safetensors", i)))
		full, cached := probeTensor(t, p, "full"), probeTensor(t, p, "cached")
		state := m.newState(len(c.IDs))
		var sf, sc errorStats
		start := time.Now()
		for pos, id := range c.IDs {
			if err := m.forward(context.Background(), id, pos, state, true); err != nil {
				t.Fatal(err)
			}
			sf.add(state.logits, tensorRow(full, pos))
			sc.add(state.logits, tensorRow(cached, pos))
		}
		t.Logf("case %d (%d tokens) vs full: %s; vs cached: %s; elapsed %s", i, len(c.IDs), sf, sc, time.Since(start))
	}
	projections := map[string]weightTensor{}
	for _, i := range []int{0, 14, 27} {
		p := fmt.Sprintf("model.layers.%d.", i)
		w := m.layers[i]
		for n, x := range map[string]weightTensor{"self_attn.q_proj": w.q, "self_attn.k_proj": w.k, "self_attn.v_proj": w.v, "self_attn.o_proj": w.o, "mlp.gate_proj": w.gate, "mlp.up_proj": w.up, "mlp.down_proj": w.down} {
			projections[p+n] = x
		}
	}
	p := openProbe(t, filepath.Join(probes, "operations.safetensors"))
	names := []string{}
	for name := range projections {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		x, y := probeTensor(t, p, name+".input"), probeTensor(t, p, name+".output")
		var stats errorStats
		for row := 0; row < x.shape[0]; row++ {
			out := make([]float32, y.shape[1])
			projections[name].mul(out, tensorRow(x, row))
			stats.add(out, tensorRow(y, row))
		}
		t.Logf("isolated %s: %s", strings.TrimPrefix(name, "model.layers."), stats)
	}
	for _, i := range []int{0, 14, 27} {
		name := fmt.Sprintf("model.layers.%d.input_layernorm", i)
		x, y := probeTensor(t, p, name+".input"), probeTensor(t, p, name+".output")
		var stats errorStats
		for row := 0; row < x.shape[0]; row++ {
			out := make([]float32, y.shape[1])
			layerNorm(out, tensorRow(x, row), m.layers[i].norm, m.layers[i].bias, m.config.LayerNormEps)
			stats.add(out, tensorRow(y, row))
		}
		t.Logf("isolated layernorm %d: %s", i, stats)
	}
}

// TestClioScalarNumerics isolates layer-normalization rounding choices using
// saved inputs and PyTorch moments; it reports errors without changing
// production arithmetic.
func TestClioScalarNumerics(t *testing.T) {
	dir, probes := os.Getenv("TINYOAI_CLIO_DIR"), os.Getenv("TINYOAI_NUMERICS_DIR")
	if dir == "" || probes == "" {
		t.Skip("set model and probe directories")
	}
	store, err := openTensorStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer store.close()
	p := openProbe(t, filepath.Join(probes, "operations.safetensors"))
	for _, i := range []int{0, 14, 27} {
		n := fmt.Sprintf("model.layers.%d.input_layernorm", i)
		w, err := store.load(n+".weight", 2816)
		if err != nil {
			t.Fatal(err)
		}
		b, err := store.load(n+".bias", 2816)
		if err != nil {
			t.Fatal(err)
		}
		x, y := probeTensor(t, p, n+".input"), probeTensor(t, p, n+".output")
		means, invs := probeTensor(t, p, n+".mean"), probeTensor(t, p, n+".rstd")
		for _, mode := range []string{"baseline", "separate", "fp32-rstd", "torch-moments-fused", "torch-moments-separate"} {
			var stats errorStats
			for row := 0; row < x.shape[0]; row++ {
				input, out := tensorRow(x, row), make([]float32, y.shape[1])
				var mean, variance float64
				for _, v := range input {
					mean += float64(v)
				}
				mean /= float64(len(input))
				for _, v := range input {
					d := float64(v) - mean
					variance += d * d
				}
				variance /= float64(len(input))
				m := float32(mean)
				inv := float32(1 / math.Sqrt(variance+1e-5))
				if mode == "fp32-rstd" {
					inv = 1 / float32(math.Sqrt(float64(float32(variance)+float32(1e-5))))
				}
				if strings.HasPrefix(mode, "torch-moments") {
					m = means.at(row)
					inv = invs.at(row)
				}
				for j, v := range input {
					if mode == "baseline" || mode == "torch-moments-fused" {
						out[j] = (v-m)*inv*w.at(j) + b.at(j)
					} else {
						out[j] = float32(float32((v-m)*inv)*w.at(j)) + b.at(j)
					}
				}
				stats.add(out, tensorRow(y, row))
			}
			t.Logf("layer %d %s: %s", i, mode, stats)
		}
	}
}
