package mlxbench

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
)

// runMatrixScreen compares stock and experimental loads at eligible, boundary,
// and ineligible row counts. Kernel overrides are restored even on failure.
func runMatrixScreen(ctx context.Context, dir, out string, progress func(string), load func(string) (Backend, error)) error {
	if progress == nil {
		progress = func(string) {}
	}
	if err := os.MkdirAll(out, 0700); err != nil {
		return err
	}
	m, err := load(dir)
	if err != nil {
		return err
	}
	defer m.Close()
	old, present := os.LookupEnv("TINYOAI_QMM_FAST_LOADS")
	defer func() {
		if present {
			_ = os.Setenv("TINYOAI_QMM_FAST_LOADS", old)
		} else {
			_ = os.Unsetenv("TINYOAI_QMM_FAST_LOADS")
		}
	}()
	f, err := os.Create(filepath.Join(out, "matrix-screen.jsonl"))
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	checks := 0
	for _, name := range []string{"attention", "up", "down"} {
		for _, rows := range []int{255, 256, 257, 511, 512, 513} {
			for _, offset := range []int{0, 1} {
				if err := ctx.Err(); err != nil {
					return err
				}
				var baseline []float32
				changed, delta := 0, 0.0
				for _, mode := range []string{"0", "1"} {
					if err := os.Setenv("TINYOAI_QMM_FAST_LOADS", mode); err != nil {
						return err
					}
					values, err := m.Matrix(ctx, name, rows, offset)
					if err != nil {
						return err
					}
					if mode == "0" {
						baseline = values
						continue
					}
					if len(values) != len(baseline) {
						return fmt.Errorf("matrix output shape changed")
					}
					for i, v := range values {
						d := math.Abs(float64(v - baseline[i]))
						if math.IsNaN(d) || math.IsInf(d, 0) {
							return fmt.Errorf("nonfinite guard result")
						}
						if d != 0 {
							changed++
						}
						delta = max(delta, d)
					}
				}
				result := map[string]any{"shape": name, "rows": rows, "input_byte_offset": 2 * offset, "changed": changed, "max_abs": delta}
				if err := enc.Encode(result); err != nil {
					return err
				}
				if err := f.Sync(); err != nil {
					return err
				}
				if changed != 0 {
					return fmt.Errorf("guard mismatch: %v", result)
				}
				progress(fmt.Sprintf("guard %s M=%d byte_offset=%d exact", name, rows, offset*2))
				checks++
			}
		}
	}
	return WriteJSON(filepath.Join(out, "matrix-complete.json"), map[string]any{"complete": true, "checks": checks})
}
