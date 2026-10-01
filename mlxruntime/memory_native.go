//go:build mlx && darwin && arm64 && cgo

package mlxruntime

import mx "github.com/wbrown/tinyoai/internal/mlx"

// Memory reads allocator statistics after any in-flight evaluation completes.
func Memory() (Stats, error) { return memory(false) }

// ResetPeakMemory releases unused allocator buffers, resets the peak counter,
// and returns the resulting statistics. Live weights and KV remain resident.
func ResetPeakMemory() (Stats, error) { return memory(true) }

// memory reads process-wide MLX counters under the evaluation lock, optionally
// clearing idle buffers and resetting the peak first.
func memory(reset bool) (s Stats, err error) {
	err = mx.Run(func() {
		if reset {
			mx.ResetMemory()
		}
		s.Active, s.Peak = mx.Memory()
		s.Cache = mx.CacheMemory()
	})
	return
}

// ClearCache releases unused allocator buffers; live weights and KV remain valid.
func ClearCache() error { return mx.Run(mx.ClearCache) }
