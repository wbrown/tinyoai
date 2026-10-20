//go:build mlx && darwin && arm64 && cgo

package mlx

/*
#cgo LDFLAGS: -framework Foundation
#include <stdint.h>
void tinyoai_mlx_process_metrics(uint64_t*, uint64_t*, int*, int*);
void tinyoai_mlx_system_memory(uint64_t*, uint64_t*, uint64_t*, uint64_t*, uint64_t*, int*);
*/
import "C"

// ProcessMetrics uses OS telemetry and does not take the MLX evaluation lock.
// AvailableBytes is the iOS process allowance, not system-free memory (0 on Mac).
type ProcessMetrics struct {
	// FootprintBytes is the process physical footprint from TASK_VM_INFO.
	FootprintBytes uint64 `json:"footprint_bytes"`
	// AvailableBytes is the additional iOS process allowance, or zero on macOS;
	// it is not system-free memory.
	AvailableBytes uint64 `json:"available_bytes"`
	ThermalState   int    `json:"thermal_state"` // 0 nominal, 1 fair, 2 serious, 3 critical
	// Error is the Mach status of the footprint query; zero indicates success.
	Error int `json:"error,omitempty"`
}

// ReadProcessMetrics samples OS footprint, available process allowance, and
// thermal state without acquiring the MLX lock. Error records a failed
// footprint query rather than an inference error.
func ReadProcessMetrics() ProcessMetrics {
	var footprint, available C.uint64_t
	var thermal, failure C.int
	C.tinyoai_mlx_process_metrics(&footprint, &available, &thermal, &failure)
	return ProcessMetrics{uint64(footprint), uint64(available), int(thermal), int(failure)}
}

// SystemMemoryMetrics reports physical VM counters, not a process allocation cap.
type SystemMemoryMetrics struct {
	PhysicalBytes   uint64 `json:"physical_bytes"`
	FreeBytes       uint64 `json:"free_bytes"`
	InactiveBytes   uint64 `json:"inactive_bytes"`
	WiredBytes      uint64 `json:"wired_bytes"`
	CompressedBytes uint64 `json:"compressed_bytes"`
	Error           int    `json:"error,omitempty"`
}

// ReadSystemMemory samples physical RAM and the OS page queues without MLX locks.
func ReadSystemMemory() SystemMemoryMetrics {
	var total, free, inactive, wired, compressed C.uint64_t
	var failure C.int
	C.tinyoai_mlx_system_memory(&total, &free, &inactive, &wired, &compressed, &failure)
	return SystemMemoryMetrics{uint64(total), uint64(free), uint64(inactive), uint64(wired), uint64(compressed), int(failure)}
}
