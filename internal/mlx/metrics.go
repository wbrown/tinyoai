//go:build mlx && darwin && arm64 && cgo

package mlx

/*
#cgo LDFLAGS: -framework Foundation
#include <stdint.h>
void tinyoai_mlx_process_metrics(uint64_t*, uint64_t*, int*, int*);
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
