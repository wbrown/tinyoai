// Package mlxruntime exposes process-wide allocator controls for native MLX.
// Calls share the inference stream's lock and are safe during generation.
package mlxruntime

// Stats reports bytes owned by MLX, excluding Go and application allocations.
type Stats struct {
	// Active counts bytes in live MLX allocations.
	Active uint64
	// Peak is the active-allocation high-water mark since the last reset.
	Peak uint64
	// Cache counts unused buffers retained by the MLX allocator.
	Cache uint64
}
