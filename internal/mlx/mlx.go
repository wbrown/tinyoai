//go:build mlx && darwin && arm64 && cgo

// Package mlx is the small portion of Apple's C API used by tinyoai. Native
// ownership is explicit: arenas release temporaries; retained arrays need Free.
// Native tensor operations, including Free, must run inside Run. OS telemetry
// from ReadProcessMetrics and the pure-Go Valid handle check do not take it.
package mlx

/*
#cgo LDFLAGS: -lmlxc -lmlx -lc++
#include "bridge.h"
*/
import "C"

import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

var mutex sync.Mutex

type nativeError struct{ message string }

// Error returns the native failure message captured for the current Run call.
func (e nativeError) Error() string { return e.message }

// Run serializes access to MLX's process-wide error handler and command
// encoders, translating native failures into Go errors. Unrelated Go panics
// propagate. Calls must not nest: the lock is not reentrant.
func Run(fn func()) (err error) {
	mutex.Lock()
	defer mutex.Unlock()
	C.tinyoai_mlx_errors()
	defer func() {
		if p := recover(); p != nil {
			if e, ok := p.(nativeError); ok {
				err = e
			} else {
				panic(p)
			}
		}
	}()
	fn()
	return
}

// check turns a failed C status into a nativeError for Run to recover.
func check(code C.int) {
	if code != 0 {
		panic(nativeError{"MLX: " + C.GoString(C.tinyoai_mlx_error())})
	}
}

// Fail aborts the current Run call with a formatted native error.
func Fail(format string, args ...any) { panic(nativeError{fmt.Sprintf(format, args...)}) }

// Array is a native array handle. Copying this value aliases ownership; use
// Retain for an independently releasable reference. Operations require Run.
type Array struct{ raw C.mlx_array }

// Valid reports whether a handle is nonzero. It cannot detect a handle whose
// ownership has already been released.
func (a Array) Valid() bool { return a.raw.ctx != nil }

// Free releases this owned handle under Run. Copies of the Go value are
// aliases, not additional references; release an owned handle only once.
func (a Array) Free() {
	if a.Valid() {
		C.mlx_array_free(a.raw)
	}
}

// Retain creates a separately owned reference to the same array. The caller
// must Free the returned handle under Run.
func (a Array) Retain() Array { var b C.mlx_array; check(C.mlx_array_set(&b, a.raw)); return Array{b} }

// Shape returns a Go-owned copy of the array dimensions without evaluating its
// graph.
func (a Array) Shape() []int {
	n := int(C.mlx_array_ndim(a.raw))
	result := make([]int, n)
	for i, v := range unsafe.Slice(C.mlx_array_shape(a.raw), n) {
		result[i] = int(v)
	}
	return result
}

// Dtype returns the MLX scalar type of the array.
func (a Array) Dtype() int { return int(C.mlx_array_dtype(a.raw)) }

// Bytes returns the logical array size in bytes, not the allocation size of a
// possibly shared backing buffer.
func (a Array) Bytes() uint64 { return uint64(C.mlx_array_nbytes(a.raw)) }

const (
	// Float16 identifies IEEE binary16 array elements.
	Float16 = int(C.MLX_FLOAT16)
	// Float32 identifies IEEE binary32 array elements.
	Float32 = int(C.MLX_FLOAT32)
	// Uint32 identifies packed unsigned 32-bit data or indices.
	Uint32 = int(C.MLX_UINT32)
)

// Context owns explicit CPU/GPU streams and the compiled SwiGLU closure. Use
// New and Close under Run.
type Context struct {
	stream C.mlx_stream
	cpu    C.mlx_stream
	swiglu C.mlx_closure
}

// New opens explicit GPU and CPU streams and a compiled SwiGLU closure. It
// must run under Run; unsupported devices fail through the native error
// boundary. It also sets process-wide allocator limits: a small idle-buffer
// cache and, on macOS, the recommended wired-memory allowance.
func New() *Context {
	// Verified with the iOS 27 simulator: Metal compute works, but MLX's shared
	// heap allocation asserts. Reject it before entering the native allocator.
	if C.tinyoai_mlx_simulator() != 0 {
		Fail("This MLX build requires shared Metal heaps, which the iOS Simulator GPU does not support. Run inference on an iPad, iPhone, or Mac (Designed for iPad).")
	}
	var available C.bool
	check(C.mlx_metal_is_available(&available))
	if !bool(available) {
		Fail("MLX requires an available Apple Metal GPU")
	}
	c := &Context{}
	ok := false
	defer func() {
		if !ok {
			c.Close()
		}
	}()
	dev := C.mlx_device_new_type(C.MLX_GPU, 0)
	defer C.mlx_device_free(dev)
	// Go goroutines migrate between threads. This stream explicitly supports
	// that; Run supplies the synchronization required by MLX.
	c.stream = C.mlx_stream_new_thread_unsafe(dev)
	if c.stream.ctx == nil {
		check(1)
	}
	cpu := C.mlx_device_new_type(C.MLX_CPU, 0)
	c.cpu = C.mlx_stream_new_thread_unsafe(cpu)
	C.mlx_device_free(cpu)
	if c.cpu.ctx == nil {
		check(1)
	}
	var old C.size_t
	cacheLimit := 256 << 20
	if runtime.GOOS == "ios" {
		cacheLimit = 32 << 20
	}
	check(C.mlx_set_cache_limit(&old, C.size_t(cacheLimit)))
	if runtime.GOOS != "ios" {
		info := C.mlx_device_info_new()
		defer C.mlx_device_info_free(info)
		check(C.mlx_device_info_get(&info, dev))
		key := C.CString("max_recommended_working_set_size")
		defer C.free(unsafe.Pointer(key))
		var recommended C.size_t
		check(C.mlx_device_info_get_size(&recommended, info, key))
		check(C.mlx_set_wired_limit(&old, recommended))
	}
	if rc := C.tinyoai_mlx_swiglu(&c.swiglu, c.stream); rc != 0 {
		check(rc)
	}
	ok = true
	return c
}

// Close releases the context streams and compiled closure, then clears idle
// allocator buffers. Call it under Run after releasing dependent arrays; a nil
// context is allowed.
func (c *Context) Close() {
	if c == nil {
		return
	}
	C.mlx_closure_free(c.swiglu)
	C.mlx_stream_free(c.stream)
	C.mlx_stream_free(c.cpu)
	C.mlx_clear_cache()
}

// Arena owns temporary array handles for a lazy graph. Operations return
// arena-owned handles unless documented otherwise; retain arrays that must
// survive Free.
type Arena struct {
	// Context provides the streams for operations; the arena borrows it and does
	// not close it.
	Context *Context
	arrays  []Array
}

// result registers a new handle before checking its status so deferred arena
// cleanup also covers failed operations.
func (s *Arena) result(a C.mlx_array, rc C.int) Array {
	out := Array{a}
	s.arrays = append(s.arrays, out)
	check(rc)
	return out
}

// Free releases every handle owned by the arena and empties it for reuse.
// Independently retained references remain valid.
func (s *Arena) Free() {
	for _, a := range s.arrays {
		a.Free()
	}
	s.arrays = nil
}

// Materialize evaluates a graph boundary while retaining only its live outputs.
// Returned handles belong to the arena; earlier arena handles are invalidated.
// Releasing the other handles before Eval allows MLX to donate their buffers.
func (s *Arena) Materialize(outputs ...Array) []Array {
	kept := make([]Array, len(outputs))
	for i, a := range outputs {
		kept[i] = a.Retain()
	}
	s.Free()
	s.arrays = kept
	Eval(kept...)
	return kept
}

// ints copies Go dimensions or token IDs into the integer representation used
// by the C API.
func ints(xs []int) []C.int {
	out := make([]C.int, len(xs))
	for i, x := range xs {
		out[i] = C.int(x)
	}
	return out
}

// ptr exposes a C integer slice for the duration of a synchronous C call; an
// empty slice yields nil.
func ptr(xs []C.int) *C.int { return unsafe.SliceData(xs) }

// opt represents an explicitly supplied integer in the MLX optional-argument
// ABI.
func opt(x int) C.mlx_optional_int { return C.mlx_optional_int{value: C.int(x), has_value: true} }

// vector packages borrowed array handles into an owned C vector. The caller
// must release the vector after the operation consuming it.
func vector(as []Array) C.mlx_vector_array {
	v := C.mlx_vector_array_new()
	for _, a := range as {
		check(C.mlx_vector_array_append_value(v, a.raw))
	}
	return v
}

// Eval materializes the supplied lazy arrays and waits for their evaluation
// under Run.
func Eval(as ...Array) { v := vector(as); defer C.mlx_vector_array_free(v); check(C.mlx_eval(v)) }

// Tokens copies token IDs into an arena-owned int32 array with shape [1,
// sequence].
func (s *Arena) Tokens(tokens []int) Array {
	data := ints(tokens)
	shape := []C.int{1, C.int(len(tokens))}
	a := C.mlx_array_new_data(unsafe.Pointer(ptr(data)), ptr(shape), 2, C.MLX_INT32)
	if a.ctx == nil {
		check(1)
	}
	return s.result(a, 0)
}

// FloatArray copies Go weights into an MLX-owned float32 array.
func (s *Arena) FloatArray(values []float32, shape ...int) Array {
	size := 1
	for _, dim := range shape {
		if dim <= 0 {
			Fail("invalid float array dimension")
		}
		size *= dim
	}
	if size != len(values) {
		Fail("float array shape does not match data")
	}
	d := ints(shape)
	a := C.mlx_array_new_data(unsafe.Pointer(unsafe.SliceData(values)), ptr(d), C.int(len(d)), C.MLX_FLOAT32)
	if a.ctx == nil {
		check(1)
	}
	return s.result(a, 0)
}

// Multiply constructs an elementwise product with MLX broadcasting rules.
func (s *Arena) Multiply(a, b Array) Array {
	var r C.mlx_array
	rc := C.mlx_multiply(&r, a.raw, b.raw, s.Context.stream)
	return s.result(r, rc)
}

// RMSNorm normalizes the final axis by its root mean square with stabilizer
// eps, then applies weight.
func (s *Arena) RMSNorm(a, weight Array, eps float64) Array {
	var r C.mlx_array
	rc := C.mlx_fast_rms_norm(&r, a.raw, weight.raw, C.float(eps), s.Context.stream)
	return s.result(r, rc)
}

// Load reads named safetensors arrays on the CPU stream. Returned handles
// belong to the arena; retain weights that must outlive it.
func (s *Arena) Load(path string) map[string]Array {
	name := C.CString(path)
	defer C.free(unsafe.Pointer(name))
	weights := C.mlx_map_string_to_array_new()
	defer C.mlx_map_string_to_array_free(weights)
	meta := C.mlx_map_string_to_string_new()
	defer C.mlx_map_string_to_string_free(meta)
	check(C.mlx_load_safetensors(&weights, &meta, name, s.Context.cpu))
	iter := C.mlx_map_string_to_array_iterator_new(weights)
	defer C.mlx_map_string_to_array_iterator_free(iter)
	result := make(map[string]Array)
	for {
		var key *C.char
		var a C.mlx_array
		rc := C.mlx_map_string_to_array_iterator_next(&key, &a, iter)
		if rc == 2 {
			break
		}
		check(rc)
		result[C.GoString(key)] = s.result(a, 0)
	}
	return result
}

// Reshape gives an array new dimensions, allowing one inferred dimension
// specified as -1.
func (s *Arena) Reshape(a Array, shape ...int) Array {
	d := ints(shape)
	var r C.mlx_array
	rc := C.mlx_reshape(&r, a.raw, ptr(d), C.size_t(len(d)), s.Context.stream)
	return s.result(r, rc)
}

// Transpose permutes dimensions into the supplied axis order.
func (s *Arena) Transpose(a Array, axes ...int) Array {
	d := ints(axes)
	var r C.mlx_array
	rc := C.mlx_transpose_axes(&r, a.raw, ptr(d), C.size_t(len(d)), s.Context.stream)
	return s.result(r, rc)
}

// Take gathers rows along axis zero, as used for token embeddings.
func (s *Arena) Take(a, indices Array) Array {
	var r C.mlx_array
	rc := C.mlx_take_axis(&r, a.raw, indices.raw, 0, s.Context.stream)
	return s.result(r, rc)
}

// Add constructs an elementwise sum with MLX broadcasting rules.
func (s *Arena) Add(a, b Array) Array {
	var r C.mlx_array
	rc := C.mlx_add(&r, a.raw, b.raw, s.Context.stream)
	return s.result(r, rc)
}

// Cast converts array elements to the requested MLX scalar type.
func (s *Arena) Cast(a Array, dtype int) Array {
	var r C.mlx_array
	rc := C.mlx_astype(&r, a.raw, C.mlx_dtype(dtype), s.Context.stream)
	return s.result(r, rc)
}

// Norm applies layer normalization over the final axis, followed by scale w
// and bias b.
func (s *Arena) Norm(a, w, b Array, eps float64) Array {
	var r C.mlx_array
	rc := C.mlx_fast_layer_norm(&r, a.raw, w.raw, b.raw, C.float(eps), s.Context.stream)
	return s.result(r, rc)
}

// Rope applies split-half rotary embeddings to the first dims features,
// starting at absolute position offset with the supplied frequency base.
func (s *Arena) Rope(a Array, dims, offset int, base float64) Array {
	var r C.mlx_array
	rc := C.mlx_fast_rope(&r, a.raw, C.int(dims), false, C.mlx_optional_float{value: C.float(base), has_value: true}, 1, C.int(offset), C.mlx_array{}, s.Context.stream)
	return s.result(r, rc)
}

// Attention computes scaled dot-product attention using MLX's fused
// implementation. With causal set, query positions align with the end of the
// key sequence so a cached prefix remains visible.
func (s *Arena) Attention(q, k, v Array, scale float64, causal bool) Array {
	mode := ""
	if causal {
		mode = "causal"
	}
	str := C.CString(mode)
	defer C.free(unsafe.Pointer(str))
	var r C.mlx_array
	rc := C.mlx_fast_scaled_dot_product_attention(&r, q.raw, k.raw, v.raw, C.float(scale), str, C.mlx_array{}, C.mlx_array{}, false, s.Context.stream)
	return s.result(r, rc)
}

// Linear projects activations through transposed affine-quantized weight rows
// using the specified group size and bit width.
func (s *Arena) Linear(a, w, scales, biases Array, group, bits int) Array {
	return s.QuantizedMatmul(a, w, scales, biases, group, bits, true)
}

// QuantizedMatmul multiplies activations by affine-quantized weights,
// optionally transposing the final two weight axes.
func (s *Arena) QuantizedMatmul(a, w, scales, biases Array, group, bits int, transpose bool) Array {
	mode := C.CString("affine")
	defer C.free(unsafe.Pointer(mode))
	var r C.mlx_array
	rc := C.mlx_quantized_matmul(&r, a.raw, w.raw, scales.raw, biases.raw, C.bool(transpose), opt(group), opt(bits), mode, s.Context.stream)
	return s.result(r, rc)
}

// Scale multiplies each element by a scalar converted to the array's dtype
// before multiplication.
func (s *Arena) Scale(a Array, value float32) Array {
	scalar := s.result(C.mlx_array_new_float32(C.float(value)), 0)
	scalar = s.Cast(scalar, a.Dtype())
	var r C.mlx_array
	rc := C.mlx_multiply(&r, a.raw, scalar.raw, s.Context.stream)
	return s.result(r, rc)
}

// Softmax normalizes the final axis with MLX's precise accumulation mode.
func (s *Arena) Softmax(a Array) Array {
	var r C.mlx_array
	rc := C.mlx_softmax_axis(&r, a.raw, -1, true, s.Context.stream)
	return s.result(r, rc)
}

// Matmul multiplies the final two axes using MLX batch broadcasting.
func (s *Arena) Matmul(a, b Array) Array {
	var r C.mlx_array
	rc := C.mlx_matmul(&r, a.raw, b.raw, s.Context.stream)
	return s.result(r, rc)
}

// Dequantize reconstructs affine-quantized values from packed data, scales,
// and biases.
func (s *Arena) Dequantize(w, scales, biases Array, group, bits int) Array {
	mode := C.CString("affine")
	defer C.free(unsafe.Pointer(mode))
	var r C.mlx_array
	rc := C.mlx_dequantize(&r, w.raw, scales.raw, biases.raw, opt(group), opt(bits), mode, C.mlx_array{}, C.mlx_optional_dtype{}, s.Context.stream)
	return s.result(r, rc)
}

// Quantize encodes values in affine groups, returning arena-owned packed data,
// scales, and biases in that order.
func (s *Arena) Quantize(x Array, group, bits int) [3]Array {
	mode := C.CString("affine")
	defer C.free(unsafe.Pointer(mode))
	values := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(values)
	check(C.mlx_quantize(&values, x.raw, opt(group), opt(bits), mode, C.mlx_array{}, s.Context.stream))
	var result [3]Array
	for i := range result {
		var r C.mlx_array
		rc := C.mlx_vector_array_get(&r, values, C.size_t(i))
		result[i] = s.result(r, rc)
	}
	return result
}

// SwiGLU computes silu(gate) * up through a compiled closure. Fusing the
// expression preserves the rounding behavior used by the MLX reference.
func (s *Arena) SwiGLU(gate, up Array) Array {
	inputs := vector([]Array{gate, up})
	defer C.mlx_vector_array_free(inputs)
	outputs := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(outputs)
	check(C.mlx_closure_apply(&outputs, s.Context.swiglu, inputs))
	var r C.mlx_array
	rc := C.mlx_vector_array_get(&r, outputs, 0)
	return s.result(r, rc)
}

// Zeros constructs an array of the requested shape and dtype initialized to
// zero.
func (s *Arena) Zeros(dtype int, shape ...int) Array {
	d := ints(shape)
	var r C.mlx_array
	rc := C.mlx_zeros(&r, ptr(d), C.size_t(len(d)), C.mlx_dtype(dtype), s.Context.stream)
	return s.result(r, rc)
}

// Concat joins two arrays along axis, preserving the other dimensions.
func (s *Arena) Concat(a, b Array, axis int) Array {
	v := vector([]Array{a, b})
	defer C.mlx_vector_array_free(v)
	var r C.mlx_array
	rc := C.mlx_concatenate_axis(&r, v, C.int(axis), s.Context.stream)
	return s.result(r, rc)
}

// Slice selects the half-open interval [start, end) along one axis, leaving
// all other axes intact.
func (s *Arena) Slice(a Array, axis, start, end int) Array {
	stop := ints(a.Shape())
	first := make([]C.int, len(stop))
	stride := make([]C.int, len(stop))
	for i := range stride {
		stride[i] = 1
	}
	first[axis] = C.int(start)
	stop[axis] = C.int(end)
	var r C.mlx_array
	n := C.size_t(len(stop))
	rc := C.mlx_slice(&r, a.raw, ptr(first), n, ptr(stop), n, ptr(stride), n, s.Context.stream)
	return s.result(r, rc)
}

// Update returns an array with b replacing a slice of a beginning at offset
// along axis. Buffer reuse is decided by MLX; callers must use the returned
// handle.
func (s *Arena) Update(a, b Array, axis, offset int) Array {
	stop := ints(a.Shape())
	first := make([]C.int, len(stop))
	stride := make([]C.int, len(stop))
	for i := range stride {
		stride[i] = 1
	}
	first[axis] = C.int(offset)
	stop[axis] = C.int(offset + b.Shape()[axis])
	var r C.mlx_array
	n := C.size_t(len(stop))
	rc := C.mlx_slice_update(&r, a.raw, b.raw, ptr(first), n, ptr(stop), n, ptr(stride), n, s.Context.stream)
	return s.result(r, rc)
}

// Contiguous requests a contiguous array suitable for reading through a native
// data pointer.
func (s *Arena) Contiguous(a Array) Array {
	var r C.mlx_array
	rc := C.mlx_contiguous(&r, a.raw, false, s.Context.stream)
	return s.result(r, rc)
}

// Floats copies an evaluated, contiguous float32 array into Go memory. The
// copy survives native-handle release; callers must cast, make contiguous, and
// evaluate first.
func (a Array) Floats() []float32 {
	p := C.mlx_array_data_float32(a.raw)
	if p == nil {
		Fail("MLX logits were not evaluated")
	}
	data := unsafe.Slice((*float32)(unsafe.Pointer(p)), int(C.mlx_array_size(a.raw)))
	return append([]float32(nil), data...)
}

// LogSumExp reduces the final axis stably while retaining a singleton
// dimension for broadcasting.
func (s *Arena) LogSumExp(a Array) Array {
	var r C.mlx_array
	rc := C.mlx_logsumexp_axis(&r, a.raw, -1, true, s.Context.stream)
	return s.result(r, rc)
}

// Subtract constructs an elementwise difference with MLX broadcasting rules.
func (s *Arena) Subtract(a, b Array) Array {
	var r C.mlx_array
	rc := C.mlx_subtract(&r, a.raw, b.raw, s.Context.stream)
	return s.result(r, rc)
}

// TopIndices selects indices of the count largest values along the final axis.
// Their order is unspecified; count must be positive and no larger than that
// axis.
func (s *Arena) TopIndices(a Array, count int) Array {
	var r C.mlx_array
	shape := a.Shape()
	vocab := shape[len(shape)-1]
	rc := C.mlx_argpartition_axis(&r, a.raw, C.int(vocab-count), -1, s.Context.stream)
	part := s.result(r, rc)
	return s.Slice(part, len(shape)-1, vocab-count, vocab)
}

// BlockTopIndices performs exact hierarchical selection using existing MLX
// operators. Metal's argpartition currently sorts; short block sorts followed
// by a much smaller candidate sort avoid globally sorting the vocabulary.
func (s *Arena) BlockTopIndices(a Array, count, block int) Array {
	shape := a.Shape()
	vocab := shape[len(shape)-1]
	if block <= count || block >= vocab || vocab%block != 0 {
		return s.TopIndices(a, count)
	}
	blocks := vocab / block
	grouped := s.Reshape(a, -1, blocks, block)
	local := s.TopIndices(grouped, count)
	offsets := make([]int, blocks)
	for i := range offsets {
		offsets[i] = i * block
	}
	ids := s.Add(local, s.Reshape(s.Cast(s.Tokens(offsets), Uint32), 1, blocks, 1))
	values := s.Reshape(s.TakeAlong(grouped, local), -1, blocks*count)
	ids = s.Reshape(ids, -1, blocks*count)
	chosen := s.TakeAlong(ids, s.TopIndices(values, count))
	shape[len(shape)-1] = count
	return s.Reshape(chosen, shape...)
}

// TakeAlong gathers values at per-row indices along the final axis.
func (s *Arena) TakeAlong(a, indices Array) Array {
	var r C.mlx_array
	rc := C.mlx_take_along_axis(&r, a.raw, indices.raw, -1, s.Context.stream)
	return s.result(r, rc)
}

// DeviceName identifies the Metal device used by native inference diagnostics.
// Call under Run, like other MLX operations.
func DeviceName() string {
	dev := C.mlx_device_new_type(C.MLX_GPU, 0)
	defer C.mlx_device_free(dev)
	info := C.mlx_device_info_new()
	defer C.mlx_device_info_free(info)
	check(C.mlx_device_info_get(&info, dev))
	key := C.CString("device_name")
	defer C.free(unsafe.Pointer(key))
	var name *C.char
	check(C.mlx_device_info_get_string(&name, info, key))
	return C.GoString(name)
}

// Memory returns active and peak MLX allocation counts in bytes under Run; it
// excludes Go allocations.
func Memory() (active, peak uint64) {
	var a, p C.size_t
	check(C.mlx_get_active_memory(&a))
	check(C.mlx_get_peak_memory(&p))
	return uint64(a), uint64(p)
}

// ResetMemory releases idle allocations and resets the measured peak. Retained
// model weights and KV arrays remain live. Call only inside Run.
func ResetMemory() {
	check(C.mlx_clear_cache())
	check(C.mlx_reset_peak_memory())
}

// CacheMemory returns bytes held in unused MLX allocator buffers under Run.
func CacheMemory() uint64 {
	var bytes C.size_t
	check(C.mlx_get_cache_memory(&bytes))
	return uint64(bytes)
}

// ClearCache releases unused buffers without resetting the measurement peak.
func ClearCache() { check(C.mlx_clear_cache()) }

// StartCapture begins an opt-in Metal diagnostic capture at path.
// MTL_CAPTURE_ENABLED=1 must be set before process launch; captured runs
// should not be used for timing claims.
func StartCapture(path string) {
	p := C.CString(path)
	defer C.free(unsafe.Pointer(p))
	check(C.mlx_metal_start_capture(p))
}

// StopCapture ends the active Metal diagnostic capture under Run.
func StopCapture() { check(C.mlx_metal_stop_capture()) }

// SetCacheLimit changes the cap on unused allocations, not live tensors.
func SetCacheLimit(bytes uint64) {
	var old C.size_t
	check(C.mlx_set_cache_limit(&old, C.size_t(bytes)))
}
