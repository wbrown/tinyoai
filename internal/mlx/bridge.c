//go:build mlx && darwin && arm64 && cgo

#include "bridge.h"
#include <stdio.h>
#include <TargetConditionals.h>

// tinyoai_mlx_simulator identifies the simulator build, whose shared-heap
// restriction is checked before opening the pinned MLX runtime.
int tinyoai_mlx_simulator(void) { return TARGET_OS_SIMULATOR; }

// All entry points are serialized by Run, including this process-wide handler.
static char last_error[4096];
// capture_error copies a native failure into storage read by Go before Run
// releases its process-wide lock. The callback owns neither input pointer.
static void capture_error(const char *msg, void *unused) {
    snprintf(last_error, sizeof(last_error), "%s", msg);
}
// tinyoai_mlx_errors installs the capturing handler under Go's Run lock.
void tinyoai_mlx_errors(void) { mlx_set_error_handler(capture_error, NULL, NULL); }
// tinyoai_mlx_error lends the last captured message until the next native error.
const char *tinyoai_mlx_error(void) { return last_error; }

// swiglu builds silu(gate) * up for compilation. Intermediate FP16 rounding
// differs if these operations run as separate kernels. Every acquired temporary
// is released even if a later C operation fails.
static int swiglu(mlx_vector_array *out, mlx_vector_array inputs, void *payload) {
    mlx_stream stream = *(mlx_stream *)payload;
    mlx_array gate = {0}, up = {0}, sig = {0}, silu = {0}, result = {0};
    int rc = mlx_vector_array_get(&gate, inputs, 0)
        || mlx_vector_array_get(&up, inputs, 1)
        || mlx_sigmoid(&sig, gate, stream)
        || mlx_multiply(&silu, gate, sig, stream)
        || mlx_multiply(&result, silu, up, stream)
        || mlx_vector_array_set_value(out, result);
    mlx_array_free(gate); mlx_array_free(up); mlx_array_free(sig);
    mlx_array_free(silu); mlx_array_free(result);
    return rc;
}
// free_stream releases the closure's retained stream and its heap payload.
static void free_stream(void *payload) {
    mlx_stream_free(*(mlx_stream *)payload);
    free(payload);
}
// tinyoai_mlx_swiglu returns an owned compiled closure that retains its stream.
// The caller releases the closure after all graph construction using it ends.
int tinyoai_mlx_swiglu(mlx_closure *out, mlx_stream stream) {
    mlx_stream *copy = calloc(1, sizeof(*copy));
    if (!copy) return 1;
    mlx_stream_set(copy, stream);
    mlx_closure fn = mlx_closure_new_func_payload(swiglu, copy, free_stream);
    int rc = mlx_compile(out, fn, true);
    mlx_closure_free(fn);
    return rc;
}
