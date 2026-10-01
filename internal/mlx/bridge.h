#include <mlx/c/mlx.h>

// All declarations below are called under the Go MLX Run lock, except the
// compile-time simulator query. Error text borrows process-global storage.
void tinyoai_mlx_errors(void);
const char *tinyoai_mlx_error(void);
int tinyoai_mlx_swiglu(mlx_closure *out, mlx_stream stream);
int tinyoai_mlx_simulator(void);
