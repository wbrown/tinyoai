// Standalone bridge lifecycle test: no model or Metal workload is executed.
// Including the implementation lets this test exercise the installed handler's
// capture function without exposing a testing API in the production library.
#include "../bridge.c"
#include <assert.h>
#include <stdio.h>

int main(void) {
    char message[16385];
    memset(message, 'x', sizeof(message) - 1);
    message[sizeof(message) - 1] = '\0';
    tinyoai_mlx_errors();
    capture_error(message, NULL);
    assert(strcmp(tinyoai_mlx_error(), message) == 0);
    capture_error("later cleanup failure", NULL);
    assert(strcmp(tinyoai_mlx_error(), message) == 0);
    tinyoai_mlx_errors();
    capture_error("next operation", NULL);
    assert(strcmp(tinyoai_mlx_error(), "next operation") == 0);
    tinyoai_mlx_errors();
    puts("native error capture: pass");
    return 0;
}
