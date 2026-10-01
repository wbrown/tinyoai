//go:build !mlx || !darwin || !arm64 || !cgo

package tinyoai

import "fmt"

// LoadMLXNative reports that this build lacks native MLX support and
// identifies the required build configuration.
func LoadMLXNative(dir string) (*MLX, error) {
	return nil, fmt.Errorf("native MLX requires the macOS arm64 release, or a cgo build with -tags mlx; see docs/mlx-native.md")
}
