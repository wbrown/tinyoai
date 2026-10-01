//go:build !mlx || !darwin || !arm64 || !cgo

package tinyoai

import (
	"strings"
	"testing"
)

// TestDefaultMLXUnavailable checks that a build without native MLX returns an
// actionable error instead of attempting inference.
func TestDefaultMLXUnavailable(t *testing.T) {
	if _, err := DefaultMLX(); err == nil || !strings.Contains(err.Error(), "-tags mlx") {
		t.Fatalf("missing native backend must return a useful error: %v", err)
	}
}
