//go:build !mlx || !darwin || !arm64 || !cgo

package mlxruntime

import "errors"

var errUnavailable = errors.New("native MLX requires -tags mlx on darwin/arm64 with cgo")

// Memory reports that native MLX allocator statistics are unavailable in this
// build.
func Memory() (Stats, error) { return Stats{}, errUnavailable }

// ResetPeakMemory reports that native MLX allocator controls are unavailable
// in this build.
func ResetPeakMemory() (Stats, error) { return Stats{}, errUnavailable }

// ClearCache reports that native MLX allocator controls are unavailable in
// this build.
func ClearCache() error { return errUnavailable }
