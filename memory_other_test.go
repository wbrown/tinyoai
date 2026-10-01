//go:build !darwin && !linux

package tinyoai

// peakRSS reports that process peak-RSS telemetry is unavailable on this
// platform.
func peakRSS() (uint64, bool) { return 0, false }
