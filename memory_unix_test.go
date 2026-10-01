//go:build darwin || linux

package tinyoai

import (
	"runtime"
	"syscall"
)

// peakRSS reads the process high-water RSS and converts platform-specific
// units to bytes, reporting whether the query succeeded.
func peakRSS() (uint64, bool) {
	var usage syscall.Rusage
	if syscall.Getrusage(syscall.RUSAGE_SELF, &usage) != nil || usage.Maxrss < 0 {
		return 0, false
	}
	bytes := uint64(usage.Maxrss)
	if runtime.GOOS == "linux" || runtime.GOOS == "android" {
		bytes *= 1024
	}
	return bytes, true
}
