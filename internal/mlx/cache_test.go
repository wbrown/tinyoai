//go:build mlx && darwin && arm64 && cgo

package mlx

import (
	"os"
	"runtime"
	"testing"
	"time"
)

// A new stream context must not silently replace a process allocator policy.
func TestContextPreservesCachePolicy(t *testing.T) {
	if os.Getenv("TINYOAI_TEST_METAL") != "1" || runtime.GOOS == "ios" {
		t.Skip("requires macOS Metal")
	}
	if err := Run(func() {
		original := SetCacheLimit(0)
		defer SetCacheLimit(original)
		for _, policy := range []uint64{0, original} {
			SetCacheLimit(policy)
			ctx := New()
			got := SetCacheLimit(policy)
			ctx.Close()
			if got != policy {
				t.Errorf("context changed cache policy: %d -> %d", policy, got)
			}
		}
	}); err != nil {
		t.Fatal(err)
	}
}

// The allocator must enforce a changed cache allowance while no model work
// is running: checkpoint serialization/hash work does not allocate tensors.
func TestIdleCacheHeadroomWorker(t *testing.T) {
	if os.Getenv("TINYOAI_TEST_HEADROOM") != "1" {
		t.Skip("requires headroom runtime")
	}
	if err := Run(func() {
		ctx := New()
		defer ctx.Close()
		old := SetCacheLimit(128 << 20)
		defer SetCacheLimit(old)
		a := &Arena{Context: ctx}
		x := a.Zeros(Float32, 1<<22)
		Eval(x)
		ctx.Synchronize()
		a.Free()
		before := CacheMemory()
		if before == 0 {
			t.Fatal("fixture did not cache buffers")
		}
		time.Sleep(250 * time.Millisecond)
		if CacheMemory() != before {
			t.Fatal("idle cache released despite sufficient headroom")
		}
		SetCacheLimit(0)
		deadline := time.Now().Add(3 * time.Second)
		for CacheMemory() != 0 && time.Now().Before(deadline) {
			time.Sleep(20 * time.Millisecond)
		}
		if got := CacheMemory(); got != 0 {
			t.Fatalf("idle buffers retained without allocator activity: %d", got)
		}
	}); err != nil {
		t.Fatal(err)
	}
}

// The native allocator refreshes the budget from actual VM counters while
// retaining caller cache choices. This test requires the patched runtime.
func TestMachineMemoryBudget(t *testing.T) {
	if os.Getenv("TINYOAI_TEST_HEADROOM") != "1" {
		t.Skip("requires headroom runtime")
	}
	if err := Run(func() {
		ctx := New()
		defer ctx.Close()
		a := &Arena{Context: ctx}
		defer a.Free()
		x := a.Zeros(Float32, 1<<24)
		Eval(x)
		sys := ReadSystemMemory()
		if sys.Error != 0 || sys.PhysicalBytes == 0 {
			t.Fatal("VM telemetry failed", sys)
		}
		limit := MemoryLimit()
		reserve := sys.PhysicalBytes / 10
		if sys.PhysicalBytes%10 != 0 {
			reserve++
		}
		if limit > sys.PhysicalBytes-reserve {
			t.Fatalf("budget %d does not leave 10%% of %d", limit, sys.PhysicalBytes)
		}
		t.Logf("physical=%d free=%d inactive=%d native_budget=%d", sys.PhysicalBytes, sys.FreeBytes, sys.InactiveBytes, limit)
	}); err != nil {
		t.Fatal(err)
	}
}
