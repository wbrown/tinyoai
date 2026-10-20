//go:build mlx && darwin && arm64 && cgo

#import <Foundation/Foundation.h>
#include <mach/mach.h>
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
#include <os/proc.h>
#endif

// tinyoai_mlx_process_metrics writes OS telemetry into caller-owned outputs.
// Available memory is the iOS process allowance, not system-free RAM; macOS
// returns zero for it. This function touches no MLX state and needs no MLX lock.
void tinyoai_mlx_process_metrics(uint64_t *footprint, uint64_t *available, int *thermal, int *error) {
    @autoreleasepool {
    task_vm_info_data_t info = {0};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    *error = task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count);
    *footprint = info.phys_footprint;
    *available = 0;
#if TARGET_OS_IPHONE
    *available = os_proc_available_memory();
#endif
    *thermal = (int)NSProcessInfo.processInfo.thermalState;
    }
}

// Physical VM counters for allocator/checkpoint diagnostics. Speculative pages
// are already included in free_count. Available is free + inactive, an OS
// reclaimability estimate, not a promise that inactive pages are all discardable.
void tinyoai_mlx_system_memory(uint64_t *physical, uint64_t *free_bytes,
    uint64_t *inactive, uint64_t *wired, uint64_t *compressed, int *error) {
    *physical = NSProcessInfo.processInfo.physicalMemory;
    vm_statistics64_data_t vm = {0};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    mach_port_t host = mach_host_self();
    *error = host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm, &count);
    mach_port_deallocate(mach_task_self(), host);
    *free_bytes = (uint64_t)vm.free_count * vm_page_size;
    *inactive = (uint64_t)vm.inactive_count * vm_page_size;
    *wired = (uint64_t)vm.wire_count * vm_page_size;
    *compressed = (uint64_t)vm.compressor_page_count * vm_page_size;
}
