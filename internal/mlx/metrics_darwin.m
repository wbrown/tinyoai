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
