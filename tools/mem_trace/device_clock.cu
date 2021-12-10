#include <stdint.h>
#include <nvbit.h>
#include <cuda.h>

#include <memtrack/util.h>


__device__ uint64_t get_global_time()
{
    uint64_t result;

    // sm30 or higher is required for %globaltimer which is a nanosecond realtime clock.
    // see https://nvidia.github.io/libcudacxx/standard_api/time_library/chrono.html
    // and https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-globaltimer

    // see [^1] for contraint letters and %-escaping
    // https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html
    asm (
        "mov.u64 %0, %%globaltimer;" : "=l"(result)
    );

    return result;
}

__global__ void probe_device_clock(uint64_t *global_time)
{
    *global_time = get_global_time();
}

uint64_t get_time_difference()
{
    int64_t device_time;


    void *buf = cudaMalloc(sizeof(device_time));

    int64_t host_time = util::time_to_ns(util::now());
    probe_device_clock<<<1, 1>>>(buf);

    cudaMemcpy(&device_time, buf, sizeof(device_time), cudaMemcpyDeviceToHost);

    return device_time - host_time;
}