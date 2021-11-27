#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <mutex>

#include <nvbit.h>


namespace meminf
{

// exported utility functions
bool is_malloc_call(nvbit_api_cuda_t cbid);
bool is_free_call(nvbit_api_cuda_t cbid);

/*
    API_CUDA_cuMemAlloc ||
    API_CUDA_cuMemAllocPitch ||
    API_CUDA_cuMemAlloc_v2 ||
    API_CUDA_cuMemAllocPitch_v2 ||
    API_CUDA_cuMemAllocManaged ||
    API_CUDA_cuMemAllocAsync ||
    API_CUDA_cuMemAllocAsync_ptsz ||
    API_CUDA_cuMemAllocFromPoolAsync ||
    API_CUDA_cuMemAllocFromPoolAsync_ptsz
 */
struct device_buffer
{
    size_t buf_size;
    void *location;

    // can be given by user
    std::string name_tag;

    union
    {
        cuMemAlloc_params cuMemAlloc;
        cuMemAllocPitch_params cuMemAllocPitch;
        cuMemAlloc_v2_params cuMemAlloc_v2;
        cuMemAllocPitch_v2_params cuMemAllocPitch_v2;
        cuMemAllocManaged_params cuMemAllocManaged;
        cuMemAllocAsync_params cuMemAllocAsync;
        cuMemAllocAsync_ptsz_params cuMemAllocAsync_ptsz;
        cuMemAllocFromPoolAsync_params cuMemAllocFromPoolAsync;
        cuMemAllocFromPoolAsync_ptsz_params cuMemAllocFromPoolAsync_ptsz;
    } allocation_parameters;

    enum allocation_type {
        cuMemAlloc,
        cuMemAllocPitch,
        cuMemAlloc_v2,
        cuMemAllocPitch_v2,
        cuMemAllocManaged,
        cuMemAllocAsync,
        cuMemAllocAsync_ptsz,
        cuMemAllocFromPoolAsync,
        cuMemAllocFromPoolAsync_ptsz
    } allocation_type;

    device_buffer(nvbit_api_cuda_t cbid, void *params) noexcept(false);
};

class device_buffer_tracker
{
private:
    // cuda API might be called by multiple threads
    mutable std::mutex mut;
    std::unordered_map<void *, device_buffer> global_device_buffers;
public:
    // on malloc()
    void track(nvbit_api_cuda_t cbid, void *params) noexcept(false);
    // on free()
    void untrack(void *location);
    // user decides to track a previously allocated buffer
    void user_track_buffer(void *location, const std::string& name);
};


// API interface for user
bool user_track_buffer(void *location, const std::string& name);

} // namespace meminf

// some convenient aliases
void TRACK_BUFFER(void *location, const std::string& name);
