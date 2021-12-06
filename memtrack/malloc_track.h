#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <mutex>

#include <nvbit.h>

#include "util.h"


namespace memtrack
{

// exported utility functions
bool is_malloc_call(nvbit_api_cuda_t cbid);
bool is_free_call(nvbit_api_cuda_t cbid);
void *get_free_address(nvbit_api_cuda_t cbid, void *params);

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

    util::time_point malloc_time;
    // can be changed afterwards
    util::time_point free_time;

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
    // cuda API might be called by multiple threads, protect maps
    mutable std::mutex mut;

    std::unordered_map<void *, device_buffer> active_buffers;
    std::unordered_multimap<void *, device_buffer> inactive_buffers;
public:
    // on malloc()
    void on_malloc(nvbit_api_cuda_t cbid, void *params);
    // on free()
    void on_free(void *location);
    void on_free(nvbit_api_cuda_t cbid, void *params);
    // user decides to track a previously allocated buffer
    void user_track_buffer(void *location, const std::string& name);

    std::string get_info_string() const;

    std::unordered_multimap<void *, device_buffer>::const_iterator begin() const noexcept { return inactive_buffers.cbegin(); }
    std::unordered_multimap<void *, device_buffer>::const_iterator end() const noexcept { return inactive_buffers.cend(); }
};

device_buffer_tracker& tracker();

} // namespace memtrack
