#include "malloc_trace.h"


namespace meminf {

bool is_malloc_call(nvbit_api_cuda_t cbid)
{
    return (
        cbid == API_CUDA_cuMemAlloc ||
        cbid == API_CUDA_cuMemAllocPitch ||
        cbid == API_CUDA_cuMemAllocHost ||
        cbid == API_CUDA_cuMemHostAlloc ||
        cbid == API_CUDA_cuMemAlloc_v2 ||
        cbid == API_CUDA_cuMemAllocPitch_v2 ||
        cbid == API_CUDA_cuMemHostAlloc_v2 ||
        cbid == API_CUDA_cuMemAllocHost_v2 ||
        cbid == API_CUDA_cuMemAllocManaged ||
        cbid == API_CUDA_cuMemGetAllocationGranularity ||
        cbid == API_CUDA_cuMemGetAllocationPropertiesFromHandle ||
        cbid == API_CUDA_cuMemRetainAllocationHandle ||
        cbid == API_CUDA_cuMemAllocAsync ||
        cbid == API_CUDA_cuMemAllocAsync_ptsz ||
        cbid == API_CUDA_cuMemAllocFromPoolAsync ||
        cbid == API_CUDA_cuMemAllocFromPoolAsync_ptsz
    );
}

bool is_free_call(nvbit_api_cuda_t cbid)
{
    return (
        cbid == API_CUDA_cuMemFree ||
        cbid == API_CUDA_cu64MemFree ||
        cbid == API_CUDA_cuMemFreeHost ||
        cbid == API_CUDA_cuMemFree_v2 ||
        cbid == API_CUDA_cuBinaryFree ||
        cbid == API_CUDA_cuMemAddressFree ||
        cbid == API_CUDA_cuMemFreeAsync ||
        cbid == API_CUDA_cuMemFreeAsync_ptsz ||
        cbid == API_CUDA_cuGraphAddMemFreeNode ||
        cbid == API_CUDA_cuGraphMemFreeNodeGetParams
    );
}

device_buffer::device_buffer(nvbit_api_cuda_t cbid, void *params)
{
    switch (cbid) {
        case API_CUDA_cuMemAlloc:
            allocation_parameters.cuMemAlloc = *reinterpret_cast<cuMemAlloc_params *>(params);
            allocation_type = allocation_type::cuMemAlloc;
            break;
        case API_CUDA_cuMemAllocPitch:
            allocation_parameters.cuMemAllocPitch = *reinterpret_cast<cuMemAllocPitch_params *>(params);
            allocation_type = allocation_type::cuMemAllocPitch;
            break;
        case API_CUDA_cuMemAlloc_v2:
            allocation_parameters.cuMemAlloc_v2 = *reinterpret_cast<cuMemAlloc_v2_params *>(params);
            allocation_type = allocation_type::cuMemAlloc_v2;
            break;
        case API_CUDA_cuMemAllocPitch_v2:
            allocation_parameters.cuMemAllocPitch_v2 = *reinterpret_cast<cuMemAllocPitch_v2_params *>(params);
            allocation_type = allocation_type::cuMemAllocPitch_v2;
            break;
        case API_CUDA_cuMemAllocManaged:
            allocation_parameters.cuMemAllocManaged = *reinterpret_cast<cuMemAllocManaged_params *>(params);
            allocation_type = allocation_type::cuMemAllocManaged;
            break;
        case API_CUDA_cuMemAllocAsync:
            allocation_parameters.cuMemAllocAsync = *reinterpret_cast<cuMemAllocAsync_params *>(params);
            allocation_type = allocation_type::cuMemAllocAsync;
            break;
        case API_CUDA_cuMemAllocAsync_ptsz:
            allocation_parameters.cuMemAllocAsync_ptsz = *reinterpret_cast<cuMemAllocAsync_ptsz_params *>(params);
            allocation_type = allocation_type::cuMemAllocAsync_ptsz;
            break;
        case API_CUDA_cuMemAllocFromPoolAsync:
            allocation_parameters.cuMemAllocFromPoolAsync = *reinterpret_cast<cuMemAllocFromPoolAsync_params *>(params);
            allocation_type = allocation_type::cuMemAllocFromPoolAsync;
            break;
        case API_CUDA_cuMemAllocFromPoolAsync_ptsz:
            allocation_parameters.cuMemAllocFromPoolAsync_ptsz = *reinterpret_cast<cuMemAllocFromPoolAsync_ptsz_params *>(params);
            allocation_type = allocation_type::cuMemAllocFromPoolAsync_ptsz;
            break;
        default:
            throw std::runtime_error("Unsupported API call!");
    }
}


} // namespace meminf