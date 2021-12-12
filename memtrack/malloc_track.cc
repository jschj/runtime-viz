#include "malloc_track.h"


namespace memtrack {

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

void *get_free_address(nvbit_api_cuda_t cbid, void *params)
{
    uint64_t free_address = 0;

    switch (cbid) {
        case API_CUDA_cuMemFree:
            free_address = reinterpret_cast<cuMemFree_params *>(params)->dptr;
            break;
        case API_CUDA_cuMemFree_v2:
            free_address = reinterpret_cast<cuMemFree_v2_params *>(params)->dptr;
            break;
        case API_CUDA_cuMemFreeAsync:
            free_address = reinterpret_cast<cuMemFreeAsync_params *>(params)->dptr;
            break;
        case API_CUDA_cuMemFreeAsync_ptsz:
            free_address = reinterpret_cast<cuMemFreeAsync_ptsz_params *>(params)->dptr;
            break;
        default:
            throw std::runtime_error("Unsupported nvbit_api_cuda_t value for free!");
    }

    return reinterpret_cast<void *>(free_address);
}

device_buffer::device_buffer(nvbit_api_cuda_t cbid, void *params, uint32_t buffer_id)
{
    malloc_time = util::now();
    id = buffer_id;

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
            {
                cuda_address_t location = static_cast<cuda_address_t>(*allocation_parameters.cuMemAlloc_v2.dptr);
                cuda_address_t buf_size = allocation_parameters.cuMemAlloc_v2.bytesize;
                range = device_buffer_range(location, location + buf_size);
            }
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


void device_buffer_tracker::on_malloc(nvbit_api_cuda_t cbid, void *params)
{
    device_buffer buf(cbid, params, next_buffer_id++);
    std::unique_lock<std::mutex> lk(mut);
    active_buffers.emplace(reinterpret_cast<void *>(buf.range.from), buf);
}

void device_buffer_tracker::on_free(void *location)
{
    util::time_point now = util::now();

    std::unique_lock<std::mutex> lk(mut);

    auto idx = active_buffers.find(location);

    if (idx == active_buffers.end())
        throw std::runtime_error("No such buffer is currently tracked!");


    auto range = user_buffers.equal_range(idx->second.range);

    for (auto it = range.first; it != range.second; it++)
        if (it->second.id == idx->second.id)
            it->second.free_time = now;

    active_buffers.erase(idx);
}

void device_buffer_tracker::on_free(nvbit_api_cuda_t cbid, void *params)
{
    on_free(get_free_address(cbid, params));
}

void device_buffer_tracker::user_track_buffer(void *location, const std::string& name)
{
    std::unique_lock<std::mutex> lk(mut);
    
    // names must be unique
    if (assigned_names.find(name) != assigned_names.end())
        throw std::runtime_error("Buffer names must be unique accross all buffers in the program!");

    assigned_names.insert(name);
    
    // find active buffer and copy it to the tracked buffers
    auto idx = active_buffers.find(location);

    if (idx == active_buffers.end())
        throw std::runtime_error("No such buffer is currently tracked!");

    idx->second.name_tag = name;
    user_buffers.emplace(idx->second.range, idx->second);
}

std::string device_buffer_tracker::get_info_string() const
{
    throw std::runtime_error("Not implemented!");
}

void device_buffer_tracker::find_associated_buffer_ids(util::time_point when, const cuda_address_t addresses[32], uint32_t ids[32]) const
{
    // when tells us the point time when this memory access happened
    // by now the associated buffer can very well be freed again

    for (uint32_t i = 0; i < 32; ++i) {
        device_buffer_range search_range(addresses[i]);
        auto range = user_buffers.equal_range(search_range);

        if (range.first == range.second) {
            goto error;
        }

        for (auto it = range.first; it != range.second; it++) {
            // this should have exactly one match

            if (it->first.in_range(addresses[i])) {
                if (!it->second.was_active_at(when)) {
                    bool a = when >= it->second.malloc_time;
                    bool b = when.time_since_epoch() >= it->second.malloc_time.time_since_epoch();
                    bool c = std::chrono::duration_cast<std::chrono::seconds>(when.time_since_epoch()).count() >=
                        std::chrono::duration_cast<std::chrono::seconds>(it->second.malloc_time.time_since_epoch()).count();

                    assert(a == b);
                    assert(a == c);

                    /*
                    std::cout << a << " " << b << " " << c << std::endl;
                    std::cout << std::chrono::duration_cast<std::chrono::seconds>(it->second.free_time.time_since_epoch()).count() << std::endl;

                    std::cout << "not active: access = " << std::chrono::duration_cast<std::chrono::seconds>(when.time_since_epoch()).count()
                        << " malloc_time = " << std::chrono::duration_cast<std::chrono::seconds>(it->second.malloc_time.time_since_epoch()).count() << std::endl;
                     */
                }

                ids[i] = it->second.id;
                //continue;
                goto next;
            }
        }

error:

        if (range.first == range.second) {
            std::cout << "WARNING: EMPTY RANGE!\n"
                << std::hex << "(" << range.first->first.from << ", " << range.first->first.to << ") "
                << "(" << range.second->first.from << ", " << range.second->first.to << ")\n";
        }

        // no match found, bad!
        search_range = device_buffer_range(addresses[i]);
        std::cout << std::hex << "address: " << addresses[i]
            << " at " << std::dec << util::time_to_ns(when) << std::endl;

        for (const auto& it : user_buffers) {
            std::cout << it.first.from << " TO " << it.first.to << ": "
                << it.second.name_tag << std::endl;
            std::cout << "cmp: " << (it.first < search_range) << std::endl;
            std::cout << "active: " << it.second.was_active_at(when) << std::endl;
            std::cout << std::dec << (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(it.second.malloc_time - util::time_zero()).count() << ' ' << i << std::endl;
        }

        throw std::runtime_error("No match found for given address and timepoint!");

next:
        void();
    }
}

std::string device_buffer_tracker::get_buffer_info_string() const
{
    std::stringstream ss;

    for (const auto& it : user_buffers) {
        ss << it.second.name_tag << '\n'
            << "range from " << std::hex << it.first.from << " to " << it.first.to << '\n'
            << "alive from " << std::dec << util::time_to_ns(it.second.malloc_time) << " to "
            << util::time_to_ns(it.second.free_time) << '\n';
    }

    return ss.str();
}

static device_buffer_tracker instance;

device_buffer_tracker& tracker()
{
    return instance;
}

} // namespace memtrack

// implement user API
void TRACK_BUFFER(void *location, const char *name)
{
    try {
        memtrack::tracker().user_track_buffer(location, name);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'
            << "Hint: Was NVbit previously attached via the LD_PRELOAD trick?" << std::endl;
        throw e;
    }
}
