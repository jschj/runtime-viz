#include "malloc_track.h"

#include "memtrack.h"


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
            {
                cuda_address_t location = static_cast<cuda_address_t>(*allocation_parameters.cuMemAlloc.dptr);
                cuda_address_t buf_size = allocation_parameters.cuMemAlloc.bytesize;
                range = device_buffer_range(location, location + buf_size);
            }
            break;
        case API_CUDA_cuMemAllocPitch:
            allocation_parameters.cuMemAllocPitch = *reinterpret_cast<cuMemAllocPitch_params *>(params);
            allocation_type = allocation_type::cuMemAllocPitch;
            {
                cuda_address_t location = static_cast<cuda_address_t>(*allocation_parameters.cuMemAllocPitch.dptr);
                pitch = *allocation_parameters.cuMemAllocPitch.pPitch;
                cuda_address_t buf_size = pitch * allocation_parameters.cuMemAllocPitch.Height;
                range = device_buffer_range(location, location + buf_size);
            }
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
            {
                cuda_address_t location = static_cast<cuda_address_t>(*allocation_parameters.cuMemAllocPitch_v2.dptr);
                pitch = *allocation_parameters.cuMemAllocPitch_v2.pPitch;
                cuda_address_t buf_size = pitch * allocation_parameters.cuMemAllocPitch_v2.Height;
                range = device_buffer_range(location, location + buf_size);
            }
            break;
        /*
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
         */
        default:
            throw std::runtime_error("Unsupported API call!");
    }
}

size_t device_buffer::get_elem_type_size() const noexcept
{
    switch (type) {
        case type_float: return sizeof(float);
        case type_double: return sizeof(double);
        case type_int32: return sizeof(int32_t);
        case type_int64: return sizeof(int64_t);
        default: return sizeof(char);
    }
}

std::string device_buffer::get_elem_type_name() const
{
    switch (type) {
        case type_float: return "t_float";
        case type_double: return "t_double";
        case type_int32: return "t_int32";
        case type_int64: return "t_int64";
        default: return "t_char";
    }
}

bool device_buffer::is_pitched() const noexcept
{
    return (allocation_type == allocation_type::cuMemAllocPitch ||
        allocation_type == allocation_type::cuMemAllocPitch_v2);
}

size_t device_buffer::address_to_index(cuda_address_t addr) const noexcept
{
    if (is_pitched()) {
        cuda_address_t off = addr - range.from;
        cuda_address_t x = (off % pitch) / get_elem_type_size();
        cuda_address_t y = off / pitch;

        return y * get_width() + x;
    }

    return (addr - range.from) / get_elem_type_size();
}

size_t device_buffer::get_width() const noexcept
{
    if (allocation_type == allocation_type::cuMemAllocPitch)
        return allocation_parameters.cuMemAllocPitch.WidthInBytes / get_elem_type_size();
    else if (allocation_type == allocation_type::cuMemAllocPitch_v2)
        return allocation_parameters.cuMemAllocPitch_v2.WidthInBytes / get_elem_type_size();

    return 0;
}

size_t device_buffer::get_height() const noexcept
{
    return range.size() / pitch;
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


    for (auto& buf : user_buffers) {
        if (buf.id != idx->second.id)
            continue;

        buf.free_time = now;
        break;
    }

    active_buffers.erase(idx);
}

void device_buffer_tracker::on_free(nvbit_api_cuda_t cbid, void *params)
{
    on_free(get_free_address(cbid, params));
}

void device_buffer_tracker::user_track_buffer(void *location, const std::string& name, device_buffer::element_type type)
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
    idx->second.type = type;
    user_buffers.emplace_back(idx->second);
}

std::string device_buffer_tracker::get_info_string() const
{
    throw std::runtime_error("Not implemented!");
}

void device_buffer_tracker::find_associated_buffers(util::time_point when, const cuda_address_t addresses[32],
        uint32_t ids[32], uint64_t indices[32]) const
{
    for (uint32_t i = 0; i < 32; ++i) {
        if (!addresses[i])
            continue;

        /*
        device_buffer_range search_range(addresses[i]);
        auto range = user_buffers.equal_range(search_range);

        if (range.first == range.second) {
            goto error;
        }
         */

        for (const auto& it : user_buffers) {
            if (it.range.in_range(addresses[i]) && it.was_active_at(when)) {
                ids[i] = it.id;
                //indices[i] = (addresses[i] - it.range.from) / it.second.get_elem_type_size();
                indices[i] = it.address_to_index(addresses[i]);
                goto next;
            }
        }

        //for (auto it = user_buffers.cbegin(); it != user_buffers.cend(); it++) {
        //    // this should have exactly one match
//
        //    if (it.in_range(addresses[i]) && it->second.was_active_at(when)) {
        //        ids[i] = it->second.id;
        //        indices[i] = (addresses[i] - it->first.from) / it->second.get_elem_type_size();
        //        goto next;
        //    }
        //}

error:
        throw std::runtime_error("No match found for given address and timepoint!");

next:
        void();
    }
}

std::string device_buffer_tracker::get_buffer_info_string() const
{
    std::stringstream ss;

    for (const auto& it : user_buffers) {
        //ss << it.second.name_tag << '\n'
        //    << "range from " << std::hex << it.first.from << " to " << it.first.to << '\n'
        //    << "alive from " << std::dec << util::time_to_ns(it.second.malloc_time) << " to "
        //    << util::time_to_ns(it.second.free_time) << '\n';
    }

    return ss.str();
}

static device_buffer_tracker instance;

device_buffer_tracker& tracker()
{
    return instance;
}

static void track_buffer_types(void *location, const char *name, device_buffer::element_type type)
{
    try {
        memtrack::tracker().user_track_buffer(location, name, type);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'
            << "Hint: Was NVbit previously attached via the LD_PRELOAD trick?" << std::endl;
        throw e;
    }
}

} // namespace memtrack

// implement user API
template <>
void TRACK_BUFFER<float>(float *location, const char *name)
{
    memtrack::track_buffer_types(location, name, memtrack::device_buffer::element_type::type_float);
}

template <>
void TRACK_BUFFER<double>(double *location, const char *name)
{
    memtrack::track_buffer_types(location, name, memtrack::device_buffer::element_type::type_double);
}
