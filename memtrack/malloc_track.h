#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <map>
#include <limits>

#include <nvbit.h>

#include "util.h"


namespace memtrack
{

// exported utility functions
bool is_malloc_call(nvbit_api_cuda_t cbid);
bool is_free_call(nvbit_api_cuda_t cbid);
void *get_free_address(nvbit_api_cuda_t cbid, void *params);

typedef uint64_t cuda_address_t;

struct device_buffer_range
{
    cuda_address_t from;
    cuda_address_t to;

    device_buffer_range() {}
    device_buffer_range(cuda_address_t lo, cuda_address_t hi): from(lo), to(hi) {}
    device_buffer_range(cuda_address_t lo): from(lo), to(lo) {}

    bool in_range(cuda_address_t addr) const noexcept { return from <= addr && addr < to; }
    uint64_t size() const noexcept { return to - from; }

    bool operator <(const device_buffer_range& other) const noexcept
    {
        if (other.from == other.to) {
            // equal iff. in range
            return !(from <= other.from < to);
        }

        return from < other.from;
    }

    bool operator ==(const device_buffer_range& other) const
    {
        throw std::runtime_error("LALA");
    }
};

struct device_buffer
{
    device_buffer_range range;
    uint32_t id;

    util::time_point malloc_time;
    // can be changed afterwards
    util::time_point free_time = util::time_zero();

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

    device_buffer(nvbit_api_cuda_t cbid, void *params, uint32_t buffer_id) noexcept(false);
    void *location() const noexcept { return reinterpret_cast<void *>(range.from); }

    bool was_active_at(util::time_point when) const noexcept
    {
        return malloc_time <= when &&
            (free_time == util::time_zero() ? true : when <= free_time);            
    }
};

class device_buffer_tracker
{
private:
    // cuda API might be called by multiple threads, protect maps
    mutable std::mutex mut;

    // currently active buffers (not necessarily tracked by user)
    std::unordered_map<void *, device_buffer> active_buffers;
    // user tracked buffers:
    // They aren't touched by on_malloc() and on_free() but must be easily
    // searchable by a given address (if it is in range). That's why the
    // key is an address range rather than a single pointer. For this
    // reason we use map as it uses a sorted tree under the hood which
    // is good for ranges.
    std::multimap<device_buffer_range, device_buffer> user_buffers;

    uint32_t next_buffer_id = 0;
    std::unordered_set<std::string> assigned_names;
public:
    // on malloc()
    void on_malloc(nvbit_api_cuda_t cbid, void *params);
    // on free()
    void on_free(void *location);
    void on_free(nvbit_api_cuda_t cbid, void *params);

    // user decides to track a previously allocated buffer
    void user_track_buffer(void *location, const std::string& name);

    std::string get_info_string() const;

    void find_associated_buffer_ids(util::time_point when, const cuda_address_t addresses[32], uint32_t ids[32]) const;
    void find_associated_buffers(util::time_point when, const cuda_address_t addresses[32],
        uint32_t ids[32], uint64_t indices[32]) const;

    std::string get_buffer_info_string() const;

    // make iterable with foreach
    decltype(user_buffers)::const_iterator begin() const noexcept { return user_buffers.cbegin(); }
    decltype(user_buffers)::const_iterator end() const noexcept { return user_buffers.cend(); }
};

device_buffer_tracker& tracker();

} // namespace memtrack
