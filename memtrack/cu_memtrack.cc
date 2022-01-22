#include "cu_memtrack.h"

#include <memory>
#include <limits>
#include <atomic>

#include <jsoncons/json.hpp>
#include <jsoncons_ext/jsonpath/jsonpath.hpp>
#include <jsoncons_ext/bson/bson.hpp>

#include "access_compression.h"
#include "malloc_track.h"


namespace memtrack
{
    std::string json_path;
    int64_t device_host_time_difference;
    std::unique_ptr<access_compression> acc_comp;


    void cu_memtrack_init(const std::string& json_file_path, const std::string& access_dump_file)
    {
        json_path = json_file_path;
        acc_comp = std::make_unique<access_compression>();
    }

    void cu_memtrack_malloc(nvbit_api_cuda_t cbid, void *params)
    {
        tracker().on_malloc(cbid, params);
    }

    void cu_memtrack_free(nvbit_api_cuda_t cbid, void *params)
    {
        tracker().on_free(cbid, params);
    }

    void cu_memtrack_access(const mem_access_t& access)
    {
        uint32_t ids[32];
        uint64_t indices[32];
        // TODO: This might change in the future!
        util::time_point when{
            std::chrono::duration_cast<util::time_point::duration>(std::chrono::nanoseconds(access.when - device_host_time_difference))
        };
        tracker().find_associated_buffers(when, access.addrs, ids, indices);

        for (uint32_t i = 0; i < 32; ++i) {
            // NULL means no access!    
            if (!access.addrs[i])
                continue;

            access_compression::raw_buffer_access acc{
                .buffer_id = static_cast<uint8_t>(ids[i]),
                .time_point = access.when - device_host_time_difference,
                .index = static_cast<uint32_t>(indices[i]),
            };

            acc_comp->track_access(acc);
        }
    }

    void cu_memtrack_dump_buffers()
    {
        std::ofstream stream(json_path, std::ios_base::out);

        if (!stream.is_open())
            throw std::runtime_error("Could not open JSON file!");

        jsoncons::json_stream_encoder enc(stream);

        enc.begin_object();
        enc.key("buffers");

        enc.begin_array();

        std::vector<device_buffer> user_buffers = tracker().get_user_buffers_copy();

        for (const auto& entry_pair : user_buffers) {
            bool is_pitched = entry_pair.is_pitched();

            enc.begin_object();
            
            enc.key("id");
            enc.uint64_value(entry_pair.id);

            enc.key("type");
            enc.string_value(is_pitched ? "pitched" : "plain");

            enc.key("name");
            enc.string_value(entry_pair.name_tag);

            if (is_pitched) {
                enc.key("height");
                enc.uint64_value(entry_pair.get_height());

                enc.key("width");
                enc.uint64_value(entry_pair.get_width());

                // TODO: Should this be in bytes?
                enc.key("pitch");
                enc.uint64_value(entry_pair.pitch);
            } else {
                enc.key("height");
                enc.uint64_value(entry_pair.range.size() / entry_pair.get_elem_type_size());
            }

            enc.key("type_name");
            enc.string_value(entry_pair.get_elem_type_name());

            enc.key("first_access_time");
            enc.uint64_value(entry_pair.first_access_time.time_since_epoch().count());

            enc.key("last_access_time");
            enc.uint64_value(entry_pair.last_access_time.time_since_epoch().count());

            enc.end_object();
        }

        enc.end_array();

        enc.key("access_files");
        enc.begin_array();

        for (const std::string& file_name : acc_comp->get_tracked_file_names())
            enc.string_value(file_name);

        enc.end_array();

        enc.end_object();
        enc.flush();
    }

    void cu_memtrack_set_time_difference(int64_t delta)
    {
        device_host_time_difference = delta;
    }

    void cu_memtrack_attach_to_kernel(const std::string& kernel_name)
    {
        acc_comp->attach_to_kernel(kernel_name);
    }
}
