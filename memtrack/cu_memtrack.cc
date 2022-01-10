#include "cu_memtrack.h"

#include <memory>
#include <limits>
#include <atomic>

#include "access_compression.h"
#include "json.h"


namespace memtrack
{
    std::unique_ptr<streaming_bson_encoder> bson_encoder;
    int64_t device_host_time_difference;
    std::unique_ptr<access_compression> acc_comp;


    void cu_memtrack_init(const std::string& json_dump_file, const std::string& acc_dump_file)
    {
        bson_encoder = std::make_unique<streaming_bson_encoder>(json_dump_file, acc_dump_file);
        acc_comp = std::make_unique<access_compression>(1000000, 1000000 * 32, "access_dump.bin");
    }

    void cu_memtrack_begin()
    {
        std::cout << "beginning bson" << std::endl;
        bson_encoder->begin();
        std::cout << "done" << std::endl;
        bson_encoder->get_encoder().begin_object();
        std::cout << "done" << std::endl;
        bson_encoder->get_encoder().key("accesses");
        std::cout << "done" << std::endl;
        bson_encoder->get_encoder().begin_array();
        std::cout << "done" << std::endl;
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
                .buffer_id = ids[i],
                .time_point = access.when - device_host_time_difference,
                .index = indices[i],
            };

            acc_comp->track_access(acc);
        }
    }

    void cu_memtrack_end()
    {
        jsoncons::bson::bson_stream_encoder& enc = bson_encoder->get_encoder();

        enc.end_array();
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

            enc.end_object();
        }

        enc.end_array();

        enc.end_object();
        enc.flush();

        bson_encoder.reset();
    }

    void cu_memtrack_set_time_difference(int64_t delta)
    {
        device_host_time_difference = delta;
    }
}
