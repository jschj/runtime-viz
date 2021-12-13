#include "cu_memtrack.h"

#include <memory>
#include <limits>
#include <atomic>

#include "json.h"


namespace memtrack
{
    std::unique_ptr<streaming_bson_encoder> bson_encoder;
    int64_t device_host_time_difference;


    void cu_memtrack_init(const std::string& json_dump_file)
    {
        bson_encoder = std::make_unique<streaming_bson_encoder>(json_dump_file);
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

        jsoncons::bson::bson_stream_encoder& enc = bson_encoder->get_encoder();

        for (uint32_t i = 0; i < 32; ++i) {
            // NULL means no access!    
            if (!access.addrs[i])
                continue;            

            enc.begin_object();
            enc.key("t");
            // hack to circumvent jsoncons moronic bound checking
            //enc.uint64_value(access.when, jsoncons::semantic_tag::epoch_nano);
            enc.int64_value(static_cast<int64_t>(access.when - device_host_time_difference));
            enc.key("b");
            enc.uint64_value(ids[i]);
            enc.key("i");
            // TODO: causes error if first bit of address is 1
            enc.uint64_value(indices[i]);
            enc.end_object();
        }
    }

    void cu_memtrack_end()
    {
        jsoncons::bson::bson_stream_encoder& enc = bson_encoder->get_encoder();

        enc.end_array();
        enc.key("buffers");

        enc.begin_array();

        for (const auto& entry_pair : tracker()) {
            enc.begin_object();
            
            enc.key("id");
            enc.uint64_value(entry_pair.second.id);

            enc.key("type");
            enc.string_value("plain");

            enc.key("name");
            enc.string_value(entry_pair.second.name_tag);

            enc.key("height");
            enc.uint64_value(entry_pair.second.range.size() / entry_pair.second.get_elem_type_size());

            enc.key("type_name");
            enc.string_value(entry_pair.second.get_elem_type_name());

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
