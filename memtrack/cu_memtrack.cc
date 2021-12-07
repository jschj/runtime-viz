#include "cu_memtrack.h"

#include <memory>
#include "json.h"


namespace memtrack
{
    std::unique_ptr<streaming_bson_encoder> bson_encoder;



    void cu_memtrack_init(const std::string& json_dump_file)
    {
        bson_encoder = std::make_unique<streaming_bson_encoder>(json_dump_file);
    }

    void cu_memtrack_begin()
    {
        bson_encoder->begin();
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

    }

    void cu_memtrack_end()
    {
        jsoncons::bson::bson_stream_encoder& enc = bson_encoder->get_encoder();

        enc.begin_object();
        enc.key("buffers");

        enc.begin_array();

        for (const auto& entry_pair : tracker()) {
            enc.begin_object();
            
            enc.key("bufferId");
            enc.uint64_value(entry_pair.second.id);

            enc.key("bufferType");
            enc.string_value("plain");

            enc.key("name");
            enc.string_value(entry_pair.second.name_tag);

            enc.key("height");
            enc.uint64_value(entry_pair.second.buf_size);

            enc.end_object();
        }

        enc.end_array();

        enc.end_object();
        enc.flush();

        bson_encoder.reset();
    }

}
