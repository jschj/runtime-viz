#include "json.h"



namespace memtrack
{
    using jsoncons::json;
    using jsoncons::ojson;
    namespace bson = jsoncons::bson; // for brevity


    streaming_bson_encoder::streaming_bson_encoder(const std::string& file_name, const std::string& access_file_name):
        out_file(file_name, std::ios::binary),
        encoder(out_file)
    {
        if (!out_file.is_open())
            throw std::runtime_error("Could not open dump files!");

        gz_file = gzopen(access_file_name.c_str(), "w");

        if (!gz_file)
            throw std::runtime_error("Could not open dump file!");

        if (gzsetparams(gz_file, Z_DEFAULT_COMPRESSION, Z_DEFAULT_STRATEGY) != Z_OK)
            throw std::runtime_error("Could nto set params!");
    }

    streaming_bson_encoder::~streaming_bson_encoder()
    {
        //gzclose(gz_file);
    }

    void streaming_bson_encoder::begin()
    {
        //encoder.key("accesses");
        //encoder.begin_array();
    }

    void streaming_bson_encoder::add_access(const mem_access_t& access)
    {
        
    }

    void streaming_bson_encoder::end()
    {
        encoder.end_array();
        //encoder.flush();
    }

    void streaming_bson_encoder::end(const jsoncons::json& j, const std::string& field_name)
    {
        throw std::runtime_error("Not implemented!");
    }

    void streaming_bson_encoder::add_raw_access(uint8_t buffer_id, uint64_t time_point, uint64_t index)
    {
        struct {
            uint8_t a1; uint64_t a2; uint64_t a3;
        } __attribute__((packed)) tmp_struct { buffer_id, time_point, index };

        if (!gzwrite(gz_file, &tmp_struct, sizeof(tmp_struct)))
            throw std::runtime_error("Could not write data!");
    }

} // namespace memtrack
