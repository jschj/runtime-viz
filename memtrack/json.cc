#include "json.h"



namespace memtrack
{
    using jsoncons::json;
    using jsoncons::ojson;
    namespace bson = jsoncons::bson; // for brevity


    streaming_bson_encoder::streaming_bson_encoder(const std::string& file_name, const std::string& access_file_name):
        out_file(file_name, std::ios::binary),
        encoder(out_file),
        acc_file(access_file_name, std::ios::binary)
    {
        if (!out_file.is_open() || !acc_file.is_open())
            throw std::runtime_error("Could not open dump files!");
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

} // namespace memtrack
