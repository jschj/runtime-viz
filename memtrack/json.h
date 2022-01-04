#pragma once

#include <fstream>
#include <vector>

#include <jsoncons/json.hpp>
#include <jsoncons_ext/jsonpath/jsonpath.hpp>
#include <jsoncons_ext/bson/bson.hpp>

#include <zlib.h>

#include <tools/mem_trace/common.h>
#include "malloc_track.h"


namespace memtrack
{
    using jsoncons::json;
    using jsoncons::ojson;
    namespace bson = jsoncons::bson; // for brevity

    /*
        This class directly writes the encoded content to a file. This is useful since
        A LOT of memory transactions can happen and we don't want to keep them in memory.
        However, this requires meta information to written before all memory transactions or after!
     */
    class streaming_bson_encoder
    {
    private:
        std::ofstream out_file;
        bson::bson_stream_encoder encoder;

        // zlib stuff
        //z_stream stream;
        gzFile gz_file;
    public:
        std::ofstream acc_file;
        streaming_bson_encoder(const std::string& bson_file_name, const std::string& access_file_name);
        ~streaming_bson_encoder();

        bson::bson_stream_encoder& get_encoder() { return encoder; }

        void begin();
        void add_access(const mem_access_t& access);
        void end();
        void end(const jsoncons::json& j, const std::string& field_name);
        // raw access functions
        void add_raw_access(uint8_t buffer_id, uint64_t time_point, uint64_t index);
    };
} // namespace memtrack
