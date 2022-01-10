#pragma once

#include <fstream>
#include <iostream>
#include <vector>
#include <zlib.h>


namespace memtrack
{

class access_compression
{
public:
    struct raw_buffer_access
    {
        uint8_t buffer_id;
        uint64_t time_point;
        uint32_t index;
    } __attribute__((packed));


private:
    std::vector<raw_buffer_access> raw_accesses;
    size_t raw_accesses_max;

    std::vector<uint8_t> compressed_buffer;
    size_t compressed_buffer_index;

    z_stream stream;

    std::ofstream target_file;

    void set_input_stream();
    void set_output_stream();

    void write_to_file(size_t len);
    void compress_accesses(bool final);
public:
    access_compression(size_t raw_accesses_max, size_t compressed_buffer_size, const std::string& target_file_path);
    ~access_compression();
    void track_access(const raw_buffer_access& access);
    void flush();
};

} // namespace memtrack