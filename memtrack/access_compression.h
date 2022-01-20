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
    gzFile gz_file;
public:
    access_compression(const std::string& target_file_path);
    ~access_compression();
    void track_access(const raw_buffer_access& access);
};

} // namespace memtrack