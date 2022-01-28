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
    gzFile gz_file = nullptr;
    std::vector<std::string> file_names;
public:
    ~access_compression();
    void attach_to_kernel(const std::string& kernel_name);
    void track_access(const raw_buffer_access& access);
    std::vector<std::string> get_tracked_file_names() const;
};

} // namespace memtrack