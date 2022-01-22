#include "access_compression.h"


namespace memtrack
{

access_compression::~access_compression()
{
    gzclose(gz_file);
}

void access_compression::attach_to_kernel(const std::string& kernel_name)
{
    std::string path = kernel_name + ".accesses.bin";
    gz_file = gzopen(kernel_name.c_str(), "w");

    if (!gz_file)
        throw std::runtime_error("Could not open dump file!");

    if (gzsetparams(gz_file, Z_DEFAULT_COMPRESSION, Z_DEFAULT_STRATEGY) != Z_OK)
        throw std::runtime_error("Could nto set params!");

    file_names.push_back(path);
}

void access_compression::track_access(const raw_buffer_access& access)
{
    if (!gzwrite(gz_file, &access, sizeof(access)))
        throw std::runtime_error("Could not write access to file!");
}

std::vector<std::string> access_compression::get_tracked_file_names() const
{
    return file_names;
}

}