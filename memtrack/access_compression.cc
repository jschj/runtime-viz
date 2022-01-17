#include "access_compression.h"


namespace memtrack
{

access_compression::access_compression(const std::string& target_file_path)
{
    gz_file = gzopen(target_file_path.c_str(), "w");

    if (!gz_file)
        throw std::runtime_error("Could not open dump file!");

    if (gzsetparams(gz_file, Z_DEFAULT_COMPRESSION, Z_DEFAULT_STRATEGY) != Z_OK)
        throw std::runtime_error("Could nto set params!");
}

access_compression::~access_compression()
{
    gzclose(gz_file);
}

void access_compression::track_access(const raw_buffer_access& access)
{
    if (!gzwrite(gz_file, &access, sizeof(access)))
        throw std::runtime_error("Could not write access to file!");
}

}