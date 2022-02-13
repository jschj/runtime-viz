/*
    This is the interface for usage in mem_trace. Do not include jsoncons headers here as nvcc cannot compile them!
 */

#pragma once

#include <string>

#include <nvbit.h>

#include <tools/mem_trace/common.h>


namespace memtrack
{
    /*
        Cuda doesn't do RAII anyways so we just throw classes out the window and stop worrying about
        non compiling jsoncons sources in nvcc. ;)
     */

    void cu_memtrack_init(const std::string& json_file_path);
    void cu_memtrack_malloc(nvbit_api_cuda_t cbid, void *params);
    void cu_memtrack_free(nvbit_api_cuda_t cbid, void *params);
    void cu_memtrack_access(const mem_access_t& access);
    void cu_memtrack_dump_buffers();
    void cu_memtrack_set_time_difference(int64_t delta);
    void cu_memtrack_attach_to_kernel(const std::string& kernel_name);
}