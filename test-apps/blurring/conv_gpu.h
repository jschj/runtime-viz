#pragma once

#include "image.h"
#include "common.h"

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

const size_t THREAD_BLOCK_SIZE = 32;
const size_t MAX_FILTER_SIZE = THREAD_BLOCK_SIZE * 2;
const size_t MAX_FILTER_RADIUS = MAX_FILTER_SIZE / 2;

// ======================
//    HELPER FUNCTIONS
// ======================
template<class T>
CUDA_HOSTDEV T max(T a, T b) {
    return a > b ? a : b;
}

template<class T>
CUDA_HOSTDEV T min(T a, T b) {
    return a < b ? a : b;
}

void conv_h_gpu_gmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel);
void conv_v_gpu_gmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel);

void conv_h_gpu_smem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel);
void conv_v_gpu_smem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel);

void conv_h_gpu_cmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel);
void conv_v_gpu_cmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel);

void conv_h_gpu_tmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel);
void conv_v_gpu_tmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel);

void conv_h_gpu_all(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel);
void conv_v_gpu_all(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel);
