#include "conv_gpu.h"

#include <assert.h>

// ======================
//    ALL MEMORY TYPES
// ======================

__constant__ float constant_kernel[MAX_FILTER_SIZE * MAX_FILTER_SIZE];

__global__ void conv_h_gpu_all_kernel(
        unsigned int* src,
        unsigned int* dst,
        int w,
        int h,
        size_t spitch,
        size_t dpitch,
        int ks) {

    assert(blockDim.x == blockDim.y);
    const int block_size = blockDim.x;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // load phase
    // ==========
    const size_t BUFFER_SIZE = 3 * THREAD_BLOCK_SIZE;
    __shared__ unsigned int buffer[BUFFER_SIZE * BUFFER_SIZE];

    // memory region is devided into 9 parts
    for (int i = 0; i <= 2; i++) {
        for (int j = 0; j <= 2; j++) {
            // calculate position in picture
            int p_x = x + (i - 1) * block_size;
            int p_y = y + (j - 1) * block_size;
            p_x = max<int>(min<int>(p_x, w-1), 0);
            p_y = max<int>(min<int>(p_y, h-1), 0);

            // calculate position in buffer
            int b_x = i * block_size + threadIdx.x;
            int b_y = j * block_size + threadIdx.y;

            // save pixel into buffer
            buffer[b_y * BUFFER_SIZE + b_x] = src[p_y * spitch + p_x];
        }
    }

    __syncthreads();

    if(x < w && y < h) {
        float rr = 0.0f, gg = 0.0f, bb = 0.0f;

        for (int i = 0; i < ks; i++) {
            // calculate buffer address
            int b_x = block_size + threadIdx.x + (i - ks / 2);
            int b_y = block_size + threadIdx.y;

            // get pixel from buffer
            unsigned int pixel = buffer[b_y * BUFFER_SIZE + b_x];

            // get colors
            unsigned char r = pixel & 0xff;
            unsigned char g = (pixel >> 8) & 0xff;
            unsigned char b = (pixel >> 16) & 0xff;

            rr += r * constant_kernel[i];
            gg += g * constant_kernel[i];
            bb += b * constant_kernel[i];
        }

        unsigned char rr_c = rr + 0.5f;
        unsigned char gg_c = gg + 0.5f;
        unsigned char bb_c = bb + 0.5f;

        // writeback
        dst[y * dpitch + x] = rr_c | (gg_c << 8) | (bb_c << 16);
    }
}

__global__ void conv_v_gpu_all_kernel(
        unsigned int* src,
        unsigned int* dst,
        int w,
        int h,
        size_t spitch,
        size_t dpitch,
        int ks) {

    assert(blockDim.x == blockDim.y);
    const int block_size = blockDim.x;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // load phase
    // ==========
    const size_t BUFFER_SIZE = 3 * THREAD_BLOCK_SIZE;
    __shared__ unsigned int buffer[BUFFER_SIZE * BUFFER_SIZE];

    // memory region is devided into 9 parts
    for (int i = 0; i <= 2; i++) {
        for (int j = 0; j <= 2; j++) {
            // calculate position in picture
            int p_x = x + (i - 1) * block_size;
            int p_y = y + (j - 1) * block_size;
            p_x = max<int>(min<int>(p_x, w-1), 0);
            p_y = max<int>(min<int>(p_y, h-1), 0);

            // calculate position in buffer
            int b_x = i * block_size + threadIdx.x;
            int b_y = j * block_size + threadIdx.y;

            // save pixel into buffer
            buffer[b_y * BUFFER_SIZE + b_x] = src[p_y * spitch + p_x];
        }
    }

    __syncthreads();

    if(x < w && y < h) {
        float rr = 0.0f, gg = 0.0f, bb = 0.0f;

        for (int i = 0; i < ks; i++) {
            // calculate buffer address
            int b_x = block_size + threadIdx.x;
            int b_y = block_size + threadIdx.y + (i - ks / 2);

            // get pixel from buffer
            unsigned int pixel = buffer[b_y * BUFFER_SIZE + b_x];

            // get colors
            unsigned char r = pixel & 0xff;
            unsigned char g = (pixel >> 8) & 0xff;
            unsigned char b = (pixel >> 16) & 0xff;

            rr += r * constant_kernel[i];
            gg += g * constant_kernel[i];
            bb += b * constant_kernel[i];
        }

        unsigned char rr_c = rr + 0.5f;
        unsigned char gg_c = gg + 0.5f;
        unsigned char bb_c = bb + 0.5f;

        // writeback
        dst[y * dpitch + x] = rr_c | (gg_c << 8) | (bb_c << 16);
    }
}

void conv_h_gpu_all(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel) {
    assert(dst.width == src.width);
    assert(dst.height == src.height);

    // copy filter kernel to constant memory
    cudaMemcpyToSymbol(constant_kernel, kernel.data, kernel.ks * sizeof(float));
    CUDA_CHECK_ERROR;

    dim3 dimBlock(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
    dim3 dimGrid(div_up(src.width, THREAD_BLOCK_SIZE), div_up(src.height, THREAD_BLOCK_SIZE));
    conv_h_gpu_all_kernel<<<dimGrid, dimBlock>>>(
            src.data,
            dst.data,
            src.width,
            src.height,
            src.pitch,
            dst.pitch,
            kernel.ks);
    CUDA_CHECK_ERROR;
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR;
}

void conv_v_gpu_all(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel) {
    assert(dst.width == src.width);
    assert(dst.height == src.height);

    // copy filter kernel to constant memory
    cudaMemcpyToSymbol(constant_kernel, kernel.data, kernel.ks * sizeof(float));
    CUDA_CHECK_ERROR;

    dim3 dimBlock(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
    dim3 dimGrid(div_up(src.width, THREAD_BLOCK_SIZE), div_up(src.height, THREAD_BLOCK_SIZE));
    conv_v_gpu_all_kernel<<<dimGrid, dimBlock>>>(
            src.data,
            dst.data,
            src.width,
            src.height,
            src.pitch,
            dst.pitch,
            kernel.ks);
    CUDA_CHECK_ERROR;
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR;
}