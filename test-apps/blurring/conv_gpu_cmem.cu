#include "conv_gpu.h"

#include <assert.h>

// =====================
//    CONSTANT MEMORY
// =====================
__constant__ float constant_kernel[MAX_FILTER_SIZE * MAX_FILTER_SIZE];


__global__ void conv_h_gpu_cmem_kernel(
        unsigned int* src,
        unsigned int* dst,
        int w,
        int h,
        size_t spitch,
        size_t dpitch,
        int ks) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < w && y < h) {
        float rr = 0.0f, gg = 0.0f, bb = 0.0f;

        for (int i = 0; i < ks; i++) {
            int xx = x + (i - ks / 2);

            // clamp
            xx = max<int>(min<int>(xx, w-1), 0);

            // get pixel
            unsigned int pixel = src[y * spitch + xx];

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

        dst[y * dpitch + x] = rr_c | (gg_c << 8) | (bb_c << 16);
    }
}

__global__ void conv_v_gpu_cmem_kernel(
        unsigned int* src,
        unsigned int* dst,
        int w,
        int h,
        size_t spitch,
        size_t dpitch,
        int ks) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w && y < h) {
        float rr = 0.0f, gg = 0.0f, bb = 0.0f;

        for (int i = 0; i < ks; i++) {
            int yy = y + (i - ks / 2);

            // clamp
            yy = max<int>(min<int>(yy, h-1), 0);

            // get pixel
            unsigned int pixel = src[yy * spitch + x];

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

        dst[y * dpitch + x] = rr_c | (gg_c << 8) | (bb_c << 16);
    }
}

void conv_h_gpu_cmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel) {
    assert(dst.width == src.width);
    assert(dst.height == src.height);

    // copy filter kernel to constant memory
    cudaMemcpyToSymbol(constant_kernel, kernel.data, kernel.ks * sizeof(float));
    CUDA_CHECK_ERROR;

    dim3 dimBlock(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
    dim3 dimGrid(div_up(src.width, dimBlock.x), div_up(src.height, dimBlock.y));
    conv_h_gpu_cmem_kernel<<<dimGrid, dimBlock>>>(
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

void conv_v_gpu_cmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel) {
    // copy filter kernel to constant memory
    cudaMemcpyToSymbol(constant_kernel, kernel.data, kernel.ks * sizeof(float));
    CUDA_CHECK_ERROR;

    dim3 dimBlock(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
    dim3 dimGrid(div_up(src.width, dimBlock.x), div_up(src.height, dimBlock.y));
    conv_v_gpu_cmem_kernel<<<dimGrid, dimBlock>>>(
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