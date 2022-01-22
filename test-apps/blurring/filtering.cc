#include "filtering.h"
#include "image.h"
#include "common.h"
#include "conv_cpu.h"
#include "conv_gpu.h"

#include <assert.h>

void filtering(const char *imgfile, int ks)
{
    if(ks > MAX_FILTER_SIZE) {
        std::cerr << "Maximum filter size is " << MAX_FILTER_SIZE << "." << std::endl;
        exit(EXIT_FAILURE);
    }

	// === Task 1 ===
	// Load image
    image_cpu vanilla_image_cpu(imgfile);
    image_gpu vanilla_image_gpu(vanilla_image_cpu.width, vanilla_image_cpu.height);
    vanilla_image_cpu.upload(vanilla_image_gpu);
    image_gpu output_tmp_gpu(vanilla_image_cpu.width, vanilla_image_cpu.height);
    image_cpu output_tmp_cpu(vanilla_image_cpu.width, vanilla_image_cpu.width);

	// Generate gaussian filter kernel
    filterkernel_cpu kernel_cpu(ks);
    filterkernel_gpu kernel_gpu(ks);
    kernel_cpu.upload(kernel_gpu);

	// Blur image on CPU
    {
        image_cpu blurred_cpu(vanilla_image_cpu.width, vanilla_image_cpu.height);
        image_cpu tmp(vanilla_image_cpu.width, vanilla_image_cpu.height);
        conv_h_cpu(tmp, vanilla_image_cpu, kernel_cpu);
        conv_v_cpu(blurred_cpu, tmp, kernel_cpu);
        blurred_cpu.save("out_cpu.ppm");
    }

    image_gpu tmp_gpu(vanilla_image_cpu.width, vanilla_image_cpu.height);

	// === Task 2 ===
	// Blur image on GPU (Global memory)
    conv_h_gpu_gmem(tmp_gpu, vanilla_image_gpu, kernel_gpu);
    conv_v_gpu_gmem(output_tmp_gpu, tmp_gpu, kernel_gpu);
    output_tmp_cpu.download(output_tmp_gpu);
    output_tmp_cpu.save("out_gpu_gmem.ppm");

	// === Task 3 ===
	// Blur image on GPU (Shared memory)
    conv_h_gpu_smem(tmp_gpu, vanilla_image_gpu, kernel_gpu);
    conv_v_gpu_smem(output_tmp_gpu, tmp_gpu, kernel_gpu);
    output_tmp_cpu.download(output_tmp_gpu);
    output_tmp_cpu.save("out_gpu_smem.ppm");

	// === Task 4 ===
	// Blur image on GPU (Constant memory)
    conv_h_gpu_cmem(tmp_gpu, vanilla_image_gpu, kernel_gpu);
    conv_v_gpu_cmem(output_tmp_gpu, tmp_gpu, kernel_gpu);
    output_tmp_cpu.download(output_tmp_gpu);
    output_tmp_cpu.save("out_gpu_cmem.ppm");

	// === Task 5 ===
	// Blur image on GPU (all memory types)
    conv_h_gpu_all(tmp_gpu, vanilla_image_gpu, kernel_gpu);
    conv_v_gpu_all(output_tmp_gpu, tmp_gpu, kernel_gpu);
    output_tmp_cpu.download(output_tmp_gpu);
    output_tmp_cpu.save("out_gpu_all.ppm");
}


/************************************************************
 * 
 * Write your text answers here!
 * 
 * (Task 7) nvprof output
 * 
 * Answer:
parzt@gccg201:/tmp/tmp.KLDCunTSCw/__build$ nvprof ./gauss_filter ../cornellBoxSphere_2048x2048.ppm 63
PMPP Hello World!
==37907== NVPROF is profiling process 37907, command: ./gauss_filter ../cornellBoxSphere_2048x2048.ppm 63
==37907== Profiling application: ./gauss_filter ../cornellBoxSphere_2048x2048.ppm 63
==37907== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   21.20%  13.917ms         4  3.4792ms  1.7661ms  8.5285ms  [CUDA memcpy DtoH]
                   16.14%  10.593ms         1  10.593ms  10.593ms  10.593ms  conv_h_gpu_gmem_kernel(unsigned int*, unsigned int*, int, int, unsigned long, unsigned long, int, float*)
                   15.38%  10.096ms         1  10.096ms  10.096ms  10.096ms  conv_v_gpu_gmem_kernel(unsigned int*, unsigned int*, int, int, unsigned long, unsigned long, int, float*)
                   13.81%  9.0661ms         1  9.0661ms  9.0661ms  9.0661ms  conv_h_gpu_smem_kernel(unsigned int*, unsigned int*, int, int, unsigned long, unsigned long, int, float*)
                   13.77%  9.0370ms         1  9.0370ms  9.0370ms  9.0370ms  conv_v_gpu_smem_kernel(unsigned int*, unsigned int*, int, int, unsigned long, unsigned long, int, float*)
                    6.01%  3.9440ms         1  3.9440ms  3.9440ms  3.9440ms  conv_v_gpu_cmem_kernel(unsigned int*, unsigned int*, int, int, unsigned long, unsigned long, int)
                    4.37%  2.8666ms         1  2.8666ms  2.8666ms  2.8666ms  conv_h_gpu_cmem_kernel(unsigned int*, unsigned int*, int, int, unsigned long, unsigned long, int)
                    3.87%  2.5424ms         2  1.2712ms  1.5360us  2.5409ms  [CUDA memcpy HtoD]
                    2.76%  1.8109ms         1  1.8109ms  1.8109ms  1.8109ms  conv_v_gpu_all_kernel(unsigned int*, unsigned int*, int, int, unsigned long, unsigned long, int)
                    2.70%  1.7703ms         1  1.7703ms  1.7703ms  1.7703ms  conv_h_gpu_all_kernel(unsigned int*, unsigned int*, int, int, unsigned long, unsigned long, int)
                    0.01%  6.4960us         4  1.6240us  1.4400us  2.1440us  [CUDA memcpy DtoD]
      API calls:   71.24%  204.58ms         2  102.29ms  292.89us  204.29ms  cudaMallocPitch
                   17.15%  49.235ms         8  6.1543ms  1.7697ms  10.603ms  cudaDeviceSynchronize
                    6.36%  18.251ms         5  3.6501ms  1.9250ms  9.7387ms  cudaMemcpy2D
                    3.32%  9.5241ms       808  11.787us     175ns  572.62us  cuDeviceGetAttribute
                    0.63%  1.8225ms         3  607.50us  214.20us  1.2542ms  cudaFree
                    0.53%  1.5268ms         8  190.85us  152.02us  246.64us  cuDeviceTotalMem
                    0.37%  1.0647ms         8  133.09us  7.8520us  849.76us  cudaLaunchKernel
                    0.25%  714.36us         8  89.295us  84.135us  107.35us  cuDeviceGetName
                    0.08%  231.52us         1  231.52us  231.52us  231.52us  cudaMalloc
                    0.05%  144.08us         4  36.018us  9.9440us  69.459us  cudaMemcpyToSymbol
                    0.01%  24.184us         8  3.0230us  1.3620us  9.4070us  cuDeviceGetPCIBusId
                    0.01%  15.094us         1  15.094us  15.094us  15.094us  cudaMemcpy
                    0.00%  11.770us        32     367ns     133ns  3.2810us  cudaGetLastError
                    0.00%  5.1740us        16     323ns     214ns  1.1830us  cuDeviceGet
                    0.00%  2.4720us         8     309ns     242ns     397ns  cuDeviceGetUuid
                    0.00%  1.9810us         3     660ns     295ns  1.1950us  cuDeviceGetCount
 ************************************************************/
