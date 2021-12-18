#include <iostream>
#include <iomanip>
#include <cstdlib>

#include <cuda_runtime.h>

#include "common.h"
#include "matrix.h"


CPUMatrix matrix_alloc_cpu(int width, int height)
{
	CPUMatrix m;
	m.width = width;
	m.height = height;
	m.elements = new float[m.width * m.height];
	return m;
}
void matrix_free_cpu(CPUMatrix &m)
{
	delete[] m.elements;
}

GPUMatrix matrix_alloc_gpu(int width, int height)
{
	// TODO (Task 4): Allocate memory at the GPU
	GPUMatrix mat;
	mat.width = width;
	mat.height = height;
	
	cudaMallocPitch(&mat.elements, &mat.pitch, width * sizeof(float), height);
	CUDA_CHECK_ERROR;

	return mat;
}

void matrix_free_gpu(GPUMatrix &m)
{
	// TODO (Task 4): Free the memory
	cudaFree(m.elements);
	CUDA_CHECK_ERROR;
}

void matrix_upload(const CPUMatrix &src, GPUMatrix &dst)
{
	// TODO (Task 4): Upload CPU matrix to the GPU
	if (src.width != dst.width || src.height != dst.height)
		throw std::runtime_error("Matrices must have equal dimension!");

	cudaMemcpy2D(dst.elements, dst.pitch, src.elements, src.width * sizeof(float),
		src.width * sizeof(float), src.height, cudaMemcpyHostToDevice);
	CUDA_CHECK_ERROR;
}

void matrix_download(const GPUMatrix &src, CPUMatrix &dst)
{
	// TODO (Task 4): Download matrix from the GPU
	if (src.width != dst.width || src.height != dst.height)
		throw std::runtime_error("Matrices must have equal dimension!");

	cudaMemcpy2D(dst.elements, dst.width * sizeof(float), src.elements, src.pitch,
		src.width * sizeof(float), src.height, cudaMemcpyDeviceToHost);
	CUDA_CHECK_ERROR;
}

void matrix_compare_cpu(const CPUMatrix &a, const CPUMatrix &b)
{
	// TODO (Task 4): compare both matrices a and b and print differences to the console
	for (int i = 0; i < a.height; ++i)
		for (int j = 0; j < a.width; ++j) {
			float valA = a.elements[i * a.width + j];
			float valB = b.elements[i * b.width + j];

			if (valA != valB)
				std::cout << "entries (" << i << ", " << j << ") do not match: " <<
					"a=" << valA << " b=" << valB << std::endl;
		}
}
