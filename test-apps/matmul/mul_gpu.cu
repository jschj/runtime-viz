#include <cuda_runtime.h>

// NOTE: if you include stdio.h, you can use printf inside your kernel

#include "common.h"
#include "matrix.h"
#include "mul_gpu.h"

// TODO (Task 4): Implement matrix multiplication CUDA kernel
__global__ void matrix_mul_kernel(GPUMatrix m, GPUMatrix n, GPUMatrix p)
{
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int j = blockDim.x * blockIdx.x + threadIdx.x;

	if (i >= p.height || j >= p.width)
		return;

	float dot = 0;

	for (int k = 0; k < m.width; ++k)
		dot += m.elements[i * (m.pitch / sizeof(float)) + k] *
			n.elements[k * (n.pitch / sizeof(float)) + j];

	p.elements[i * (p.pitch / sizeof(float)) + j] = dot;
}

void matrix_mul_gpu(const GPUMatrix &m, const GPUMatrix &n, GPUMatrix &p)
{
	// TODO (Task 4): Determine execution configuration and call CUDA kernel

	int minGridSize;
	int suggestedBlockSize;

	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &suggestedBlockSize, matrix_mul_kernel);
	CUDA_CHECK_ERROR;

	int blockSizeX = int(std::ceil(std::sqrt(suggestedBlockSize)));
	int blockSizeY = suggestedBlockSize / blockSizeX;

	int gridSizeX = div_up(p.width, blockSizeX);
	int gridSizeY = div_up(p.height, blockSizeY);

	dim3 gridSize(gridSizeX, gridSizeY);
	dim3 blockSize(blockSizeX, blockSizeY);

	std::cout << "Launching kernel with grid size " <<
		gridSize.x << ' ' << gridSize.y << ' ' << gridSize.z <<
		" and block size " << blockSize.x << ' ' << blockSize.y << ' ' << blockSize.z << std::endl;

	matrix_mul_kernel<<<gridSize, blockSize>>>(m, n, p);
	//cudaDeviceSynchronize();
	CUDA_CHECK_ERROR;

	std::cout << "Kernel done" << std::endl;
}
