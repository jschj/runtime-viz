#include <iostream>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#include <memtrack/memtrack.h>

#include "matmul.h"
#include "test.h"
#include "common.h"
#include "mul_cpu.h"
#include "mul_gpu.h"
#include "timer.h"

void print_cuda_devices()
{
	// TODO: Task 2

	/*
		- Compute capability
		- Multiprocessor count
		- GPU clock rate in GHz
		- Total global memory in MiB
		- L2 cache size in KiB
	 */

	int device_count;

	cudaGetDeviceCount(&device_count);
	CUDA_CHECK_ERROR;

	for (int i = 0; i < device_count; ++i) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		CUDA_CHECK_ERROR;

		std::cout <<
			"================ Device #" << i << " ================\n" <<
			"Compute capability: " << prop.major << '.' << prop.minor << '\n' <<
			"Multiprocessor count: " << prop.multiProcessorCount << '\n' <<
			"GPU clock rate: " << prop.clockRate / 1000000.0 << " GHz\n" <<
			"Total global memory: " << prop.totalGlobalMem / double(1024 * 1024) << " MiB\n" <<
			"L2 cache size: " << prop.l2CacheSize / double(1024) << " KiB" << std::endl;

	}

	std::cout << "CUDA device count: " << device_count << std::endl;
}

void print_matrix(const CPUMatrix& mat)
{
	for (int i = 0; i < mat.height; ++i) {
		for (int j = 0; j < mat.width; ++j)
			std::cout << mat.elements[i * mat.width + j] << " ";

		std::cout << '\n';
	}
}

void matmul()
{
	// === Task 3 ===
	// TODO: Allocate CPU matrices (see matrix.cc)
	//       Matrix sizes:
	//       Input matrices:
	//       Matrix M: pmpp::M_WIDTH, pmpp::M_HEIGHT
	//       Matrix N: pmpp::N_WIDTH, pmpp::N_HEIGHT
	//       Output matrices:
	//       Matrix P: pmpp::P_WIDTH, pmpp::P_HEIGHT
	CPUMatrix M = matrix_alloc_cpu(pmpp::M_WIDTH, pmpp::M_HEIGHT);
	CPUMatrix N = matrix_alloc_cpu(pmpp::N_WIDTH, pmpp::N_HEIGHT);
	CPUMatrix P = matrix_alloc_cpu(pmpp::P_WIDTH, pmpp::P_HEIGHT);
	
	// TODO: Fill the CPU input matrices with the provided test values (pmpp::fill(CPUMatrix &m, CPUMatrix &n))
	pmpp::fill(M, N);

	// TODO (Task 5): Start CPU timing here!
	auto t1 = timer_now();

	// TODO: Run your implementation on the CPU (see mul_cpu.cc)
	matrix_mul_cpu(M, N, P);

	// TODO (Task 5): Stop CPU timing here!
	auto t2 = timer_now();
	double t_delta = timer_elapsed(t1, t2);
	
	std::cout << "CPU matrix multiplication of size " <<
		pmpp::M_HEIGHT << 'x' << pmpp::M_WIDTH << "   " <<
		pmpp::N_HEIGHT << 'x' << pmpp::N_WIDTH << "   " <<
		" took " << t_delta << "ms" << std::endl;

	// TODO: Check your matrix for correctness (pmpp::test_cpu(const CPUMatrix &p))
	pmpp::test_cpu(P);

	// === Task 4 ===
	// TODO: Set CUDA device
	cudaSetDevice(0);
	CUDA_CHECK_ERROR;

	// TODO: Allocate GPU matrices (see matrix.cc)
	GPUMatrix GM = matrix_alloc_gpu(pmpp::M_WIDTH, pmpp::M_HEIGHT);
	GPUMatrix GN = matrix_alloc_gpu(pmpp::N_WIDTH, pmpp::N_HEIGHT);
	GPUMatrix GP = matrix_alloc_gpu(pmpp::P_WIDTH, pmpp::P_HEIGHT);
	CPUMatrix CP = matrix_alloc_cpu(pmpp::P_WIDTH, pmpp::P_HEIGHT);

	TRACK_BUFFER(GM.elements, "GM");
	TRACK_BUFFER(GN.elements, "GN");
	TRACK_BUFFER(GP.elements, "GP");

	// TODO: Upload the CPU input matrices to the GPU (see matrix.cc)
	matrix_upload(M, GM);
	matrix_upload(N, GN);

	// TODO (Task 5): Start GPU timing here!
	cudaEvent_t evStart, evStop;
	cudaEventCreate(&evStart);
	cudaEventCreate(&evStop);

	// TODO: Run your implementation on the GPU (see mul_gpu.cu)
	cudaEventRecord(evStart);
	matrix_mul_gpu(GM, GN, GP);
	cudaEventRecord(evStop);
	cudaEventSynchronize(evStop);

	// TODO (Task 5): Stop GPU timing here!
	// TODO: Download the GPU output matrix to the CPU (see matrix.cc)
	matrix_download(GP, CP);

	float gpuMs;
	cudaEventElapsedTime(&gpuMs, evStart, evStop);
	
	std::cout << "GPU matrix multiplication of size " <<
		pmpp::M_HEIGHT << 'x' << pmpp::M_WIDTH << "   " <<
		pmpp::N_HEIGHT << 'x' << pmpp::N_WIDTH << "   " <<
		" took " << gpuMs << "ms" << std::endl;

	// TODO: Check your downloaded matrix for correctness (pmpp::test_gpu(const CPUMatrix &p))
	pmpp::test_gpu(CP);

	// TODO: Compare CPU result with GPU result (see matrix.cc)
	matrix_compare_cpu(CP, P);

	// TODO (Task3/4/5): Cleanup ALL matrices and and events
	matrix_free_gpu(GM);
	matrix_free_gpu(GN);
	matrix_free_gpu(GP);
	matrix_free_cpu(CP);

	matrix_free_cpu(M);
	matrix_free_cpu(N);
	matrix_free_cpu(P);
}


/************************************************************
 * 
 * TODO: Write your text answers here!
 * 
 * (Task 4) 6. Where do the differences come from?
 * 
 * Answer: NVCC applies a whole bunch of optimizations for floating point computations. One of them is to combine multiply and add
 * in the dot product computation (FMA). FMA can produce different results than consecutive MUL and ADD. To verify this one can compile
 * with -fmad=false to disable FMA instructions and thus receive the same results as on the CPU.
 * 
 * 
 ************************************************************/
