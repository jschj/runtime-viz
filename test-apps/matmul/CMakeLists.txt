project(MatMul LANGUAGES CXX CUDA)
cmake_minimum_required(VERSION 3.8)

enable_language(CUDA)

# Set the default build type
#if(NOT CMAKE_BUILD_TYPE)
#	set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
#endif()

# Specify the files that need to be compiled and linked.
# This will create an executable named 'matmul'.
add_executable(matmul
	main.cc
	matmul.cu
	matmul.h
	test.cc
	matrix.cu
	matrix.h
	mul_cpu.cc
	mul_gpu.cu
	mul_gpu.h
)

set_target_properties(matmul PROPERTIES CUDA_ARCHITECTURES "52")

target_link_libraries(matmul memtrack)