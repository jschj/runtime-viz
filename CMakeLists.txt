cmake_minimum_required(VERSION 3.10)

project(PMPPNVBIT)

add_subdirectory(${CMAKE_SOURCE_DIR}/meminf)
add_subdirectory(${CMAKE_SOURCE_DIR}/memtrack)
add_subdirectory(${CMAKE_SOURCE_DIR}/tools/mem_trace)
add_subdirectory(${CMAKE_SOURCE_DIR}/test-apps/vectoradd)
add_subdirectory(${CMAKE_SOURCE_DIR}/test-apps/matmul)
add_subdirectory(${CMAKE_SOURCE_DIR}/test-apps/blurring)
