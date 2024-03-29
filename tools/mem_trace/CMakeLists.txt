cmake_minimum_required(VERSION 3.10)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

enable_language(CUDA)

project(NVBITMemTrace)

find_package(NVBit REQUIRED)

add_library(mem_trace_inject_funcs OBJECT inject_funcs.cu)
add_library(mem_trace MODULE mem_trace.cu device_clock.cu)

set_property(TARGET mem_trace PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_source_files_properties(inject_funcs.cu PROPERTIES COMPILE_FLAGS " -maxrregcount=24 -Xptxas -astoolspatch --keep-device-functions -arch=sm_35")

target_link_libraries(mem_trace_inject_funcs NVBit)
target_link_libraries(mem_trace
    NVBit
    cuda
    $<TARGET_OBJECTS:mem_trace_inject_funcs>
    $<TARGET_OBJECTS:meminf_impl>
    $<TARGET_OBJECTS:memtrack>
    #$<TARGET_OBJECTS:digest>
)

target_include_directories(mem_trace PRIVATE
    $<TARGET_PROPERTY:meminf_impl,INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:memtrack,INCLUDE_DIRECTORIES>
    #$<TARGET_PROPERTY:digest,INCLUDE_DIRECTORIES>
)

#set_property(TARGET mem_trace PROPERTY CXX_STANDARD 20)

#target_compile_options(mem_trace PRIVATE -Werror -Wall -Wextra)

#get_target_property(_inc jsoncons INTERFACE_INCLUDE_DIRECTORIES)
#target_include_directories(jsoncons SYSTEM INTERFACE $<TARGET_PROPERTY:jsoncons,INTERFACE_INCLUDE_DIRECTORIES>)

# load global config
#include (${CMAKE_SOURCE_DIR} ../../../build/cmake/config.cmake)

target_link_libraries(mem_trace ${CMAKE_SOURCE_DIR}/libs/zlib/libz.so)
