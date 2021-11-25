cmake_minimum_required(VERSION 3.10)

add_library(meminf SHARED meminf_default.cc)
add_library(meminf_impl SHARED meminf_impl.cc)
add_library(malloc_trace SHARED malloc_trace.cc)

target_include_directories(meminf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(meminf_impl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_include_directories(malloc_trace PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../core)
