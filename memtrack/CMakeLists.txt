cmake_minimum_required(VERSION 3.10)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")


add_library(memtrack SHARED
    malloc_track.cc
    json.cc
    cu_memtrack.cc
    util.cc
)

target_compile_features(memtrack PRIVATE cxx_std_17)

# need CUDA headers
target_include_directories(memtrack PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../core
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
    # header only library
    "${CMAKE_SOURCE_DIR}/jsoncons/include/"
    ${CMAKE_SOURCE_DIR}/libs/zlib/
)

#set_target_properties(memtrack PROPERTIES
#    PUBLIC_HEADER memtrack.h
#)

target_link_libraries(memtrack ${CMAKE_SOURCE_DIR}/libs/zlib/libz.so)
