cmake_minimum_required(VERSION 3.10)

add_library(memtrack SHARED
    malloc_track.cc
    json.cc
    cu_memtrack.cc
    util.cc
)

# need CUDA headers
target_include_directories(memtrack PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../core
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
    # header only library
    "${CMAKE_SOURCE_DIR}/jsoncons/include/"
)

#set_target_properties(memtrack PROPERTIES
#    PUBLIC_HEADER memtrack.h
#)
