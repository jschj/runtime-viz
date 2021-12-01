cmake_minimum_required(VERSION 3.10)

add_library(memtrack SHARED
    malloc_track.cc
)

# need CUDA headers
target_include_directories(memtrack PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../core
    ${CMAKE_CURRENT_SOURCE_DIR}
)

#set_target_properties(memtrack PROPERTIES
#    PUBLIC_HEADER memtrack.h
#)
