cmake_minimum_required(VERSION 3.10)

enable_language(CUDA)

add_executable(vectoradd vectoradd.cu)

target_link_libraries(vectoradd meminf)
