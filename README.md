# Runtime-Viz

With _Runtime-Viz_ you can record memory accesses to global memory in your _CUDA_ kernel and create an interactive visualization of the data.

[NVBit](https://github.com/NVlabs/NVBit) is NVidias binary instrumentation tool. It works by dynamically loading a shared library with the `LD_PRELOAD` trick. Then any _CUDA_ API call can be intercepted and custom code can be run. This project uses this functionality to intercept kernel launches and replace the memory access instructions in the kernel code with jumps to a method which saves the access addresses and sends them to the CPU. The addresses get mapped to buffers which must be manually tracked beforehand by calling `TRACK_BUFFER<T>(T *buf, const char *name)`.

## Usage

```
[TOOL_VERBOSE=1] [ONLY_WRITES=1] LD_PRELOAD=path_to_libmem_trace.so ./program_to_execute [program arguments]
```

Options:
- `TOOL_VERBOSE=1` prints all kind of information
- `ONLY_WRITES=1` only records memory accesses that perform a write operation

Executing this will generate a bunch of `*.accesses.bin` files and a `buffers.json` file. `*.accesses.bin` contain the raw access data for each kernel. Calling the same kernel multiple times overwrites the old file. So this should be avoided. `buffers.json` contains all the buffer information. An interactive visualization can then be made by calling `python visualization/main.py buffers.json`. Note that the visualization tool accumulates all buffer accesses. It can be configured to only show certain kernels. Use `python visualization/main.py --help` for more information.

## Compiling

Dependencies for the _Runtime-Viz_ are [zlib](https://github.com/madler/zlib.git) and [jsoncons](https://github.com/danielaparker/jsoncons.git). Both are included as submodules. Of course you will also need the _CUDA_ compiler and libraries. _CMake_ is used as the build system. For visualization you will need Python 3.x along with the python packages _numpy_ and _matplotlib_. They can be installed with `pip install -r visualization/requirements.txt`.

First make sure that the submodules _jsoncons_ and _zlib_ are pulled. Then build _zlib_ by executing `make zlib` in the root directory. Next build _memtrack_ and _mem_trace_ by executing:

```
make build
cd build
cmake ..
make
```

This will generate `libmem_trace.so` and all programs in `test-apps`.

## Integrating a Custom Project

1. Create a directory in `test-apps` for your application.
2. In the accompanying `CMakeLists.txt` add the line `target_link_libraries(name_to_target memtrack)`.
3. In the root directory `CMakeLists.txt` add the line `add_subdirectory(${CMAKE_SOURCE_DIR}/test-apps/name_of_app)`.

Then `cd` into the `build` directory and excute:
```
cmake ..
make
```

This should create an executable in `build/test-apps/name_of_app`.
