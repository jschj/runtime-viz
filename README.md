# Runtime-Viz

With _Runtime-Viz_ you can record memory accesses to global memory in your _CUDA_ kernel and create an interactive visualization of the data.

[NVBit](https://github.com/NVlabs/NVBit) is NVidias binary instrumentation tool. It works by dynamically loading a shared library with the `LD_PRELOAD` trick. Then any _CUDA_ API call can be intercepted and custom code can be run. This project uses this functionality to intercept kernel launches and replace the memory access instructions in the kernel code with jumps to a method which saves the access addresses and sends them to the CPU. The addresses get mapped to buffers which must be manually tracked beforehand by calling `TRACK_BUFFER(T *buf, const char *name)`.

## Usage

```
[TOOL_VERBOSE=1] [ONLY_WRITES=1] LD_PRELOAD=path_to_libmem_trace.so ./program_to_execute
```

Options:
- `TOOL_VERBOSE=1` prints all kind of information
- `ONLY_WRITES=1` only records memory accesses that perform a write operation

## Compiling

First make sure that the submodules _jsoncons_ and _zlib_ are pulled. The build _zlib_ by executing `make zlib` in the root directory. Next build _memtrack_ and _mem_trace_ by executing:

```
make build
cd build
cmake ..
make
```

This will generate `libmem_trace.so`.

## A Working Example

