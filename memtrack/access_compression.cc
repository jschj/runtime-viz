#include "access_compression.h"


namespace memtrack
{

access_compression::access_compression(size_t raw_accesses_max, size_t compressed_buffer_size, const std::string& target_file_path) :
    raw_accesses_max(raw_accesses_max),
    compressed_buffer(compressed_buffer_size),
    compressed_buffer_index(0),
    target_file(target_file_path, std::ios::binary)
{
    if (!target_file.is_open())
        throw std::runtime_error("Could not open dump file :(!");

    stream.zalloc = [](voidpf opaque, uInt items, uInt size) -> voidpf {
        return std::malloc(items * size);
    };

    stream.zfree = [](voidpf opaque, voidpf address) -> void {
        std::free(address);
    };

    stream.opaque = nullptr;

    if (deflateInit(&stream, Z_DEFAULT_COMPRESSION) != Z_OK)
        throw std::runtime_error("Could not init zlib deflation!");

    stream.next_in = reinterpret_cast<z_const Bytef *>(raw_accesses.data());
    stream.avail_in = 0;
    stream.total_in = 0;

    stream.next_out = compressed_buffer.data();
    stream.avail_out = compressed_buffer.size() * sizeof(*compressed_buffer.data());
    stream.total_out = 0;
}

access_compression::~access_compression()
{
    flush();
    int result = deflateEnd(&stream);

    if (result != Z_OK)
        std::cerr << "Error when ending deflation!" << std::endl;
}

void access_compression::set_input_stream()
{
    stream.next_in = reinterpret_cast<z_const Bytef *>(raw_accesses.data());
    stream.avail_in = raw_accesses.size() * sizeof(*raw_accesses.data());
    stream.total_in = 0;
}

void access_compression::set_output_stream()
{
    stream.next_out = compressed_buffer.data();
    stream.avail_out = compressed_buffer.size() * sizeof(*compressed_buffer.data());
    stream.total_out = 0;
}

void access_compression::write_to_file(size_t len)
{
    target_file.write(reinterpret_cast<const char *>(compressed_buffer.data()), len);
}

void access_compression::compress_accesses(bool final)
{
    static auto throw_error = [](const std::string& text) { throw std::runtime_error(text); };

    if (final) {
        int result = deflate(&stream, Z_FINISH);

        if (result != Z_STREAM_END)
            throw_error("deflate() failed for final call!");

        std::cout << "processed " << stream.total_in << " input bytes!" << std::endl;
        std::cout << "writing final " << stream.total_out << " bytes to file!" << std::endl;
        write_to_file(stream.total_out);
    } else {
        int result = deflate(&stream, Z_NO_FLUSH);

        if (result == Z_BUF_ERROR) {
            if (stream.avail_out == 0) {
                std::cout << "writing " << stream.total_out << " bytes to file!" << std::endl;
                write_to_file(stream.total_out);
                // reset output buffer
                stream.next_out = reinterpret_cast<Bytef *>(compressed_buffer.data());
                stream.avail_out = compressed_buffer.size() * sizeof(*compressed_buffer.data());
                stream.total_out = 0;
            } else {
                throw_error("deflate() returned Z_BUF_ERROR buf output buffer is not full!");    
            }
        } else if (result != Z_OK) {
            throw_error("deflate() failed!");
        }
    }
}

void access_compression::track_access(const raw_buffer_access& access)
{
    raw_accesses.push_back(access);

    if (raw_accesses.size() == raw_accesses_max) {
        stream.next_in = reinterpret_cast<Bytef *>(raw_accesses.data());
        stream.avail_in = raw_accesses.size() * sizeof(*raw_accesses.data());
        compress_accesses(false);
        raw_accesses.resize(0);
    }
}

void access_compression::flush()
{
    stream.next_in = reinterpret_cast<Bytef *>(raw_accesses.data());
    stream.avail_in = raw_accesses.size() * sizeof(*raw_accesses.data());
    compress_accesses(true);
}

}