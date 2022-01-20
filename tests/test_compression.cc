#include <access_compression.h>


#define SAMPLE_COUNT 0x10000000


using namespace memtrack;


void deflate_bytes(const uint8_t *in_bytes, size_t in_size,
    uint8_t *out_bytes, size_t out_size)
{
    
}

bool test_compression()
{
    std::vector<access_compression::raw_buffer_access> accesses;

    // generate accesses
    for (uint64_t i = 0; i < SAMPLE_COUNT; ++i) {
        access_compression::raw_buffer_access acc {
            .buffer_id = rand(),
            .time_point = rand(),
            .index = rand()
        };

        accesses.push_back(acc);
    }


    return false;
}
