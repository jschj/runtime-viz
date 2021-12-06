#include "cu_memtrack.h"

#include <memory>
#include "json.h"


namespace memtrack
{
    std::unique_ptr<streaming_bson_encoder> bson_encoder;



    void cu_memtrack_init(const std::string& json_dump_file)
    {
        bson_encoder = std::make_unique<streaming_bson_encoder>(json_dump_file);
    }

    void cu_memtrack_begin()
    {
        bson_encoder->begin();
    }

    void cu_memtrack_malloc(nvbit_api_cuda_t cbid, void *params)
    {
        tracker().on_malloc(cbid, params);
    }

    void cu_memtrack_free(nvbit_api_cuda_t cbid, void *params)
    {
        tracker().on_free(cbid, params);
    }

    void cu_memtrack_access()
    {

    }

    void cu_memtrack_end()
    {
        bson_encoder->end();
    }

}