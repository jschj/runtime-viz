#include "util.h"


namespace memtrack::util
{
    time_point now()
    {
        return std::chrono::high_resolution_clock::now();
    }
} // namespace memtrack::util