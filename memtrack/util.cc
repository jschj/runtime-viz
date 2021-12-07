#include "util.h"


namespace memtrack::util
{
    time_point now()
    {
        return std::chrono::high_resolution_clock::now();
    }

    time_point time_zero()
    {
        return time_point();
    }
} // namespace memtrack::util