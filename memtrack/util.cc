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

    uint64_t time_to_ns(const time_point& t)
    {
        return time_to_ns(t, time_zero());
    }

    uint64_t time_to_ns(const time_point& before, const time_point& after)
    {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(after - before).count();
    }

} // namespace memtrack::util