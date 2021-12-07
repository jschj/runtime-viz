#pragma once

#include <chrono>
#include <x86intrin.h>


namespace memtrack::util
{
    typedef std::chrono::time_point<std::chrono::high_resolution_clock> time_point;

    time_point now();
    // can effectively never happen
    time_point time_zero();

    /*
    class low_overhead_clock
    {
    public:
        typedef std::chrono::time_point<low_overhead_clock> time_point;

        static time_point now() noexcept
        {
            //uint64_t t = __rdtsc();


        }
    };
     */
}
