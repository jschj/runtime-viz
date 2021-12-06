#pragma once

#include <chrono>


namespace memtrack::util
{
    typedef std::chrono::time_point<std::chrono::high_resolution_clock> time_point;

    time_point now();
}
