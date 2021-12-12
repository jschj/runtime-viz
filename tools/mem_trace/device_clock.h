#pragma once


#include <stdint.h>
#include <nvbit.h>
#include <cuda.h>

#include <memtrack/util.h>

/*
    This functions returns an estimate of how many nanoseconds the device global time is in the
    future versus the host global time. This result can be negative. This function will return
    wrong results in the year 2269.
 */
extern "C" int64_t probe_global_time_difference();
