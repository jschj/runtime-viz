#pragma once

#include "common.h"

#include <shared_mutex>
#include <thread>
#include <vector>
#include <queue>


/*
    This class provides means to temporarily save the memory accesses and dump them out in interoperable format in the end.
    All of this happens in a threadsafe manner and with minimal interferance.
 */
class mem_acc_recorder
{
private:
    mutable std::mutex mut;
    
    // protected by mut
    std::condition_variable queue_not_empty;
    std::queue<mem_access_t> mem_accesses;
    bool is_done = false;
    
    std::string file_path;
    std::thread dump_thread;

    void dump_loop();
public:

    mem_acc_recorder(const std::string& dump_file_path);
    
    void begin();

    void signal_done();
    void record(const mem_access_t& acc);
};