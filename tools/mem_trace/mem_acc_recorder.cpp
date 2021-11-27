#include "mem_acc_recorder.h"

#include <fstream>



void mem_acc_recorder::dump_loop()
{
    std::ofstream of(file_path, std::ios_base::out);

    if (!of.is_open())
        throw std::runtime_error("Could not open dump file!");

    // some hacky JSON for now

    /*
        {
            "accesses": [
                {"locations": [...], "when": ... },
                ...
            ]
        }
    
     */

    of << "{\"accesses\":["

    while (true) {
        std::unique_lock<std::mutex> lk(mut);
        queue_not_empty.wait(lk, [this](){ return !mem_accesses.empty() || is_done; });

        while (!mem_accesses.empty()) {
            mem_access_t acc = mem_accesses.front();
            mem_accesses.pop();

            of << "{"
                << "\"locations\": [";

            for (uint64_t addr : acc.addresses)
                of << addr << ", ";
            
            of << "], "
                << "\"when\": " << acc.when
                << "}";
        }

        if (is_done)
            break;
    }

    of << "]}";
}

mem_acc_recorder::mem_acc_recorder(const std::string& dump_file_path):
    file_path(dump_file_path)
{
    
}

void mem_acc_recorder::begin()
{
    dump_thread = std::thread(mem_acc_recorder::dump_loop, this, dump_file_path);
}

void mem_acc_recorder::signal_done()
{
    {
        std::unique_lock<std::mutex> lk(mut);
        is_done = true;
    }

    queue_not_empty.notify_one();
    // BUG: Currently multiple joins possible
    dump_thread.join();
}

void mem_acc_recorder::record(const mem_access_t& acc)
{
    std::unique_lock<std::mutex> lk(mut);
    mem_accesses.push(acc);
    queue_not_empty.notify_one();
}
