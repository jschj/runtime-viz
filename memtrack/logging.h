#pragma once

#include <vector>
#include <chrono>
#include <mutex>
#include <thread>
#include <atomic>
#include <string>
#include <iostream>
#include <sstream>

namespace memtrack
{
// dump type for all log entries
struct LogEntry {
    unsigned long timestamp;
    void dump();
    virtual std::string serialize() = 0;
};

struct MemAllocEntry : public LogEntry {
    unsigned int device_ptr;
    unsigned int bytesize;
    std::string serialize() override;
};

struct MemAccessEntry : public LogEntry {
    std::string serialize() override;
};

struct KernelLaunchEntry : public LogEntry {
    std::string kernel_name;
    std::string serialize() override;
};

class Logger {
private:
    // indicates whether Logger is still active
    bool still_logging;

    // timepoint of logger initialization
    std::chrono::time_point<std::chrono::steady_clock> start_time;

    // mutex protecting the vector of log entries
    std::mutex mutex;

    // log entries
    std::vector<std::unique_ptr<LogEntry>> entries;

    // current time in milliseconds (since start_time). is updated repeatedly by the metronome
    std::atomic<unsigned long> current_time;

    // thread to update current time
    std::thread metronome;

    // stop flag for metronome thread
    std::atomic_bool request_stop;

    // function to be executed by the metronome thread
    void metronome_fct();

public:
    Logger();
    ~Logger();

    // log a new entry
    void log(std::unique_ptr<LogEntry>&& entry);

    // deactive logger
    void stop();

    // print all log entries to stdout. deactivates logger if necessary.
    void dump_all();
};
} // namespace memtrack
