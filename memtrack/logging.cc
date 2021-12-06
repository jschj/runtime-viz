#include "logging.h"

#include <stdio.h>


namespace memtrack
{
Logger::Logger() : 
    metronome(&Logger::metronome_fct, this),
    entries{},
    mutex{},
    request_stop{false},
    still_logging{true},
    start_time(std::chrono::steady_clock::now()),
    current_time{0} {
}

void Logger::stop() {
    if(this->still_logging) {
        this->request_stop = true;
        this->metronome.join();
        this->still_logging = false;
    }

    
}

Logger::~Logger() {
    if(this->still_logging) {
        this->stop();
    }
}

void Logger::metronome_fct() {
    while(!this->request_stop) {
        // calculate current time
        auto now = std::chrono::steady_clock::now();
        unsigned long duration = (now - this->start_time) / std::chrono::milliseconds(1);
        
        // update atomic variable
        this->current_time.store(duration);
    }
}

void Logger::dump_all() {
    if(this->still_logging) {
        this->stop();
    }

    for (auto& i : this->entries) {
        i->dump();
    }
}

void Logger::log(std::unique_ptr<LogEntry>&& entry) {
    if(this->still_logging) {
        entry->timestamp = this->current_time;
        this->entries.push_back(std::move(entry));
    }
}

void LogEntry::dump() {
    char t[10];
    sprintf(t, "%05lu", this->timestamp);
    std::cout << t << ": " << this->serialize() << std::endl;
}

std::string MemAccessEntry::serialize() {
    std::stringstream ss;
    ss << "Memory Access!";
    return ss.str();
}

std::string MemAllocEntry::serialize() {
    std::stringstream ss;
    ss << "Allocation: ptr = " << this->device_ptr << " bytesize = " << this->bytesize;
    return ss.str();
}

std::string KernelLaunchEntry::serialize() {
    std::stringstream ss;
    ss << "Kernel launch!: name = " << this->kernel_name;
    return ss.str();
}
} // namespace memtrack