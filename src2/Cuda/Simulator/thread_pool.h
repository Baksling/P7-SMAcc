#pragma once

#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <queue>
#include <mutex>
#include <functional>
#include <thread>

class thread_pool
{
public:
    explicit thread_pool() = default;
    void start();
    void queue_job(const std::function<void()>& job);
    void stop();
    bool is_busy();

private:
    void thread_loop();

    bool should_terminate_ = false;           // Tells threads to stop looking for jobs
    std::mutex queue_mutex_;                  // Prevents data races to the job queue
    std::condition_variable mutex_condition_; // Allows threads to wait on new jobs or termination 
    std::vector<std::thread> threads_;
    std::queue<std::function<void()>> jobs_;
};

#endif