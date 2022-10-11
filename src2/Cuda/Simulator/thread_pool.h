#pragma once

#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <queue>
#include <mutex>
#include <functional>
#include <thread>
#include <condition_variable>

class thread_pool
{
public:
    //Will check hardware capable concurrency on negative 
    explicit thread_pool(const unsigned int max_concurrency = 0)
    {
        this->max_concurrency_ = max_concurrency;
    }
    void start();
    void queue_job(const std::function<void()>& job);
    void stop();
    bool is_busy();

private:
    void thread_loop();
    
    unsigned int max_concurrency_;
    bool should_terminate_ = false;           // Tells threads to stop looking for jobs
    std::mutex queue_mutex_;                  // Prevents data races to the job queue
    std::condition_variable mutex_condition_; // Allows threads to wait on new jobs or termination 
    std::vector<std::thread> threads_;
    std::queue<std::function<void()>> jobs_;
};

#endif