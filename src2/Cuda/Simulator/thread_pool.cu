#include "thread_pool.h"

void thread_pool::start()
{
    const uint32_t num_threads = std::thread::hardware_concurrency(); // Max # of threads the system supports
    threads_.resize(num_threads);
    for (uint32_t i = 0; i < num_threads; i++) {
        threads_.at(i) = std::thread([this]() {this->thread_loop();});
    }
}

void thread_pool::queue_job(const std::function<void()>& job)
{
    {
        std::unique_lock<std::mutex> lock(this->queue_mutex_);
        this->jobs_.push(job);
    }
    this->mutex_condition_.notify_one();
}

void thread_pool::stop()
{
    {
        std::unique_lock<std::mutex> lock(this->queue_mutex_);
        this->should_terminate_ = true;
    }
    this->mutex_condition_.notify_all();
    for (std::thread& active_thread : this->threads_) {
        active_thread.join();
    }
    this->threads_.clear();
}

bool thread_pool::is_busy()
{
    bool pool_busy;
    {
        std::unique_lock<std::mutex> lock(this->queue_mutex_);
        pool_busy = !this->jobs_.empty();
    }
    return pool_busy;
}

void thread_pool::thread_loop()
{
    while (true) {
        std::function<void()> job;
        {
            std::unique_lock<std::mutex> lock(this->queue_mutex_);
            mutex_condition_.wait(lock, [this] {
                return !this->jobs_.empty() || this->should_terminate_;
            });
            if (this->should_terminate_) {
                return;
            }
            job = this->jobs_.front();
            this->jobs_.pop();
        }
        job();
    }
}
