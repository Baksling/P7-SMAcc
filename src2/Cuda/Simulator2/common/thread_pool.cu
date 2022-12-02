#include "thread_pool.h"

thread_pool::thread_pool(const unsigned max_concurrency)
{
    this->max_concurrency_ = max_concurrency;
}

void thread_pool::await_run()
{
    this->start();
    while(this->is_busy())
    {
        std::this_thread::yield();
    }
    this->stop();
}

void thread_pool::start()
{
    const unsigned int hardware_max = std::thread::hardware_concurrency();
    // if 0 threads supplied, default to hardware default
    this->max_concurrency_ = this->max_concurrency_ == 0 ? hardware_max : this->max_concurrency_;

    //pick lowest between user parameter and supported concurrency.
    const unsigned int num_threads = this->max_concurrency_ < hardware_max ? this->max_concurrency_ : hardware_max;

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
