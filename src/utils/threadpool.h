/*
    Extended from code:
    Copyright (c) 2012 Jakob Progsch, Václav Zeman
    Copyright (c) 2017-2019 Gian-Carlo Pascutto and contributors
    Modifications:
    Copyright (c) 2020-2021 Hung Zhe Lin

    This software is provided 'as-is', without any express or implied
    warranty. In no event will the authors be held liable for any damages
    arising from the use of this software.

    Permission is granted to anyone to use this software for any purpose,
    including commercial applications, and to alter it and redistribute it
    freely, subject to the following restrictions:

    1. The origin of this software must not be misrepresented; you must not
    claim that you wrote the original software. If you use this software
    in a product, an acknowledgment in the product documentation would be
    appreciated but is not required.

    2. Altered source versions must be plainly marked as such, and must not be
    misrepresented as being the original software.

    3. This notice may not be removed or altered from any source
    distribution.
*/

# pragma once

#include <atomic>
#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include <iostream>

class ThreadPool {
public:
    ThreadPool(size_t threads);
    ~ThreadPool();

    static ThreadPool& Get(size_t threads=0);

    template<class F, class... Args>
    auto AddTask(F&& f, Args&&... args)
        -> std::future<typename std::result_of<F(Args...)>::type>;
    
    size_t GetNumThreads() const;

private:
    void AddThread(std::function<void()> initializer);
    
    bool IsStopRunning() const;
    std::atomic<bool> stop_running_{false};

    // Number of allocated threads.
    std::atomic<size_t> num_threads_{0};
  
    // Need to keep track of threads so we can join them.
    std::vector<std::thread> workers_;

    // The task queue.
    std::queue<std::function<void(void)>> tasks_;

    std::mutex queue_mutex_;
    
    std::condition_variable cv_;
};

// Get the global thread pool.
inline ThreadPool& ThreadPool::Get(size_t threads) {
    static ThreadPool pool(0);
    while (threads > pool.GetNumThreads()) {
        pool.AddThread([](){});
    }
    while (threads < pool.GetNumThreads() && threads != 0) {
        // TODO: Destory the unused thread.
        break;
    }
    return pool;
}

// The constructor just launches some amount of workers
inline ThreadPool::ThreadPool(size_t threads) {
    stop_running_.store(false, std::memory_order_relaxed);
    for (int t = 0; t < threads ; ++t) {
        AddThread([](){});
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
}

inline void ThreadPool::AddThread(std::function<void()> initializer) {
    num_threads_.fetch_add(1);
    workers_.emplace_back(
        [this, initializer]() -> void {
            initializer();
            while (true) {
                auto task = std::function<void(void)>{};
                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    cv_.wait(lock,
                        [this](){ return IsStopRunning() || !tasks_.empty(); });
                    if (IsStopRunning() && tasks_.empty()) break;
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
                task();
            }
        }
    );
}

inline size_t ThreadPool::GetNumThreads() const {
    return num_threads_.load(std::memory_order_relaxed);
}

inline bool ThreadPool::IsStopRunning() const {
    return stop_running_.load(std::memory_order_relaxed);
}

// Add new work item to the pool.
template<class F, class... Args>
auto ThreadPool::AddTask(F&& f, Args&&... args)
    -> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
    std::future<return_type> res = task->get_future();
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        tasks_.emplace([task](){ (*task)(); });
    }
    cv_.notify_one();
    return res;
}

// The destructor joins all threads.
inline ThreadPool::~ThreadPool()
{
    stop_running_.store(true, std::memory_order_relaxed);
    cv_.notify_all();
    for(auto &worker: workers_) {
        worker.join();
    }
}

template<typename T>
class ThreadGroup {
public:
    ThreadGroup(ThreadPool *pool) {
        pool_ = pool;
    }
    ThreadGroup(ThreadGroup &&group) {
        pool_ = group.pool_;
    }

    template<class F, class... Args>
    void AddTask(F&& f, Args&&... args) {
        tasks_future_.emplace_back(
            pool_->AddTask(std::forward<F>(f), std::forward<Args>(args)...));
    }

    void WaitToJoin(bool dump = false) {
        for (auto &&res : tasks_future_) {
            auto out = res.get();
            if (dump) {
                std::cout << out << std::endl;
            }
        }
    }

private:
    ThreadPool *pool_;
    std::vector<std::future<T>> tasks_future_;
};
