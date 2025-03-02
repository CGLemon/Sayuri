/*
    Extended from code:
    Copyright (c) 2012 Jakob Progsch, Václav Zeman
    Copyright (c) 2017-2019 Gian-Carlo Pascutto and contributors
    Modifications:
    Copyright (c) 2020-2025 Hung Tse Lin

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
#include <algorithm>
#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include <string>
#include <sstream>
#include <unordered_map>
#include <iomanip>
#include <iostream>

template<typename K>
class ThreadPoolItem {
public:
    ThreadPoolItem() = default;

    size_t GetNumThreads() const {
        return num_threads_;
    }
    size_t GetNumThreads(const K key) const {
        if (key_to_idx_.count(key) == 0) {
            return 0;
        }
        size_t idx = key_to_idx_.at(key);
        return num_threads_per_item_.at(idx);
    }
    size_t GetNoItemNumThreads() const {
        return num_threads_for_no_item_;
    }

    size_t AddNumThreads(size_t t) {
        num_threads_for_no_item_ += t;
        num_threads_ += t;
        return t;
    }
    size_t AddNumThreads(const K key, size_t t) {
        if (key_to_idx_.count(key) == 0) {
            key_to_idx_[key] = num_threads_per_item_.size();
            num_threads_per_item_.emplace_back(0);
        }
        size_t item_threads = t;
        size_t addition_threads = t;

        if (num_threads_for_no_item_ > 0) {
            // assign no item threads for new item
            if (addition_threads >= num_threads_for_no_item_) {
                addition_threads -= num_threads_for_no_item_;
                num_threads_for_no_item_ = 0;
            } else {
                num_threads_for_no_item_ -= addition_threads;
                addition_threads = 0;
            }
        }
        size_t idx = key_to_idx_[key];
        num_threads_per_item_[idx] += item_threads;
        num_threads_ += addition_threads;
        return addition_threads;
    }

    std::vector<K> GetKeys() const {
        auto keys = std::vector<K>{};
        for (auto &it : key_to_idx_) {
            keys.emplace_back(it.first);
        }
        return keys;
    }

private:
    size_t num_threads_{0};
    size_t num_threads_for_no_item_{0};
    std::unordered_map<K, size_t> key_to_idx_;
    std::vector<size_t> num_threads_per_item_;
};

class ThreadPool {
public:
    ThreadPool(size_t threads) {
        stop_running_.store(false);
        for (auto t = size_t{0}; t < threads ; ++t) {
            AddThread([](){});
        }
        // Wait some milliseconds until all the threads are constructed.
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    ~ThreadPool() {
        stop_running_.store(true);
        cv_.notify_all();
        for(auto &worker: workers_) {
            worker.join();
        }
        workers_.clear();
    }

    // Get the global thread pool.
    static ThreadPool& Get(size_t threads=0) {
        auto &pool = GetInternal();
        while (threads > pool.GetNumThreads()) {
            pool.AddThread([](){});
        }
        return pool;
    }
    static ThreadPool& Get(std::string key, size_t threads=0) {
        auto &pool = GetInternal();
        while (threads > pool.GetNumThreads(key)) {
            pool.AddThread(key, [](){});
        }
        return pool;
    }

    // Add the task function to waiting queue.
    template<class F, class... Args>
    auto AddTask(F&& f, Args&&... args)
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

    size_t GetNumThreads() const {
        return num_threads_per_item_.GetNumThreads();
    }
    size_t GetNumThreads(std::string key) const {
        return num_threads_per_item_.GetNumThreads(key);
    }

    std::string ToString() const {
        auto oss = std::ostringstream{};
        auto keys = num_threads_per_item_.GetKeys();
        size_t max_keysize = std::max_element(std::begin(keys), std::end(keys),
                                 [](std::string &a, std::string &b) {
                                     return a.size() < b.size();
                                 })->size();
        size_t wsize = max_keysize + 2;

        oss << "totally threads= " << GetNumThreads() << std::endl;
        for (auto &key: keys) {
            oss << std::setw(wsize) << key
                    << std::setw(4) << GetNumThreads(key)
                    << std::endl;
        }
        oss << std::setw(wsize) << "others"
                << std::setw(4) << num_threads_per_item_.GetNoItemNumThreads()
                << std::endl;
        return oss.str();
    }

private:
    static ThreadPool& GetInternal() {
        static ThreadPool pool(0);
        return pool;
    }
    void AddWorker(std::function<void()> initializer) {
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
    void AddThread(std::function<void()> initializer) {
        num_threads_per_item_.AddNumThreads(1);
        AddWorker(initializer);
    }
    void AddThread(std::string key, std::function<void()> initializer) {
        size_t t = num_threads_per_item_.AddNumThreads(key, 1);
        if (t > 0) {
            AddWorker(initializer);
        }
    }

    bool IsStopRunning() const {
        return stop_running_.load();
    }
    std::atomic<bool> stop_running_{false};

    // Number of allocated threads.
    ThreadPoolItem<std::string> num_threads_per_item_;

    // Need to keep track of threads so we can join them.
    std::vector<std::thread> workers_;

    // The task queue.
    std::queue<std::function<void(void)>> tasks_;

    std::mutex queue_mutex_;

    std::condition_variable cv_;
};


template<typename T>
class ThreadGroup {
public:
    ThreadGroup(ThreadPool *pool) {
        pool_ = pool;
    }
    ThreadGroup(ThreadGroup &&group) {
        pool_ = group.pool_;
    }
    ~ThreadGroup() {
        WaitToJoin();
    }

    template<class F, class... Args>
    void AddTask(F&& f, Args&&... args) {
        tasks_future_.emplace_back(
            pool_->AddTask(std::forward<F>(f), std::forward<Args>(args)...));
    }

    void WaitToJoin() {
        for (auto &&res : tasks_future_) {
            res.get();
        }
        tasks_future_.clear();
    }

    bool FutureEmpty() {
        return tasks_future_.empty();
    }

private:
    ThreadPool *pool_;
    std::vector<std::future<T>> tasks_future_;
};
