#include "neural/batch_forward_pipe.h"
#include "utils/log.h"
#include "utils/format.h"
#include "utils/option.h"

OutputResult BatchForwardPipe::SendQueryAndWait(const InputData& input) {
    OutputResult output;
    InputData reordered_input = input;

    // Reorder the inputs data.
    const int planes_bsize = input.board_size;
    const bool should_reorder = planes_bsize != board_size_per_nn_;

    if (should_reorder) {
        // The input data's board size doesn't match the NN's expected
        // input size. We are reordering the original input data to conform
        // to the NN's input dimensions.
        for (int c = 0; c < weights_->input_channels; ++c) {
            int offset_r = c * board_size_per_nn_ * board_size_per_nn_; // data's ordering index
            int offset_p = c * planes_bsize * planes_bsize; // NN's ordering index

            for (int idx = 0; idx < board_size_per_nn_ * board_size_per_nn_; ++idx) {
                const int x = idx % board_size_per_nn_;
                const int y = idx / board_size_per_nn_;
                if (x < planes_bsize && y < planes_bsize) {
                    reordered_input.planes[offset_r++] = input.planes[offset_p++];
                } else {
                    reordered_input.planes[offset_r++] = 0.f;
                }
            }
        }
    }

    auto entry = std::make_shared<ForwardEntry>(reordered_input, output);
    std::unique_lock<std::mutex> lock(entry->mutex);
    {
        // Push the entry into queue.
        std::lock_guard<std::mutex> queue_lock(queue_mutex_);
        entry_queue_.emplace_back(entry);
    }

    if (static_cast<int>(entry_queue_.size()) >= forwarding_batch_per_nn_) {
        cv_.notify_one(); // Wake up one worker if there are enough batch size.
    }
    entry->cv.wait(lock); // Wait for batch forwarding worker.

    // Reorder the outputs data.
    OutputResult reordered_ouput = output;

    if (should_reorder) {
        // Reorder the NN's outputs data to fit the correct data format.
        int offset_r = 0; // data order index
        int offset_p = 0; // NN order index
        for (int idx = 0; idx < board_size_per_nn_ * board_size_per_nn_; ++idx) {
            const int x = idx % board_size_per_nn_;
            const int y = idx / board_size_per_nn_;
            if (x < planes_bsize && y < planes_bsize) {
                reordered_ouput.probabilities[offset_r] = output.probabilities[offset_p];
                reordered_ouput.ownership[offset_r] = output.ownership[offset_p];
                offset_r++;
                offset_p++;
            } else {
                offset_p++;
            }
        }
    }

    return reordered_ouput;
}

void BatchForwardPipe::SetForwardingSize(int batch_size) {
    forwarding_batch_per_nn_ = batch_size;
}

void BatchForwardPipe::SetBoardSize(int board_size) {
    board_size_per_nn_ = board_size;
}

void BatchForwardPipe::AssignWorkers(int num_gpus) {
    if (!group_) {
        group_ = std::make_unique<ThreadGroup<void>>(&ThreadPool::Get());
    }
    if (num_gpus <= 0) {
        return;
    }
    worker_running_.store(true);
    waittime_.store(GetOption<int>("gpu_waittime"), std::memory_order_relaxed);

    // Allocate threads for BatchForwardPipe.
    ThreadPool::Get("batch-forward-pipe", num_gpus);
    if (group_->FutureEmpty()) {
        for (int gpu = 0; gpu < num_gpus; ++gpu) {
            group_->AddTask([g=gpu, this](){ Worker(g); });
        }
    }
}

void BatchForwardPipe::Worker(int gpu) {
    // Collects a batch of forwarding entries from the queue.
    // The function blocks until:
    //   1. Enough entries are accumulated (>= forwarding_batch_per_nn_), OR
    //   2. A timeout occurs (adaptive wait time), OR
    //   3. The worker is stopped.
    const auto GatherBatches = [this](int gpu_waittime) {
        auto entries = std::vector<std::shared_ptr<ForwardEntry>>{};

        {
            std::unique_lock<std::mutex> lock(worker_mutex_);
            while(true) {
                // If the worker is shutting down, return immediately.
                if (!worker_running_.load(std::memory_order_relaxed)) {
                    return entries;
                }
                // If enough entries are already queued, stop waiting.
                if (static_cast<int>(entry_queue_.size()) >= forwarding_batch_per_nn_) {
                    break;
                }

                // If wait time is enabled, wait with timeout to avoid busy spinning.
                bool timeout = false;
                if (waittime_.load(std::memory_order_relaxed) != 0) {
                    timeout = !cv_.wait_for(
                        lock, std::chrono::milliseconds(waittime_.load(std::memory_order_relaxed)),
                        [this]() {
                            return !worker_running_.load(std::memory_order_relaxed) ||
                                       static_cast<int>(entry_queue_.size()) >= forwarding_batch_per_nn_; });
                }

                if (entry_queue_.empty()) {
                    // Queue is empty. Gradually increase wait time to reduce CPU usage
                    // when no work is available.
                    auto last_waittime = waittime_.fetch_add(1, std::memory_order_relaxed);
                    if (last_waittime >= gpu_waittime) {
                        waittime_.store(gpu_waittime, std::memory_order_relaxed);
                    }
                } else {
                    if (timeout) {
                        // Likely CPU-bound or low traffic, so boost forwarding by disabling
                        // wait time for immediate execution.
                        waittime_.store(0, std::memory_order_relaxed);
                    }
                    break;
                }
            }
        }

        // Return entris.
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            auto count = std::min(static_cast<int>(entry_queue_.size()), forwarding_batch_per_nn_);
            auto end = std::begin(entry_queue_);
            std::advance(end, count);
            std::move(std::begin(entry_queue_), end, std::back_inserter(entries));
            entry_queue_.erase(std::begin(entry_queue_), end);
        }
        return entries;
    };

    const auto gpu_waittime_base = GetOption<int>("gpu_waittime");
    while (true) {
        if (!worker_running_.load(std::memory_order_relaxed)) return;

        auto entries = GatherBatches(gpu_waittime_base);
        const auto batch_size = entries.size();

        // No work available; retry.
        if (batch_size == 0) {
            continue;
        }

        auto inputs = std::vector<InputData>(batch_size);
        for (auto b = size_t{0}; b < batch_size; ++b) {
            inputs[b] = entries[b]->input;
        }
        auto outputs = BatchForward(gpu, inputs);

        // Write back results and notify waiting threads.
        for (auto b = size_t{0}; b < batch_size; ++b) {
            entries[b]->output = outputs[b];
            {
                // Locking guarantees the waiting thread sees the output.
                std::unique_lock<std::mutex> lk(entries[b]->mutex);
            }
            // Wake up all threads waiting on this entry.
            entries[b]->cv.notify_all();
        }
    }
}

void BatchForwardPipe::QuitWorkers() {
    worker_running_.store(false);
    cv_.notify_all();
    group_->WaitToJoin();
}
