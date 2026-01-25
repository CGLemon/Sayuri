#pragma once

#include <atomic>
#include <condition_variable>
#include <list>
#include <memory>
#include <mutex>
#include <vector>

#include "neural/network_basic.h"
#include "utils/threadpool.h"

class BatchForwardPipe : public NetworkForwardPipe {
public:
    virtual void Initialize(std::shared_ptr<DNNWeights> weights) = 0;

    virtual OutputResult Forward(const InputData &input) = 0;

    virtual bool Valid() const = 0;

    virtual void Construct(ForwardPipeOption option, std::shared_ptr<DNNWeights> weights) = 0;

    virtual void Release() = 0;

    virtual void Destroy() = 0;

    virtual std::vector<OutputResult> BatchForward(int gpu, const std::vector<InputData>& inputs) = 0;

protected:
    OutputResult SendQueryAndWait(const InputData& input);
    void SetForwardingSize(int batch_size);
    void SetBoardSize(int board_size);
    void AssignWorkers(int num_gpus);
    void QuitWorkers();

private:
    struct ForwawrdEntry {
	    const InputData& input;
        OutputResult& output;

        std::condition_variable cv;
        std::mutex mutex;

        ForwawrdEntry(const InputData &in,
                      OutputResult &out) :
                      input(in), output(out) {}
    };

    void Worker(int gpu);

    std::list<std::shared_ptr<ForwawrdEntry>> entry_queue_;
    std::mutex worker_mutex_;
    std::mutex queue_mutex_;

    std::condition_variable cv_;
    std::unique_ptr<ThreadGroup<void>> group_{nullptr};

    std::atomic<int> waittime_{0};
    std::atomic<bool> worker_running_;

    int forwarding_batch_per_nn_{0};
    int board_size_per_nn_{0};
};
