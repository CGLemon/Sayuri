#include "benchmark.h"
#include "utils/option.h"
#include "utils/splitter.h"
#include "utils/time.h"
#include "utils/threadpool.h"
#include "utils/log.h"

#include <atomic>

void Benchmark::Initialize() {
    const auto query_cnt = IsOptionDefault("benchmark_query") ?
                               0 : GetOptionCount("benchmark_query");

    for (int idx = 0; idx < query_cnt; ++idx) {
        auto query = GetOption<std::string>("benchmark_query", idx);

        if (query.empty()) {
            break;
        }
        for (char &c : query) {
            if (c == ':') {
                c = ' ';
            }
        }
        auto spt = Splitter(query);
        auto maintoken = spt.GetWord(0);

        Query q;
        if (maintoken->Get<>() == "btt" && spt.GetCount() == 4) {
            q.batch_size = spt.GetWord(1)->Get<int>();
            q.threads = spt.GetWord(2)->Get<int>();
            q.timelimit = spt.GetWord(3)->Get<float>();
        }
        quries_list_.emplace_back(q);
    }

    if (quries_list_.empty()) {
        Query q;
        q.batch_size = GetOption<int>("batch_size");
        q.threads = GetOption<int>("threads");
        q.timelimit = 10.0f;
        quries_list_.emplace_back(q);
    }
}

void Benchmark::Run() {
    LOGGING << "Start Benchmark...\n";

    std::atomic<bool> running{false};
    std::atomic<int> count{0};
    const auto Worker = [&, this]() -> void {
        while (running.load(std::memory_order_relaxed)) {
            count.fetch_add(1, std::memory_order_relaxed);
            agent_->GetNetwork().GetOutput(
                agent_->GetState(), Network::kRandom,
                Network::Query::Get().SetCache(false));
        }
    };

    for (auto& q : quries_list_) {
        agent_->SetBatchSize(q.batch_size);
        agent_->SetThreads(q.threads);

        Timer timer;
        timer.Clock();

        auto group = ThreadGroup<void>(&ThreadPool::Get("search", q.threads));
        running.store(true, std::memory_order_relaxed);
        count.store(0, std::memory_order_relaxed);

        for (int i = 0; i < q.threads; ++i) {
            group.AddTask(Worker);
        }
        while (timer.GetDuration() < q.timelimit) {
            std::this_thread::yield();
        }
        running.store(false, std::memory_order_relaxed);
        group.WaitToJoin();
        const auto elapsed = timer.GetDuration();

        LOGGING <<
            Format("Eval Stats - Total: %d evals | Rate: %.2f evals/s | Batch Size: %d | Threads %d\n",
                count.load(std::memory_order_relaxed),
                count.load(std::memory_order_relaxed)/elapsed,
                q.batch_size,
                q.threads);

    }
}
