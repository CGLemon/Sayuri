#include "benchmark.h"
#include "utils/option.h"
#include "utils/splitter.h"
#include "utils/time.h"
#include "utils/threadpool.h"
#include "utils/log.h"
#include "utils/random.h"
#include "game/sgf.h"

#include <atomic>
#include <cmath>

double ComputeEloEffect(int playouts_per_move, double playouts_per_second, int threads) {
    // From: https://github.com/lightvector/KataGo/blob/9030f72d152da42c1dd03590aa5116993ea842f6/cpp/program/playutils.cpp#L850
    auto ComputeEloCost = [&](int p, int t) {
        // Completely ad-hoc formula that approximately fits noisy tests. Probably not very good
        // but then again the recommendation of this benchmark program is very rough anyways, it
        // doesn't need to be all that great.]
        return t * 7.0 * std::pow(1600.0 / (800.0 + p), 0.85);
    };
    // From some test matches by lightvector using g170
    static constexpr double kEloGainPerDoubling = 250;
    double gain = kEloGainPerDoubling * std::log(playouts_per_second) / std::log(2);
    double cost = ComputeEloCost(playouts_per_move, threads);
    return gain - cost;
}

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

        if (maintoken->Get<>() == "tbg" && spt.GetCount() == 4) {
            // threads-batchsize-games
            // "btg:24:12:10"
            //
            // threads = 24
            // batch size = 12
            // test performance over 10 games
            Query q;
            q.threads = spt.GetWord(1)->Get<int>();
            q.batch_size = spt.GetWord(2)->Get<int>();
            q.games = spt.GetWord(3)->Get<int>();
            queries_list_.emplace_back(q);
        }
    }
    auto ite = std::remove_if(std::begin(queries_list_), std::end(queries_list_),
                              [](Query &q) {
                                  if (q.threads <= 0 ||
                                          q.batch_size <= 0 ||
                                          q.games <= 0) {
                                      return true;
                                  }
                                  return false;
                              });
    queries_list_.erase(ite, std::end(queries_list_));

    if (queries_list_.empty()) {
        Query q;
        q.threads = GetOption<int>("threads");
        q.batch_size = GetOption<int>("batch_size");
        q.games = 10;
        queries_list_.emplace_back(q);
    }
}

std::vector<std::string> Benchmark::GenerateTestSet(int num_games) {
    auto out = std::vector<std::string>{};

    for (int n = 0; n < num_games; ++n) {
        agent_->GetSearch().ReleaseTree();
        agent_->GetNetwork().ClearCache();
        agent_->GetState().ClearBoard();

        auto dist = std::normal_distribution<float>(
            0.f, (float)agent_->GetState().GetBoardSize()/4);
        const int random_moves_cnt =
            dist(Random<>::Get()) + 0.08f * agent_->GetState().GetNumIntersections();

        for (int i = 0; i < random_moves_cnt; ++i) {
            if (agent_->GetState().GetPasses() >= 2) {
                break;
            }

            int random_move = agent_->GetNetwork().
                GetVertexWithPolicy(agent_->GetState(), 0.95, false);
            agent_->GetState().PlayMove(random_move);
        }
        out.emplace_back(Sgf::Get().ToString(agent_->GetState()));
    }
    agent_->GetSearch().ReleaseTree();
    agent_->GetNetwork().ClearCache();
    agent_->GetState().ClearBoard();

    return out;
}

void Benchmark::Run() {
    LOGGING << "Prepare benchmark set...\n";
    int max_games = 0;
    for (auto& q : queries_list_) {
        max_games = std::max(q.games, max_games);
    }
    auto sgf_list = GenerateTestSet(max_games);

    LOGGING << "Start Benchmark...\n";
    Parameters * param = agent_->GetSearch().GetParams();

    auto elo_list = std::vector<double>{};
    for (int i = 0; i < (int)queries_list_.size(); ++i) {
        auto& q = queries_list_[i];
        LOGGING << Format(
            "query %d/%d -> threads= %d, batch size= %d\n",
             i+1, (int)queries_list_.size(), q.threads, q.batch_size);
        agent_->SetThreads(q.threads);
        agent_->SetBatchSize(q.batch_size);

        double avg_playouts = 0.0;
        double avg_elapsed = 0.0;
        for (int game_idx = 0; game_idx < q.games; ++game_idx) {
            agent_->GetState() = Sgf::Get().FromString(sgf_list[game_idx], 9999);
            agent_->GetSearch().ReleaseTree();
            agent_->GetNetwork().ClearCache();
            agent_->GetNetwork().ResetNumQueries();

            auto result = agent_->GetSearch().Computation(
                              param->playouts, Search::kThinking);

            LOGGING << Format(
                "   - game %d/%d -> %.2f p/s\n",
                game_idx+1,
                q.games,
                result.playouts / result.elapsed);
            avg_playouts += result.playouts / (double)q.games;
            avg_elapsed += result.elapsed / (double)q.games;
        }
        double playouts_per_move = avg_playouts;
        double playouts_per_second = playouts_per_move / avg_elapsed;

        elo_list.emplace_back(
            ComputeEloEffect(playouts_per_move, playouts_per_second, q.threads));
        double elodiff = elo_list[i] - elo_list[0];
        char sign = elodiff >= 0.0 ? '+' : '-';
        LOGGING << Format(
            "   - final result -> %.2f p/s (elo %.1f, elo diff %c%.1f)\n",
            playouts_per_second, elo_list[i], sign, std::abs(elodiff));
    }
}
