#include "pattern/mm_trainer.h"
#include "pattern/pattern.h"
#include "game/sgf.h"
#include "game/simple_board.h"
#include "game/iterator.h"
#include "utils/format.h"
#include "utils/log.h"

MmTrainer& MmTrainer::Get() {
    static MmTrainer mm_trainer;
    return mm_trainer;
}

void MmTrainer::Run(std::string sgf_name) {
    auto sgfs = SgfParser::Get().ChopAll(sgf_name);

    feature_spat_dicts_.resize(kMaxPatternDist+1);
    feature_orders_.resize(kMaxPatternDist+1);
    feature_order_dicts_.resize(kMaxPatternDist+1);

    // gather mm patterns
    for (const auto &sgf_string : sgfs) {
        FillPatterns(sgf_string);
    }

    // init mm training pipe
    auto features_size = kMaxPatternDist+1;
    auto features = std::vector<int>{};
    for (int i = 0; i < features_size; ++i) {
        features.emplace_back(feature_orders_.size());
    }
    mm_.Initialize(features);

    // fill mm participant
    for (const auto &sgf_string : sgfs) {
        FillMmParticipant(sgf_string);
    }

    // start training
    mm_.StartTraining();
}

void MmTrainer::FillPatterns(std::string sgfstring) {
    GameState state;

    try {
        state = Sgf::Get().FromString(sgfstring, 9999);
    } catch (const char *err) {
        LOGGING << "Fail to load the SGF file! Discard it." << std::endl
                    << Format("\tCause: %s.", err) << std::endl;
        return;
    }

    auto game_ite = GameStateIterator(state);

    if (game_ite.MaxMoveNumber() == 0) {
        return;
    }

    // Remove the double pass moves in the middle.
    game_ite.RemoveUnusedDoublePass();

    do {
        const int pattern_dist = 3;
        FeatureSpatDict &spat_dict = feature_spat_dicts_[pattern_dist];
        FeatureOrder &order = feature_orders_[pattern_dist];
        FeatureOrderDict &order_dict = feature_order_dicts_[pattern_dist];

        auto vtx = game_ite.GetVertex();
        SimpleBoard& board = game_ite.GetState().board_;

        bool success = true;
        for (int symm = 0; symm < 8; ++symm) {
            for (int c = 0; c < 2; ++c) {
                auto hash = board.GetSymmetryPatternHash(vtx, c, pattern_dist, 0);
                auto it = spat_dict.find(hash);

                if (it != std::end(spat_dict)) {
                    // The pattern was already in the dictionary.
                    success = false;
                    break;
                }
            }
            if (!success) break;
        }
        if (success) {
            auto hash = board.GetPatternHash(vtx, kBlack, pattern_dist);
            auto spat = board.GetPatternSpat(vtx, kBlack, pattern_dist);
            auto index = order.size();

            spat_dict.insert({hash, spat});
            order.emplace_back(hash);
            order_dict.insert({hash, index});
        }
    } while (game_ite.Next());
}

void MmTrainer::FillMmParticipant(std::string sgfstring) {
    GameState state;

    try {
        state = Sgf::Get().FromString(sgfstring, 9999);
    } catch (const char *err) {
        LOGGING << "Fail to load the SGF file! Discard it." << std::endl
                    << Format("\tCause: %s.", err) << std::endl;
        return;
    }

    auto game_ite = GameStateIterator(state);

    if (game_ite.MaxMoveNumber() == 0) {
        return;
    }

    // Remove the double pass moves in the middle.
    game_ite.RemoveUnusedDoublePass();

    do {
        const int pattern_dist = 3;
        FeatureOrderDict &order_dict = feature_order_dicts_[pattern_dist];
        Participant part;
        part.winner_team_idx = -1;

        auto winner_vtx = game_ite.GetVertex();
        auto color = game_ite.GetToMove();
        SimpleBoard& board = game_ite.GetState().board_;

        const int empty_cnt = board.GetEmptyCount();
        for (int i = 0; i < empty_cnt; ++i) {
            const auto vtx = board.GetEmpty(i);
            if (board.IsLegalMove(vtx, color)) {

                int matched_index = -1;
                Participant::GammasTeam team;
                for (int symm = 0; symm < 8; ++symm) {
                    for (int c = 0; c < 2; ++c) {
                        auto hash = board.GetSymmetryPatternHash(vtx, c, pattern_dist, 0);
                        auto it = order_dict.find(hash);

                        if (it != std::end(order_dict)) {
                            // The pattern is in the dictionary.
                            matched_index = it->second;
                            break;
                        }
                    }
                    if (matched_index >= 0) break;
                }

                if (matched_index >= 0) {
                    Participant::GammaLoc gloc;
                    gloc.feature = pattern_dist;
                    gloc.index = matched_index;
                    team.emplace_back(gloc);
                    part.all_teams.emplace_back(team);

                    if (vtx == winner_vtx) {
                        // It is the winner team.
                        part.winner_team_idx = 0;
                        int last = part.all_teams.size() - 1;
                        std::swap(part.all_teams[part.winner_team_idx], part.all_teams[last]);
                    }
                }
            }

            if (part.winner_team_idx >= 0) {
                mm_.AppendParticipant(part);
            }
        }
    } while (game_ite.Next());
}

void MmTrainer::SaveResult(std::string filename) {

    std::ofstream file(filename, std::ofstream::app | std::ofstream::out);
    if (!file.is_open()) {
        return;
    }

    for (int feature = 0; feature < (int)feature_orders_.size(); ++feature) {
        FeatureSpatDict &spat_dict = feature_spat_dicts_[feature];
        FeatureOrder &order = feature_orders_[feature];

        for (int index = 0; index < (int)order.size(); ++index) {
            float gamma = mm_.GetMmGamma(feature, index).gamma;
            std::uint64_t hash = order[index];
            std::string spat = spat_dict.find(hash)->second;
            int dist = feature;

            if (dist <= kMaxPatternDist) {
                file << gamma << ' ' << dist << ' ' << spat << '\n';
            }
        }
    }
    file.close();
}
