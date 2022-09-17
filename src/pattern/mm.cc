#include "pattern/mm.h"

#include <set>
#include <cmath>
#include <iostream>

void MinorizationMaximization::Initialize(std::vector<int> features) {
    features_ = features;
    num_features_ = (int)features.size();

    mm_gammas_each_feature_.resize(num_features_);
    for (int i = 0; i < num_features_; ++i) {
        auto &mm_gammas = mm_gammas_each_feature_[i];

        const auto num_gammas_each_features = features_[i];
        mm_gammas.resize(num_gammas_each_features);

        for (int j = 0; j < num_gammas_each_features; ++j) {
            auto &mm = mm_gammas[j];
            mm.wins = 0;
            mm.c = 0.f;
            mm.sigma = 0.f;
            mm.gamma = 1.f;
        }
    }

    participants_.clear();
}

void MinorizationMaximization::AppendParticipant(Participant &p) {
    bool success = true;
    for (auto &team : p.all_teams) {
        std::set<int> buf;
        for (const auto loc : team) {
            if (buf.count(loc.feature) != 0) {
                success = false;
            }
            buf.insert(loc.feature);
        }
    }

    if (success) {
        participants_.emplace_back(p);
    } else {
        std::cerr << "Illegal participant, discard it." << std::endl;
    }
}

void MinorizationMaximization::StartTraining() {
    ComputeVictories();

    constexpr int NumSteps = 300;
    for (int s = 0; s < NumSteps; ++s) {
        MmUpdate();

        const auto log_likelihood = ComputeLogLikelihood();
        std::cerr << "steps: " << s << ", lose: " << std::exp(-log_likelihood) << std::endl;
    }
}

void MinorizationMaximization::MmUpdate() {
    for (const auto &p : participants_) {
        double all_gammas = 1.f;

        // gather the C_ij and E_j
        for (const auto &team : p.all_teams) {

            double team_gamma = 1.f;

            // compute team gamma
            for (const auto loc : team) {
                team_gamma *= GetMmGamma(loc.feature, loc.index).gamma;
            }

            // gather the C_ij
            for (const auto loc : team) {
                auto &mm = GetMmGamma(loc.feature, loc.index);
                mm.c += team_gamma/mm.gamma;
            }

            // gather the E_j
            all_gammas += team_gamma;
        }

        // update sigma
        for (auto &mm_gammas: mm_gammas_each_feature_) {
            for (auto &mm : mm_gammas) {
                mm.sigma += mm.c/all_gammas;
                mm.c = 0.f;
            }
        }
    }

    // compute the new gammas

    constexpr double kPriorVictories = 1.f;
    constexpr double kPriorGames = 2.f;
    constexpr double kPriorOpponentGamma = 1.f;

    for (auto &mm_gammas: mm_gammas_each_feature_) {
        for (auto &mm : mm_gammas) {
            const double new_gamma = (mm.wins + kPriorVictories) /
                                         (mm.sigma + kPriorGames / (mm.gamma + kPriorOpponentGamma));
            mm.gamma = new_gamma;
            mm.sigma = 0.f;
        }
    }
}

double MinorizationMaximization::ComputeLogLikelihood() const {
    double res = 0.f;

    for (auto &p : participants_) {
        int team_idx = 0;
        for (const auto &team : p.all_teams) {

            double team_gamma = 1.f;

            // compute team gamma
            for (const auto loc : team) {
                team_gamma *= mm_gammas_each_feature_[loc.feature][loc.index].gamma;
            }

            if (team_idx == p.winner_team_idx) {
                res += std::log(team_gamma);
            } else {
                res -= std::log(team_gamma);
            }
        }
        team_idx += 1;
    }

    return res;
}

void MinorizationMaximization::ComputeVictories() {
    for (auto &p : participants_) {
        for (const auto loc : p.all_teams[p.winner_team_idx]) {
            auto &mm = GetMmGamma(loc.feature, loc.index);
            mm.wins += 1;
        }
    }
}

MinorizationMaximization::MmGamma &MinorizationMaximization::GetMmGamma(int feature, int index) {
    return mm_gammas_each_feature_[feature][index];
}
