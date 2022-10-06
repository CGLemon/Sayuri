#include "pattern/mm.h"

#include <set>
#include <cmath>
#include <iostream>
#include <fstream>

void MinorizationMaximization::Initialize(std::vector<int> features,
                                              std::vector<std::string> names) {
    features_ = features;
    num_features_ = (int)features.size();
    features_acc_.resize(num_features_);

    if (!features_acc_.empty()) {
        features_acc_[0] = 0;
    }
    for (int i = 1; i < num_features_; ++i) {
        features_acc_[i] = features_acc_[i-1] + features_[i-1];
    }

    num_nonzero_features_ = 0;
    for (int i = 0; i < num_features_; ++i) {
        if (features_[i] > 0) {
            num_nonzero_features_ += 1;
        }
    }

    num_gammas_ = 0;
    mm_gammas_each_feature_.resize(num_features_);
    for (int i = 0; i < num_features_; ++i) {
        auto &mm_gammas = mm_gammas_each_feature_[i];

        const auto num_gammas_each_features = features_[i];
        mm_gammas.resize(num_gammas_each_features);

        for (int j = 0; j < num_gammas_each_features; ++j) {
            auto &mm = mm_gammas[j];
            mm.used = false;
            mm.wins = 0;
            mm.c = 0.f;
            mm.sigma = 0.f;
            mm.gamma = 1.f;

            num_gammas_ += 1;
        }
    }

    participants_.clear();

    if (names.size() != features.size()) {
        std::cerr << "Features Number: " << num_features_ << "\n";
        for (int i = 0; i < num_features_; ++i) {
            const auto num_gammas_each_features = features_[i];
            std::cerr << "  Feature " << i << ": " << num_gammas_each_features << "\n";
        }
    } else {
        std::cerr << "Features Number: " << num_features_ << "\n";
        for (int i = 0; i < num_features_; ++i) {
            const auto num_gammas_each_features = features_[i];
            std::cerr << "  " << names[i] << ": " << num_gammas_each_features << "\n";
        }
    }
    std::cerr << "Gammas Number: " <<  num_gammas_ << "\n";
}

void MinorizationMaximization::AppendParticipantGroup(ParticipantGroup &p) {
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
        std::cerr << "Illegal participant group, discard it." << std::endl;
    }
}

void MinorizationMaximization::StartTraining() {
    std::cerr << "Participant groups number: " << participants_.size() << std::endl;

    ComputeVictories();

    std::cerr << "start training..." << std::endl;

    constexpr int kMaxNumSteps = 300;
    constexpr double kEarlyStoppingDelta = 1e-4f;
    auto log_likelihood = ComputeLogLikelihood();

    std::cerr << "steps: " << 0
                  << ", lose: " << std::exp(-log_likelihood)
                  << " (" << log_likelihood << ")" << std::endl;


    // TODO: Accelerate the training pipe line by Multi-threads.
    for (int s = 0; s < kMaxNumSteps; ++s) {
        for (int i = 0; i < num_features_; ++i) {
            if (features_[i] > 0) {
                MmUpdate(i);
            }
        }

        auto next_log_likelihood = ComputeLogLikelihood();
        auto delta = std::abs(log_likelihood - next_log_likelihood);

        log_likelihood = next_log_likelihood;
        std::cerr << "steps: " << s+1
                      << ", lose: " << std::exp(-log_likelihood)
                      << "(" << log_likelihood << ")" << std::endl;

        if (delta < kEarlyStoppingDelta) {
            break;
        }
    }
}

void MinorizationMaximization::MmUpdate(int feature) {
    for (const auto &p : participants_) {
        double all_gammas = 0.f;

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
                if (feature == loc.feature) {
                    mm.c += team_gamma/mm.gamma;
                    mm.used = true;
                }
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
            if (mm.used) {
                const double new_gamma = (mm.wins + kPriorVictories) /
                                             (mm.sigma + kPriorGames / (mm.gamma + kPriorOpponentGamma));
                mm.gamma = new_gamma;
            }
            mm.used = false;
            mm.sigma = 0.f;
        }
    }
}

double MinorizationMaximization::ComputeLogLikelihood() const {
    double res = 0.f;

    for (auto &p : participants_) {
        int team_idx = 0;

        double winner_gammas = 0.f;
        double all_gammas = 0.f;

        for (const auto &team : p.all_teams) {

            double team_gamma = 1.f;

            // compute team gamma
            for (const auto loc : team) {
                team_gamma *= mm_gammas_each_feature_[loc.feature][loc.index].gamma;
            }

            if (team_idx == p.winner_team_idx) {
                winner_gammas += team_gamma;
            }
            all_gammas += team_gamma;

            team_idx += 1;
        }
        res += std::log(winner_gammas);
        res -= std::log(all_gammas);
    }

    return res / participants_.size();
}

void MinorizationMaximization::ComputeVictories() {
    int all_wins = 0;
    for (auto &p : participants_) {
        for (const auto loc : p.all_teams[p.winner_team_idx]) {
            auto &mm = GetMmGamma(loc.feature, loc.index);
            mm.wins += 1;
            all_wins += 1;
        }
    }
    (void) all_wins;
}

MinorizationMaximization::MmGamma &MinorizationMaximization::GetMmGamma(int feature, int index) {
    return mm_gammas_each_feature_[feature][index];
}

int MinorizationMaximization::GetLineIndex(int feature, int index) const {
    return features_acc_[feature] + index;
}

void MinorizationMaximization::SaveMmFIle(std::string filename) {
    // https://www.remi-coulom.fr/Amsterdam2007/

    std::ofstream file(filename, std::ofstream::out);

    if (!file.is_open()) {
        return;
    }
    file << "! " << num_gammas_ << std::endl;
    file << num_nonzero_features_ << std::endl;

    for (int i = 0; i < num_features_; ++i) {
        if (features_[i] > 0) {
            file << features_[i] << " Feature" << i << std::endl;
        }
    }
    file << "!" << std::endl;

    for (auto &p : participants_) {
        file << "#" << std::endl;

        int i = 0;
        for (const auto loc : p.all_teams[p.winner_team_idx]) {
            if (i++) file << " ";
            file << GetLineIndex(loc.feature, loc.index);
        }
        file << std::endl;

        for (const auto &team : p.all_teams) {
            int i = 0;
            for (const auto loc : team) {
                if (i++) file << " ";
                file << GetLineIndex(loc.feature, loc.index);
            }
            file << std::endl;
        }
    }

    file.close();
}
