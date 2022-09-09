#pragma once

#include <vector>

struct Participant {
    struct GammaLoc {
        int feature;
        int index;
    };
    using GammasTeam = std::vector<GammaLoc>;

    int winner_team_idx;
    std::vector<GammasTeam> all_teams;
};

class MinorizationMaximization {
public:
    struct MmGamma {
        int wins;
        double c;
        double sigma;
        double gamma;
    };
    using MmGammas = std::vector<MmGamma>;

    static MinorizationMaximization& Get();

    MmGamma &GetMmGamma(int feature, int index);
    void Initialize(std::vector<int> features);
    void AppendParticipant(Participant &p);
    void StartTraining();

private:
    void MmUpdate();
    void ComputeVictories();
    double ComputeLogLikelihood() const;

    int num_features_;
    std::vector<int> features_;
    std::vector<MmGammas> mm_gammas_each_feature_;
    std::vector<Participant> participants_;
};
