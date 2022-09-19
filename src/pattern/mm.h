#pragma once

#include <vector>
#include <string>

struct ParticipantGroup {
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
        bool used;
        int wins;
        double c;
        double sigma;
        double gamma;
    };
    using MmGammas = std::vector<MmGamma>;

    MmGamma &GetMmGamma(int feature, int index);
    void Initialize(std::vector<int> features);
    void AppendParticipantGroup(ParticipantGroup &p);
    void StartTraining();
    void SaveMmFIle(std::string filename);

private:
    void MmUpdate(int feature);

    void ComputeVictories();
    double ComputeLogLikelihood() const;
    int GetLineIndex(int feature, int index) const;

    int num_features_;
    int num_nonzero_features_;
    int num_gammas_;
    std::vector<int> features_;
    std::vector<int> features_acc_;

    std::vector<MmGammas> mm_gammas_each_feature_;
    std::vector<ParticipantGroup> participants_;
};
