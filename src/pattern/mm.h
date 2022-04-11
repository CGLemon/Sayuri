#pragma once

#include <vector>
#include <string>

class CTeam {
public:
    CTeam(): Index(vi.size()), Size(0) {}

    int GetSize() const { return Size; }
    int GetIndex(int i) const { return vi[Index+i]; }
    void Append(int i) { vi.emplace_back(i); Size++; }

private:
    static std::vector<int> vi;
    int Index;
    int Size;
};

class CGame {
public: 
    CTeam Winner;
    std::vector<CTeam> vParticipants;
};

class CGameCollection {
public:
    std::vector<CGame> vgame;
    std::vector<double> vGamma;
    std::vector<int> vFeatureIndex;
    std::vector<std::string> vFeatureName;
    std::vector<double> vVictories;
    std::vector<int> vParticipations;
    std::vector<int> vPresences;

    double GetTeamGamma(const CTeam &team) const;
    void ComputeVictories();
    void MM(int Feature);
    double LogLikelihood() const;
};

void MinorizationMaximizationTraining(CGameCollection &gcol, std::istream &data);
