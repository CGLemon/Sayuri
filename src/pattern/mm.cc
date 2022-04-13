/////////////////////////////////////////////////////////////////////////////
//
// mm.cpp
//
// Rémi Coulom
//
// February, 2007
//
/////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <iomanip>
#include <sstream>
#include <map>
#include <cmath>
#include <fstream>
#include <assert.h>

#include "mm.h"

constexpr double PriorVictories = 1.0;
constexpr double PriorGames = 2.0;
constexpr double PriorOpponentGamma = 1.0;

std::vector<int> CTeam::vi;

double CGameCollection::GetTeamGamma(const CTeam &team) const {
    double Result = 1.0;
    for (int i = team.GetSize(); --i >= 0;) {
        Result *= vGamma[team.GetIndex(i)];
    }
    return Result;
}

double CGameCollection::LogLikelihood() const {
    double L = 0;

    for (int i = vgame.size(); --i >= 0;) {
        const CGame &game = vgame[i];
        double Opponents = 0; 
        const std::vector<CTeam> &v = game.vParticipants;

        for (int j = v.size(); --j >= 0;) {
            Opponents += GetTeamGamma(v[j]);
        }

        L += std::log(GetTeamGamma(game.Winner));
        L -= std::log(Opponents);
    }

    return L;
}

void CGameCollection::ComputeVictories() {
    vVictories.resize(vGamma.size());
    vParticipations.resize(vGamma.size());
    vPresences.resize(vGamma.size());

    for (int i = vVictories.size(); --i >= 0;) {
        vVictories[i] = 0;
        vParticipations[i] = 0;
        vPresences[i] = 0;
    }

    for (int i = vgame.size(); --i >= 0;) {
        const CTeam &Winner = vgame[i].Winner;
        for (int j = Winner.GetSize(); --j >= 0;) {
            vVictories[Winner.GetIndex(j)]++;
        }

        int tParticipations[vGamma.size()];
        for (int j = vGamma.size(); --j >= 0;) {
            tParticipations[j] = 0;
        }

        for (int k = vgame[i].vParticipants.size(); --k >= 0;) {
            for (int j = vgame[i].vParticipants[k].GetSize(); --j >= 0;) {
                int Index = vgame[i].vParticipants[k].GetIndex(j);
                vParticipations[Index]++;
                tParticipations[Index]++;
            }
        }

        for (int i = vGamma.size(); --i >= 0;) {
            if (tParticipations[i]) {
                vPresences[i]++;
            }
        }
    }
}

void CGameCollection::MM(int Feature) {
    //
    // Interval for this feature
    //
    int Max = vFeatureIndex[Feature + 1];
    int Min = vFeatureIndex[Feature];

    //
    // Compute denominator for each gamma
    //
    std::vector<double> vDen(vGamma.size());
    for (int i = vDen.size(); --i >= 0;) {
        vDen[i] = 0.0;
    }

    //
    // Main loop over games
    //
    std::map<int,double> tMul;


    for (int i = vgame.size(); --i >= 0;) {

        tMul.clear();
        double Den = 0.0;

        std::vector<CTeam> &v = vgame[i].vParticipants;
        for (int i = v.size(); --i >= 0;) {
            const CTeam &team = v[i];
            double Product = 1.0;
            int FeatureIndex = -1;

            for (int i = 0; i < team.GetSize(); i++) {
                int Index = team.GetIndex(i);
                if (Index >= Min && Index < Max) {
                    FeatureIndex = Index;
                } else {
                    Product *= vGamma[Index];
                }
            }

            if (FeatureIndex >= 0) {
                if (tMul.count(FeatureIndex)) {
                    tMul[FeatureIndex] += Product;
                } else {
                    tMul[FeatureIndex] = Product;
                }

                Product *= vGamma[FeatureIndex];
            }

            Den += Product;
        }


        for (std::map<int,double>::iterator it=tMul.begin();it!=tMul.end(); ++it) {
            int key=it->first;
            vDen[key]+=it->second / Den;
        }
    }

    //
    // Update Gammas
    //
    for (int i = Max; --i >= Min;) {
        double NewGamma = (vVictories[i] + PriorVictories) /
                              (vDen[i] + PriorGames / (vGamma[i] + PriorOpponentGamma));
        vGamma[i] = NewGamma;
    }
}

int gamma_to_feature(int gamma, std::vector<int> &vFeatureIndex) {
	for (unsigned int i = 0; i < vFeatureIndex.size(); i++) {
		if (vFeatureIndex[i] > gamma) {
			return i;
        }
	}
	return vFeatureIndex.size();
}

CTeam ReadTeam(std::string &s, std::vector<int> &vFeatureIndex, int Gammas) {
    std::istringstream in(s);
    CTeam team;
    int Index;
    while(1) {
        in >> Index;
        if (Index < 0 || Index >= Gammas) {
	        std::cerr << '\n' << s << '\n';
	        fprintf(stderr, "invalid gamma: %i\n", Index);
        }

        if (in) {
	        int feature = gamma_to_feature(Index, vFeatureIndex);
	        for (int i = team.GetSize(); --i >= 0;) {
		        if (feature == gamma_to_feature(team.GetIndex(i), vFeatureIndex)) {
			        std::cerr << '\n' << s << '\n';
			        fprintf(stderr, "%i and %i are same feature !\n", Index, team.GetIndex(i));
		        }
	        }
	        team.Append(Index);
        } else {
           break;
        }
    }
    return team;
}

void ReadGameCollection(CGameCollection &gcol, std::istream &in) {
    //
    // Read number of gammas in the first line
    //
    int MaxGamma;
    {
        std::string sLine;
        std::getline(in, sLine);
        std::istringstream is(sLine);
        std::string s;
        int Gammas = 0;
        is >> s >> Gammas;
        MaxGamma = Gammas;
        gcol.vGamma.resize(Gammas);
        for (int i = Gammas; --i >= 0;) {
            gcol.vGamma[i] = 1.0;
        }
    }

    gcol.MaxGamma = MaxGamma;

    //
    // Features
    //
    {
        gcol.vFeatureIndex.push_back(0);
        int Features = 0;
        in >> Features;
        for (int i = 0; i < Features; i++) {
            int Gammas;
            in >> Gammas;
            int Min = gcol.vFeatureIndex.back();
            gcol.vFeatureIndex.push_back(Min + Gammas);
            std::string sName;
            in >> sName;
            gcol.vFeatureName.push_back(sName);
        }
    }

    //
    // Main loop over games
    //
    std::string sLine;
    std::getline(in, sLine);

    while(in) {
        //
        // Parse a game
        //
        if (sLine == "#") {
            CGame game;

            //
            // Winner
            //
            std::getline(in, sLine);
            game.Winner = ReadTeam(sLine, gcol.vFeatureIndex, MaxGamma);

            //
            // Participants
            //
            std::getline(in, sLine);
            while (sLine[0] != '#' && sLine[0] != '!' && in) {
                CTeam team = ReadTeam(sLine, gcol.vFeatureIndex, MaxGamma);
                game.vParticipants.push_back(team);
                std::getline(in, sLine);
            }

            gcol.vgame.push_back(game);
        } else {
            std::getline(in, sLine);
            std::cerr << '.';
        }
    }
    std::cerr << '\n';
}

void MinorizationMaximizationTraining(CGameCollection &gcol, std::istream &data) {
     ReadGameCollection(gcol, data);
     gcol.ComputeVictories();
     std::cerr << "Games = " << gcol.vgame.size() << '\n';
     std::cerr << "Gammas = " << gcol.MaxGamma << '\n';

     double LogLikelihood = gcol.LogLikelihood() / gcol.vgame.size();
     const int Features = gcol.vFeatureName.size();
     double tDelta[Features];

    for (int k = 2; --k >= 0;) {
        for (int i = Features; --i >= 0;) {
            tDelta[i] = 10.0;
        }

        while(true) {
            //
            // Select feature with max delta
            //
            int Feature = 0;
            double MaxDelta = tDelta[0];
            for (int j = Features; --j > 0;) {
                if (tDelta[j] > MaxDelta) {
                    MaxDelta = tDelta[Feature = j];
                }
            }

            if (MaxDelta < 0.0001) break;
   
            //
            // Run one MM iteration over this feature
            //
            std::cerr << std::setw(20) << gcol.vFeatureName[Feature] << ' ';
            std::cerr << std::setw(9) << LogLikelihood << ' ';
            std::cerr << std::setw(9) << std::exp(-LogLikelihood) << ' ';

            gcol.MM(Feature);
            double NewLogLikelihood = gcol.LogLikelihood() / gcol.vgame.size();
            double Delta = NewLogLikelihood - LogLikelihood;

            tDelta[Feature] = Delta;
            std::cerr << std::setw(9) << Delta << '\n';
            LogLikelihood = NewLogLikelihood;
        }
    }
}
