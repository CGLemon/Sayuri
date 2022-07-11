#include <vector>

#include "pattern/patterns_scan.h"
#include "pattern/pattern.h"
#include "pattern/mm.h"
#include "utils/format.h"
#include "utils/log.h"
#include "game/sgf.h"
#include "game/iterator.h"

class TeamList {
public:
    void InsertWinners(std::vector<int> vals) {
        std::sort(std::begin(vals), std::end(vals));
        winners_ = vals;
        InsertParticipants(vals);
    }

    void InsertParticipants(std::vector<int> vals) {
        std::sort(std::begin(vals), std::end(vals));
        participants_.emplace_back(vals);
    }

    std::vector<int> winners_;
    std::vector<std::vector<int>> participants_;
};

void WritePatterns(std::ostream &out, TeamList list) {
    if (list.participants_.size() <= 1) {
        // Only winner.
        return;
    }

    // Remove the same patterns.
    std::sort(std::begin(list.participants_),
                  std::end(list.participants_));
    list.participants_.erase(std::unique(std::begin(list.participants_),
                                             std::end(list.participants_)),
                             std::end(list.participants_));

    out << "\n#\n"; 
    for (int idx: list.winners_) {
        out << idx << '\n';
    }
    for (int idx: list.winners_) {
        out << idx;
    }
    for (auto &participant: list.participants_) {
        out << '\n';
        for (int idx: participant) {
            out << idx << ' ';
        }
    }
}

void WriteHeader(std::ostream &out, GammasDict &dict) {
    out << Format("! %d\n", dict.Size());
    out << Format("%zu\n", dict.GetAllFeatures().size());

    for (auto f : dict.GetAllFeatures()) {
        out << dict.GetNumFeatures(f)
                << " "
                << kFeaturesNameMap[f]
                << '\n';
    }
    out << "!";
}

void OutputGammas(CGameCollection &gcol, GammasDict &dict, std::string filename) {
    std::ofstream ofs(filename);

    for (unsigned i = 0; i < gcol.vGamma.size(); i++) {
        if (i != 0) {
            ofs << '\n';
        }
        ofs << dict.GetPattern(i)() << ' ' <<  gcol.vGamma[i];
    }

    ofs.close();
}

PatternsScan& PatternsScan::Get() {
    static PatternsScan ps;
    return ps;
}

void PatternsScan::MMTraining(std::string sgf_name, std::string filename) const {
    auto sgfs = SgfParser::Get().ChopAll(sgf_name);

    GammasDict dict;

    // Collect all winner patterns from the data set as patterns set.
    for (const auto &sgf: sgfs) {
        CollectPatterns(sgf, dict);
    }

    // Sort the patterns set that beacuse the MM algorithm updating each
    // features.
    dict.Sort();

    LOGGING << "Total " << dict.Size() << " gammas.\n";
    LOGGING << "Total " << dict.GetAllFeatures().size() << " features.\n";

    CGameCollection gcol;
    std::stringstream ss;
    WriteHeader(ss, dict);

    LOGGING << ss.str() << '\n';

    // Collect winners and participants from the data set.
    int i = 0;
    for (const auto &sgf: sgfs) {
        CollectGammas(sgf, dict, ss);
        if (++i % 100 == 0) {
            LOGGING << "Parsed " << i << " games\n";
        }
    }

    // Start the MM training.
    MinorizationMaximizationTraining(gcol, ss);

    // Save the training result.
    OutputGammas(gcol, dict, filename);
}

void PatternsScan::CollectPatterns(std::string sgfstring, GammasDict &dict) const {
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

    do {
        const auto vertex = game_ite.GetVertex();
        const auto color = game_ite.GetToMove();
        GameState& main_state = game_ite.GetState();

        auto plist = main_state.board_.GetAllPatterns(vertex, color);

        for (auto p : plist) {
            dict.InsertPattern(p);
        }
    } while (game_ite.Next());
}

void PatternsScan::CollectGammas(std::string sgfstring,
                                     GammasDict &dict,
                                     std::ostream &out) const {
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

    const auto board_size = state.GetBoardSize();
    const auto num_intersections = state.GetNumIntersections();
    TeamList team_list;

    do {
        const auto vertex = game_ite.GetVertex();
        const auto color = game_ite.GetToMove();
        GameState& main_state = game_ite.GetState();

        // Gather winner patterns.
        auto winners = std::vector<int>{};
        auto plist = main_state.board_.GetAllPatterns(vertex, color);
        for (auto p : plist) {
            const int pattern_index = dict.GetIndex(p());
            if (pattern_index >= 0) {
                winners.emplace_back(pattern_index);
            }
        }

        if (!winners.empty()) {
            team_list.InsertWinners(winners);

            for (auto idx = 0; idx < num_intersections; ++idx) {
                const auto x = idx % board_size;
                const auto y = idx / board_size;
                const auto other_vtx = main_state.GetVertex(x, y);

                // Gather other participant patterns.
                if (main_state.IsLegalMove(other_vtx) && other_vtx != vertex) {
                    auto participants = std::vector<int>{};
                    plist = main_state.board_.GetAllPatterns(vertex, color);

                    for (auto p : plist) {
                        const int pattern_index = dict.GetIndex(p());
                        if (pattern_index >= 0) {
                            participants.emplace_back(pattern_index);
                        }
                    }
                    if (!participants.empty()) {
                        team_list.InsertParticipants(participants);
                    }
                }
            }
        }
        WritePatterns(out, team_list);
    } while (game_ite.Next());
}
