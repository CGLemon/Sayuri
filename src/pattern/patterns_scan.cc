#include <set>

#include "pattern/patterns_scan.h"
#include "pattern/pattern.h"
#include "pattern/mm.h"
#include "utils/format.h"
#include "utils/log.h"
#include "game/sgf.h"

template<typename T=int>
class CTeamList {
public:
    void InsertWinner(T val) {
        winner_ = val;
        set_.insert(val);
    }
    void Insert(T val) {
        set_.insert(val);
    }

    std::set<T> set_;
    T winner_;
};

void WritePatterns(std::ostream &out, CTeamList<>& list) {
    if (list.set_.size() <= 1) {
        return;
    }

    int winner = list.winner_;

    out << "\n#\n" << winner;

    for (int p: list.set_) {
        out << '\n' << p;
    }
}

void WriteHeader(std::ostream &out, int size) {
    out << Format("! %d\n", size);
    out << "1\n";
    out << Format("%d s3\n", size);
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

    for (const auto &sgf: sgfs) {
        CollectPatterns(sgf, dict);
    }

    CGameCollection gcol;
    std::stringstream ss;
    WriteHeader(ss, dict.Size());

    for (const auto &sgf: sgfs) {
        CollectGammas(sgf, dict, ss);
    }

    std::cout << ss.str(); 

    MinorizationMaximizationTraining(gcol, ss);
    OutputGammas(gcol, dict, filename);
}

void PatternsScan::CollectPatterns(std::string sgfstring, GammasDict &dict) const {
    GameState state;
    try {
        state = Sgf::Get().FromString(sgfstring, 9999);
    } catch (const char *err) {
        ERROR << "Fail to load the SGF file! Discard it." << std::endl
                  << Format("\tCause: %s.", err) << std::endl;
        return;
    }

    auto history = state.GetHistory();
    auto movelist = std::vector<int>{};

    for (const auto &board : history) {
        auto vtx = board->GetLastMove();
        if (vtx != kNullVertex) {
            movelist.emplace_back(vtx);
        }
    }

    GameState main_state;
    main_state.Reset(state.GetBoardSize(), state.GetKomi());

    for (int i = 0; i < (int)movelist.size(); ++i) {
        const int vertex = movelist[i];
        const auto hash = main_state.board_.GetPattern3x3(vertex);

        dict.InsertPattern(Pattern::GetSpatial3x3(hash));

        main_state.PlayMove(vertex);
    }
}

void PatternsScan::CollectGammas(std::string sgfstring,
                                     GammasDict &dict,
                                     std::ostream &out) const {
    GameState state;
    try {
        state = Sgf::Get().FromString(sgfstring, 9999);
    } catch (const char *err) {
        ERROR << "Fail to load the SGF file! Discard it." << std::endl
                  << Format("\tCause: %s.", err) << std::endl;
        return;
    }

    auto history = state.GetHistory();
    auto movelist = std::vector<int>{};

    for (const auto &board : history) {
        auto vtx = board->GetLastMove();
        if (vtx != kNullVertex) {
            movelist.emplace_back(vtx);
        }
    }

    GameState main_state;
    main_state.Reset(state.GetBoardSize(), state.GetKomi());
    int board_size = state.GetBoardSize();
    int num_intersections = state.GetNumIntersections();

    CTeamList<> cteam_list;


    for (int i = 0; i < (int)movelist.size(); ++i) {

        const int vertex = movelist[i];

        int pattern_index = dict.GetIndex(Pattern::Bind(kSpatial3x3,
                                                            main_state.board_.GetPattern3x3(vertex)));
        // set winner
        if (pattern_index >= 0) {
            cteam_list.InsertWinner(pattern_index);
        }

        for (auto idx = 0; idx < num_intersections; ++idx) {
            const auto x = idx % board_size;
            const auto y = idx / board_size;
            const auto other_vtx = main_state.GetVertex(x, y);

            if (main_state.IsLegalMove(other_vtx) && other_vtx != vertex) {

                pattern_index = dict.GetIndex(Pattern::Bind(kSpatial3x3,
                                                                main_state.board_.GetPattern3x3(other_vtx)));
                if (pattern_index >= 0) {
                    cteam_list.InsertWinner(pattern_index);
                }
            }
        }

        WritePatterns(out, cteam_list);

        main_state.PlayMove(vertex);
    }
}
