#include "game/gtp.h"
#include "game/sgf.h"
#include "game/commands_list.h"
#include "utils/log.h"
#include "utils/time.h"
#include "utils/komi.h"
#include "utils/gogui_helper.h"
#include "utils/filesystem.h"
#include "pattern/mm_trainer.h"
#include "neural/encoder.h"

#include <array>
#include <atomic>
#include <iomanip>
#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>

void GtpLoop::Loop() {
    while (true) {
        auto input = std::string{};
        if (std::getline(std::cin, input)) {

            auto spt = Splitter(input);
            WRITING << ">>" << ' ' << input << std::endl;

            curr_id_ = -1;

            // check the command id here
            if (const auto token = spt.GetWord(0)) {
                if (token->IsDigit()) {
                    bool error;
                    curr_id_ = token->Get<int>(curr_id_, error);
                    if (error) {
                        LOGGING << "error: GTP ID must be INT\n";
                    }
                    spt.RemoveWord(token->Index());
                }
            }

            if (!spt.Valid()) {
                continue;
            }

            auto out = std::string{};
            auto stop = false;
            auto try_ponder = false;

            if (spt.GetCount() == 1 && spt.Find("quit")) {
                agent_->Quit();
                out = GtpSuccess("");
                stop = true;
            }

            if (out.empty()) {
                out = Execute(spt, try_ponder);
            }
            prev_pondering_ = try_ponder; // save the last pondering status

            if (!out.empty()) {
                DUMPING << out;
            }

            if (stop) {
                break;
            }
            if (try_ponder) {
                agent_->GetSearch().TryPonder();
            }
        }
    }
}

std::string GtpLoop::Execute(Splitter &spt, bool &try_ponder) {
    if (!agent_) {
        return std::string{};
    }

    auto out = std::ostringstream{};

    if (const auto res = spt.Find("protocol_version", 0)) {
        out << GtpSuccess(std::to_string(kProtocolVersion));
    } else if (const auto res = spt.Find("name", 0)) {
        out << GtpSuccess(GetProgramName());
    } else if (const auto res = spt.Find("version", 0)) {
        out << GtpSuccess(version_verbose_);
    } else if (const auto res = spt.Find("showboard", 0)) {
        agent_->GetState().ShowBoard();
        out << GtpSuccess("");
    } else if (const auto res = spt.Find("boardsize", 0)){
        int bsize = -1;
        if (const auto input = spt.GetWord(1)) {
            bool error;
            bsize = input->Get<int>(bsize, error);
            if (error) {
                LOGGING << "error: board size must be INT\n";
            }
        }

        if (bsize <= kBoardSize &&
                bsize <= kMaxGTPBoardSize &&
                bsize >= kMinGTPBoardSize) {
            agent_->SetBoardSize(bsize);
            out << GtpSuccess("");
        } else {
            out << GtpFail("invalid board size");
        }
    } else if (const auto res = spt.Find("clear_board", 0)){
        agent_->GetSearch().ReleaseTree();
        agent_->GetNetwork().ClearCache();
        agent_->GetState().ClearBoard();
        out << GtpSuccess("");
    } else if (const auto res = spt.Find("komi", 0)) {
        auto komi = agent_->GetState().GetKomi();
        bool success = true;
        bool error;

        if (const auto input = spt.GetWord(1)) {
            komi = input->Get<float>(komi, error);
        } else {
            success = false;
        }
        if (success && !error) {
            agent_->GetState().SetKomi(komi);
            out << GtpSuccess("");
        } else {
            out << GtpFail("invalid komi");
        }
    } else if (const auto res = spt.Find("play", 0)) {
        const auto end = spt.GetCount() < 3 ? spt.GetCount() : 3;
        auto cmd = std::string{};

        if (const auto input = spt.GetSlice(1, end)) {
            cmd = input->Get<>();
        }
        if (agent_->GetState().PlayTextMove(cmd)) {
            out << GtpSuccess("");
        } else {
            out << GtpFail("invalid play");
        }
    } else if (const auto res = spt.Find("fixed_handicap", 0)) {
        auto handicaps = -1;
        if (const auto input = spt.GetWord(1)) {
            bool error;
            handicaps = input->Get<int>(handicaps, error);
            if (error) {
                LOGGING << "error: handicaps must be <INT>\n";
            }
        }
        if (handicaps >= 1 &&
                agent_->GetState().SetFixdHandicap(handicaps)) {
            out << GtpSuccess("");
        } else {
            out << GtpFail("invalid handicap");
        }
    } else if (const auto res = spt.Find("place_free_handicap", 0)) {
        auto handicaps = -1;
        if (const auto input = spt.GetWord(1)) {
            bool error;
            handicaps = input->Get<int>(handicaps, error);
            if (error) {
                LOGGING << "error: handicaps must be <INT>\n";
            }
        }
        bool network_valid = agent_->GetNetwork().Valid();
        int max_handicaps = network_valid ?
                                agent_->GetState().GetNumIntersections() / 4 :
                                9;
        auto stones_list = std::vector<int>{};

        if (handicaps >= 1 && handicaps <= max_handicaps) {
            if (network_valid) {
                for (int i = 0; i < handicaps; ++i) {
                    const int vtx =
                        agent_->GetNetwork().GetVertexWithPolicy(
                            agent_->GetState(), 1.f, false);
                    stones_list.emplace_back(vtx);
                    // agent_->GetState().ClearBoard();
                    agent_->GetState().PlayHandicapStones(stones_list, true);
                }
            } else {
                stones_list = agent_->GetState().PlaceFreeHandicap(handicaps);
            }
        }

        if (!stones_list.empty()) {
            auto vtx_list = std::ostringstream{};
            for (size_t i = 0; i < stones_list.size(); i++) {
                const auto vtx = stones_list[i];
                vtx_list << agent_->GetState().VertexToText(vtx);
                if (i != stones_list.size() - 1) vtx_list << ' ';
            }
            out << GtpSuccess(vtx_list.str());
        } else {
            out << GtpFail("invalid handicap");
        }
    } else if (const auto res = spt.Find("set_free_handicap", 0)) {
        auto movelist = std::vector<std::string>{};
        for (auto i = size_t{1}; i < spt.GetCount(); ++i) {
            movelist.emplace_back(spt.GetWord(i)->Get<>());
        }
        if (agent_->GetState().SetFreeHandicap(movelist)) {
            out << GtpSuccess("");
        } else {
            out << GtpFail("invalid handicap");
        }
    } else if (const auto res = spt.Find("loadsgf", 0)) {
        auto movenum = 9999;
        auto filename = std::string{};
        if (const auto input = spt.GetWord(1)) {
            filename = input->Get<>();
        }
        if (const auto input = spt.GetWord(2)) {
            bool error;
            movenum = input->Get<int>(movenum, error);
            if (error) {
                LOGGING << "error: movenum must be <INT>\n";
            }
        }
        try {
            agent_->GetState() = Sgf::Get().FromFile(filename, movenum);
            out << GtpSuccess("");
        } catch (const std::exception& e) {
            out << GtpFail(Format("invalid SGF file, cause %s.", e.what()));
        }
    } else if (const auto res = spt.Find("is_legal", 0)) {
        auto color = agent_->GetState().GetToMove();;
        auto move = kNullVertex;

        if (const auto input = spt.GetWord(1)) {
            color = agent_->GetState().TextToColor(input->Get<>());
        }
        if (const auto input = spt.GetWord(2)) {
            move = agent_->GetState().TextToVertex(input->Get<>());
        }

        if (color == kInvalid || move == kNullVertex) {
            out << GtpFail("invalid is_legal");
        } else {
            if (agent_->GetState().IsLegalMove(move, color)) {
                out << GtpSuccess("1"); // legal move
            } else {
                out << GtpSuccess("0"); // illegal move
            }
        }
    } else if (const auto res = spt.Find("color", 0)) {
        auto move = kNullVertex;

        if (const auto input = spt.GetWord(1)) {
            move = agent_->GetState().TextToVertex(input->Get<>());
        }

        if (move != kNullVertex) {
            auto color = agent_->GetState().GetState(move);
            if (color == kBlack) {
                out << GtpSuccess("black");
            } else if (color == kWhite) {
                out << GtpSuccess("white");
            } else if (color == kEmpty) {
                out << GtpSuccess("empty");
            } else {
                out << GtpSuccess("invalid");
            }
        } else {
            out << GtpFail("invalid color");
        }
    } else if (const auto res = spt.Find("printsgf", 0)) {
        auto filename = std::string{};
        if (const auto input = spt.GetWord(1)) {
            filename = input->Get<>();
        }
        if (filename.empty()) {
            out << GtpSuccess(Sgf::Get().ToString(agent_->GetState()));
        } else {
            Sgf::Get().ToFile(filename, agent_->GetState());
            out << GtpSuccess("");
        }
    } else if (const auto res = spt.Find("get_komi", 0)) {
        out << GtpSuccess(std::to_string(agent_->GetState().GetKomi()));
    } else if (const auto res = spt.Find("get_handicap", 0)) {
        out << GtpSuccess(std::to_string(agent_->GetState().GetHandicap()));
    } else if (const auto res = spt.Find("query_boardsize", 0)) {
        out << GtpSuccess(std::to_string(agent_->GetState().GetBoardSize()));
    } else if (const auto res = spt.Find("clear_cache", 0)) {
        agent_->GetSearch().ReleaseTree();
        agent_->GetNetwork().ClearCache();
        out << GtpSuccess("");
    } else if (const auto res = spt.Find("final_score", 0)) {
        auto result = agent_->GetSearch().Computation(400, Search::kForced);
        auto color = agent_->GetState().GetToMove();
        auto final_score = result.root_score_lead;

        if (agent_->GetState().GetPasses() >= 2) {
            auto dead_list = std::vector<int>{};
            auto fork_state = agent_->GetState();

            for (const auto &string : result.dead_strings) {
                for (const auto vtx: string) {
                    dead_list.emplace_back(vtx);
                }
            }
            fork_state.RemoveDeadStrings(dead_list);
            final_score = fork_state.GetFinalScore(color);
        }

        final_score = AdjustKomi<float>(final_score);
        if (std::abs(final_score) < 1e-4f) {
            color = kEmpty;
        } else if (final_score < 0.f) {
            final_score = -final_score;
            color = !color;
        }

        auto ss = std::ostringstream{};
        if (color == kEmpty) {
            ss << "0";
        } else if (color == kBlack) {
            ss << "B+" << final_score;
        } else if (color == kWhite) {
            ss << "W+" << final_score;
        }
        out << GtpSuccess(ss.str());
    } else if (const auto res = spt.Find("genmove", 0)) {
        auto color = agent_->GetState().GetToMove();
        if (const auto input = spt.GetWord(1)) {
            auto get_color = agent_->GetState().TextToColor(input->Get<>());
            if (get_color != kInvalid) {
                color = get_color;
            }
        }
        agent_->GetState().SetToMove(color);
        auto move = agent_->GetSearch().ThinkBestMove();
        agent_->GetState().PlayMove(move);
        out << GtpSuccess(agent_->GetState().VertexToText(move));
        try_ponder = true;
    } else if (const auto res = spt.Find("selfplay-genmove", 0)) {
        auto color = agent_->GetState().GetToMove();
        if (const auto input = spt.GetWord(1)) {
            auto get_color = agent_->GetState().TextToColor(input->Get<>());
            if (get_color != kInvalid) {
                color = get_color;
            }
        }
        agent_->GetState().SetToMove(color);
        auto move = agent_->GetSearch().GetSelfPlayMove();
        agent_->GetState().PlayMove(move);
        if (agent_->GetState().IsGameOver()) {
            agent_->GetSearch().UpdateTerritoryHelper();
        }
        out << GtpSuccess(agent_->GetState().VertexToText(move));
    } else if (const auto res = spt.Find("selfplay", 0)) {
        while (!agent_->GetState().IsGameOver()) {
            agent_->GetState().PlayMove(agent_->GetSearch().GetSelfPlayMove());
            agent_->GetState().ShowBoard();
        }
        agent_->GetSearch().UpdateTerritoryHelper();
        out << GtpSuccess("");
    } else if (const auto res = spt.Find("dump_training_buffer", 0)) {
        auto filename = std::string{};
        if (const auto input = spt.GetWord(1)) {
            filename = input->Get<>();
        }

        if (!agent_->GetState().IsGameOver()) {
            out << GtpFail("it is not game over yet");
        } else if (filename.empty()) {
            out << GtpFail("invalid file name");
        } else {
            agent_->GetSearch().SaveTrainingBuffer(filename);
            out << GtpSuccess("");
        }
    } else if (const auto res = spt.Find("clear_training_buffer", 0)) {
        agent_->GetSearch().ClearTrainingBuffer();
        out << GtpSuccess("");
    }else if (const auto res = spt.Find("kgs-game_over", 0)) {
        agent_->GetNetwork().ClearCache();
        out << GtpSuccess("");
    } else if (const auto res = spt.Find("kgs-chat", 0)) {
        auto type = std::string{};
        auto name = std::string{};
        auto message = std::string{};
        if (spt.GetCount() < 3) {
            out << GtpFail("invalid chat settings");
        } else {
            type = spt.GetWord(1)->Get<>();
            name = spt.GetWord(2)->Get<>();
            message = spt.GetSlice(3)->Get<>();
            out << GtpSuccess("I'm a go bot, not a chat bot.");
        }
    } else if (const auto res = spt.Find({"analyze",
                                              "lz-analyze",
                                              "kata-analyze",
                                              "sayuri-analyze"}, 0)) {
        auto color = agent_->GetState().GetToMove();
        auto config = ParseAnalysisConfig(spt, color);

        if (curr_id_ >= 0) {
            DUMPING << "=" << curr_id_ << "\n";
        } else {
            DUMPING << "=\n";
        }

        agent_->GetState().SetToMove(color);
        agent_->GetSearch().Analyze(true, config);
        DUMPING << "\n";
    } else if (const auto res = spt.Find({"genmove_analyze",
                                             "lz-genmove_analyze",
                                             "kata-genmove_analyze",
                                             "sayuri-genmove_analyze"}, 0)) {
        auto color = agent_->GetState().GetToMove();
        auto config = ParseAnalysisConfig(spt, color);

        if (curr_id_ >= 0) {
            DUMPING << "=" << curr_id_ << "\n";
        } else {
            DUMPING << "=\n";
        }

        agent_->GetState().SetToMove(color);
        auto move = agent_->GetSearch().Analyze(false, config);
        agent_->GetState().PlayMove(move);
        DUMPING << "play " << agent_->GetState().VertexToText(move) << "\n\n";
        try_ponder = true;
    } else if (const auto res = spt.Find("undo", 0)) {
        if (agent_->GetState().UndoMove()) {
            out << GtpSuccess("");
        } else {
            out << GtpFail("can't do the undo move");
        }
    } else if (const auto res = spt.Find("kgs-time_settings", 0)) {
        // none, absolute, byoyomi, or canadian
        int main_time = 0, byo_yomi_time = 0, byo_yomi_stones = 0, byo_yomi_periods = 0;
        bool success = true;
        bool error;

        if (const auto res = spt.Find("none", 1)) {
            // infinite time
            main_time = byo_yomi_time = byo_yomi_stones = byo_yomi_periods;
        } else if (const auto res = spt.Find("absolute", 1)) {
            main_time = spt.GetWord(2)->Get<int>(main_time, error);
        } else if (const auto res = spt.Find("canadian", 1)) {
            main_time = spt.GetWord(2)->Get<int>(main_time, error);
            byo_yomi_time = spt.GetWord(3)->Get<int>(byo_yomi_time, error);
            byo_yomi_stones = spt.GetWord(4)->Get<int>(byo_yomi_stones, error);
        } else if (const auto res = spt.Find("byoyomi", 1)) {
            main_time = spt.GetWord(2)->Get<int>(main_time, error);
            byo_yomi_time = spt.GetWord(3)->Get<int>(byo_yomi_time, error);
            byo_yomi_periods = spt.GetWord(4)->Get<int>(byo_yomi_periods, error);
        } else {
            success = false;
        }
        if (success && !error) {
            agent_->GetSearch().TimeSettings(main_time, byo_yomi_time,
                                                 byo_yomi_stones, byo_yomi_periods);
            out << GtpSuccess("");
        } else {
            out << GtpFail("invalid time settings");
        }
    } else if (const auto res = spt.Find("time_settings", 0)) {
        int main_time = -1, byo_yomi_time = -1, byo_yomi_stones = -1;

        if (const auto input = spt.GetWord(1)) {
            main_time = input->Get<int>(main_time);
        }
        if (const auto input = spt.GetWord(2)) {
            byo_yomi_time = input->Get<int>(byo_yomi_time);
        }
        if (const auto input = spt.GetWord(3)) {
            byo_yomi_stones = input->Get<int>(byo_yomi_stones);
        }

        if (main_time == -1 || byo_yomi_time == -1 || byo_yomi_stones == -1) {
            out << GtpFail("invalid time settings");
        } else {
            agent_->GetSearch().TimeSettings(main_time, byo_yomi_time, byo_yomi_stones, 0);
            out << GtpSuccess("");
        }
    } else if (const auto res = spt.Find("time_left", 0)) {
        int color = kInvalid, time = -1, stones = -1;

        if (const auto input = spt.GetWord(1)) {
            auto get_color = agent_->GetState().TextToColor(input->Get<>());
            if (get_color != kInvalid) {
                color = get_color;
            }
        }
        if (const auto input = spt.GetWord(2)) {
            time = input->Get<int>(time);
        }
        if (const auto input = spt.GetWord(3)) {
            stones = input->Get<int>(stones);
        }

        if (color == kInvalid || time == -1 || stones == -1) {
            out << GtpFail("invalid time settings");
        } else {
            agent_->GetSearch().TimeLeft(color, time, stones);
            out << GtpSuccess("");
        }
        try_ponder = true;
    } else if (const auto res = spt.Find("final_status_list", 0)) {
        auto result = agent_->GetSearch().Computation(400, Search::kForced);
        auto vtx_list = std::ostringstream{};

        // TODO: support seki option.

        if (const auto input = spt.Find("alive", 1)) {
            for (size_t i = 0; i < result.alive_strings.size(); i++) {
                vtx_list << (i == 0 ? "" : "\n");
                auto &string = result.alive_strings[i];
                for (size_t j = 0; j < string.size(); j++) {
                    auto vtx = string[j];
                    vtx_list << agent_->GetState().VertexToText(vtx);
                    if (j != string.size() - 1) vtx_list << ' ';
                }
            }
            out << GtpSuccess(vtx_list.str());
        } else if (const auto input = spt.Find("dead", 1)) {
             for (size_t i = 0; i < result.dead_strings.size(); i++) {
                vtx_list << (i == 0 ? "" : "\n");
                auto &string = result.dead_strings[i];
                for (size_t j = 0; j < string.size(); j++) {
                    auto vtx = string[j];
                    vtx_list << agent_->GetState().VertexToText(vtx);
                    if (j != string.size() - 1) vtx_list << ' ';
                }
            }
            out << GtpSuccess(vtx_list.str());
        } else if (const auto input = spt.Find({"black_area",
                                                    "white_area",
                                                    "black_territory",
                                                    "white_territory"}, 1)) {
            bool counted = false;
            const bool is_black = (input->Get<>().find("black") != std::string::npos);
            const bool is_area = (input->Get<>().find("area") != std::string::npos);

            auto check_color = is_black == true ? kBlack : kWhite;
            const auto color = agent_->GetState().GetToMove();
            const auto board_size = agent_->GetState().GetBoardSize();
            const auto num_intersections = board_size * board_size;

            for (int idx = 0; idx < num_intersections; ++idx) {
                const auto x = idx % board_size;
                const auto y = idx / board_size;
                const auto vtx = agent_->GetState().GetVertex(x,y);

                // -1 ~ 1
                auto owner_val = result.root_ownership[idx];
                if (color == kWhite) {
                    owner_val = 0.f - owner_val;
                }

                static constexpr float kThreshold = 0.35f; // give the low threshold
                if ((is_black && owner_val >= kThreshold) ||
                        (!is_black && owner_val <= -kThreshold)) {
                    if (is_area || agent_->GetState().GetState(vtx) != check_color) {
                        vtx_list << agent_->GetState().VertexToText(vtx) << ' ';
                        counted = true;
                    }
                }
            }
            if (counted) {
                int pos = vtx_list.tellp();
                vtx_list.seekp(pos-1);
            }
            out << GtpSuccess(vtx_list.str());
        } else {
            out << GtpFail("invalid status type");
        }
    } else if (const auto res = spt.Find({"help", "list_commands"}, 0)) {
        auto list_commands = std::ostringstream{};
        auto idx = size_t{0};

        std::sort(std::begin(kGtpCommandsList), std::end(kGtpCommandsList));

        for (const auto &cmd : kGtpCommandsList) {
            list_commands << cmd;
            if (++idx != kGtpCommandsList.size()) list_commands << std::endl;
        }
        out << GtpSuccess(list_commands.str());
    } else if (const auto res = spt.Find("known_command", 0)) {
        auto cmd = std::string{};
        if (const auto input = spt.GetWord(1)) {
            cmd = input->Get<>();
        }
        auto ite = std::find(std::begin(kGtpCommandsList), std::end(kGtpCommandsList), cmd);
        if (ite != std::end(kGtpCommandsList)) {
            out << GtpSuccess("true");
        } else {
            out << GtpSuccess("false");
        }
    } else if (const auto res = spt.Find("sayuri-planes", 0)) {
        int symmetry = Symmetry::kIdentitySymmetry;

        if (const auto symm = spt.GetWord(1)) {
            symmetry = symm->Get<int>(-1);
        }

        if (symmetry <= 8 && symmetry >= 0) {
            out << GtpSuccess(
                Encoder::Get().GetPlanesString(
                    agent_->GetState(), symmetry, agent_->GetNetwork().GetVersion()));
        } else {
            out << GtpFail("symmetry must be from 0 to 7");
        }
    } else if (const auto res = spt.Find("sayuri-raw_nn", 0)) {
        int symmetry = Symmetry::kIdentitySymmetry;
        bool use_avg = false;

        if (const auto symm = spt.GetWord(1)) {
            if (symm->Get<>() == "avg" || symm->Get<>() == "average") {
                use_avg = true;
            } else {
                symmetry = symm->Get<int>(-1);
            }
        }
        if (symmetry <= 8 && symmetry >= 0) {
            auto ensemble = use_avg ? Network::kAverage : Network::kDirect;
            out << GtpSuccess(agent_->GetNetwork().GetOutputString(
                       agent_->GetState(), ensemble, Network::Query::Get().SetSymmetry(symmetry)));
        } else {
            out << GtpFail("symmetry must be from 0 to 7, or avg");
        }
    } else if (const auto res = spt.Find("sayuri-setoption", 0)) {
        std::string rep;
        bool success = ParseOption(spt, rep);
        if (success) {
            out << GtpSuccess("");
        } else {
            out << GtpFail(rep);
        }
    } else if (const auto res = spt.Find("netbench", 0)) {
        std::string rep;
        bool success = true;
        try {
            success = NetBench(spt, rep);
        } catch (const std::exception& e) {
            success = false;
            rep = Format("error: %s", e.what());
        }
        if (success) {
            out << GtpSuccess("");
        } else {
            out << GtpFail(rep);
        }
    } else if (const auto res = spt.Find("genbook", 0)) {
        auto sgf_file = std::string{};
        auto data_file = std::string{};

        if (const auto sgf = spt.GetWord(1)) {
            sgf_file = sgf->Get<>();
        }
        if (const auto data = spt.GetWord(2)) {
            data_file = data->Get<>();
        }

        if (!sgf_file.empty() && !data_file.empty()) {
            Book::Get().GenerateBook(sgf_file, data_file);
            out << GtpSuccess("");
        } else {
            out << GtpFail("file name is empty");
        }
    } else if (const auto res = spt.Find("genpatterns", 0)) {
        auto sgf_file = std::string{};
        auto data_file = std::string{};
        int min_count = 0;

        if (const auto sgf = spt.GetWord(1)) {
            sgf_file = sgf->Get<>();
        }
        if (const auto data = spt.GetWord(2)) {
            data_file = data->Get<>();
        }
        if (const auto mcount = spt.GetWord(3)) {
            min_count = mcount->Get<int>(min_count);
        }

        if (!sgf_file.empty() && !data_file.empty()) {
            MmTrainer::Get().Run(sgf_file, data_file, min_count);
            out << GtpSuccess("");
        } else {
            out << GtpFail("file name is empty");
        }

    } else if (const auto res = spt.Find("genopenings", 0)) {
        auto save_dir = std::string{};
        int num_sgfs = 0;
        int opening_moves = agent_->GetState().GetBoardSize() / 2;

        if (const auto dir = spt.GetWord(1)) {
            save_dir = dir->Get<>();
        }
        if (const auto num = spt.GetWord(2)) {
            num_sgfs = num->Get<int>(num_sgfs);
        }
        if (const auto num = spt.GetWord(3)) {
            opening_moves = num->Get<int>(opening_moves);
        }

        if (!save_dir.empty()) {
            try {
                TryCreateDirectory(save_dir);
            } catch (char * err) {
                (void) err;
            }

            const auto fair_result = agent_->GetSearch().Computation(400, Search::kForced);
            const auto fair_winrate = fair_result.root_eval;

            auto buf = std::vector<std::uint64_t>{};
            int games = 0;
            while (games < num_sgfs) {
                agent_->GetState().ClearBoard();
                for (int i = 0; i < opening_moves; ++i) {
                    const int vtx =
                        agent_->GetNetwork().GetVertexWithPolicy(
                            agent_->GetState(), 1.2f, false);
                    agent_->GetState().PlayMove(vtx);
                }
                const auto hash = agent_->GetState().GetHash();
                if (std::find(std::begin(buf), std::end(buf), hash) != std::end(buf)) {
                    continue;
                }

                const auto range = 0.05f;
                const auto result = agent_->GetSearch().Computation(400, Search::kForced);
                auto winrate_upper = fair_winrate;
                if (result.to_move != fair_result.to_move) {
                    winrate_upper = 1.f - winrate_upper;
                }
                winrate_upper += range/2.0f;

                if (result.root_eval > winrate_upper ||
                        result.root_eval < (1.f - winrate_upper)) {
                    continue;
                }
                for (int symm = Symmetry::kIdentitySymmetry; symm < Symmetry::kNumSymmetris; ++symm) {
                    buf.emplace_back(agent_->GetState().ComputeSymmetryHash(symm));
                }
                const auto sgf_name = std::to_string(games++) + ".sgf";
                const auto filename = ConcatPath(save_dir, sgf_name);
                Sgf::Get().ToFile(filename, agent_->GetState());
            }
            out << GtpSuccess("");
        } else {
            out << GtpFail("directory name is empty");
        }
    } else if (const auto res = spt.Find("debug_search", 0)) {
        int playouts = -1;

        if (const auto p = spt.GetWord(1)) {
            playouts = std::max(p->Get<int>(playouts), 1);
        }

        if (playouts > 0) {
            // clean current state
            agent_->GetSearch().ReleaseTree();
            agent_->GetNetwork().ClearCache();
            agent_->GetSearch().Computation(playouts, Search::kNullTag);
            out << GtpSuccess("done");
        } else {
            out << GtpFail("invalid playouts");
        }
    } else if (const auto res = spt.Find("debug_moves", 0)) {
        auto moves = std::vector<int>{};
        for (auto i = size_t{1}; i < spt.GetCount(); ++i) {
            auto move = spt.GetWord(i)->Get<>();
            moves.emplace_back(agent_->GetState().TextToVertex(move));
        }
        out << GtpSuccess(
                   agent_->GetSearch().GetDebugMoves(moves));
    } else if (const auto res = spt.Find("gogui-analyze_commands", 0)) {
        auto gogui_cmds = std::ostringstream{};

        gogui_cmds << "gfx/Win-Draw-Loss Rating/gogui-wdl_rating";
        gogui_cmds << "\ngfx/Policy Heatmap/gogui-policy_heatmap";
        gogui_cmds << "\ngfx/Policy Rating/gogui-policy_rating";
        gogui_cmds << "\ngfx/Target Policy Rating/gogui-target_policy_rating";
        gogui_cmds << "\ngfx/Ownership Heatmap/gogui-ownership_heatmap 0";
        gogui_cmds << "\ngfx/Ownership Influence/gogui-ownership_influence 0";
        gogui_cmds << "\ngfx/MCTS Ownership Heatmap/gogui-ownership_heatmap 400";
        gogui_cmds << "\ngfx/MCTS Ownership Influence/gogui-ownership_influence 400";
        gogui_cmds << "\ngfx/Book Rating/gogui-book_rating";
        gogui_cmds << "\ngfx/Gammas Heatmap/gogui-gammas_heatmap";
        gogui_cmds << "\ngfx/Ladder Map/gogui-ladder_map";

        out << GtpSuccess(gogui_cmds.str());
    } else if (const auto res = spt.Find("gogui-wdl_rating", 0)) {
        const auto result = agent_->GetNetwork().GetOutput(agent_->GetState(), Network::kNone);
        const auto board_size = result.board_size;
        const auto num_intersections = board_size * board_size;
        const auto ave_pol = 1.f / (float)num_intersections;

        auto first = true;
        auto wdl_rating = std::ostringstream{};

        for (int idx = 0; idx < num_intersections; ++idx) {
            const auto x = idx % board_size;
            const auto y = idx / board_size;
            const auto vtx = agent_->GetState().GetVertex(x,y);

            auto prob = result.probabilities[idx];
            if (prob > ave_pol) {
                if (agent_->GetState().PlayMove(vtx)) {
                    const auto next_result = agent_->GetNetwork().GetOutput(
                                                 agent_->GetState(), Network::kNone);

                    const float wdl = next_result.wdl_winrate;
                    if (!first) {
                        wdl_rating << '\n';
                    }
                    wdl_rating << GoguiLable(1.f - wdl, agent_->GetState().VertexToText(vtx));
                    first = false;

                    agent_->GetState().UndoMove();
                }
            }
        }

        out << GtpSuccess(wdl_rating.str());
    } else if (const auto res = spt.Find("gogui-policy_heatmap", 0)) {
        const auto result = agent_->GetNetwork().GetOutput(agent_->GetState(), Network::kNone);
        const auto board_size = result.board_size;
        const auto num_intersections = board_size * board_size;

        auto policy_map = std::ostringstream{};

        for (int idx = 0; idx < num_intersections; ++idx) {
            if (idx != 0) {
                policy_map << '\n';
            }

            const auto x = idx % board_size;
            const auto y = idx / board_size;
            const auto vtx = agent_->GetState().GetVertex(x,y);

            auto prob = result.probabilities[idx];
            if (prob > 0.0001f) {
                // highlight the probability
                prob = std::sqrt(prob);
            }

            policy_map << GoguiColor(prob, agent_->GetState().VertexToText(vtx));
        }

        out << GtpSuccess(policy_map.str());
    } else if (const auto res = spt.Find("gogui-policy_rating", 0)) {
        const auto result = agent_->GetNetwork().GetOutput(agent_->GetState(), Network::kNone);
        const auto board_size = result.board_size;
        const auto num_intersections = board_size * board_size;
        const auto ave_pol = 1.f / (float)num_intersections;

        auto policy_rating = std::ostringstream{};
        int max_idx = -1;

        for (int idx = 0; idx < num_intersections; ++idx) {
            const auto x = idx % board_size;
            const auto y = idx / board_size;
            const auto vtx = agent_->GetState().GetVertex(x,y);

            auto prob = result.probabilities[idx];
            if (prob > ave_pol) {
                if (max_idx < 0 ||
                        result.probabilities[max_idx] < prob) {
                    max_idx = idx;
                }

                policy_rating << '\n';
                policy_rating << GoguiLable(prob, agent_->GetState().VertexToText(vtx));
            }
        }

        auto policy_rating_var = std::ostringstream{};

        const auto x = max_idx % board_size;
        const auto y = max_idx / board_size;
        const auto max_vtx = agent_->GetState().GetVertex(x,y);

        if (agent_->GetState().GetToMove() == kBlack) {
            policy_rating_var << Format("VAR b %s", agent_->GetState().VertexToText(max_vtx).c_str());
        } else {
            policy_rating_var << Format("VAR w %s", agent_->GetState().VertexToText(max_vtx).c_str());
        }
        policy_rating_var << policy_rating.str();

        out << GtpSuccess(policy_rating_var.str());
    } else if (const auto res = spt.Find("gogui-target_policy_rating", 0)) {
        agent_->GetSearch().ReleaseTree();
        agent_->GetNetwork().ClearCache();
        auto result = agent_->GetSearch().Computation(GetOption<int>("playouts"), Search::kNullTag);
        const auto board_size = result.board_size;
        const auto num_intersections = board_size * board_size;
        const auto ave_pol = 1.f / (float)num_intersections;

        auto policy_rating = std::ostringstream{};
        int max_idx = -1;

        for (int idx = 0; idx < num_intersections; ++idx) {
            const auto x = idx % board_size;
            const auto y = idx / board_size;
            const auto vtx = agent_->GetState().GetVertex(x,y);

            auto prob = result.target_policy_dist[idx];
            if (prob > ave_pol) {
                if (max_idx < 0 ||
                        result.target_policy_dist[max_idx] < prob) {
                    max_idx = idx;
                }

                policy_rating << '\n';
                policy_rating << GoguiLable(prob, agent_->GetState().VertexToText(vtx));
            }
        }

        auto policy_rating_var = std::ostringstream{};

        const auto x = max_idx % board_size;
        const auto y = max_idx / board_size;
        const auto max_vtx = agent_->GetState().GetVertex(x,y);

        if (agent_->GetState().GetToMove() == kBlack) {
            policy_rating_var << Format("VAR b %s", agent_->GetState().VertexToText(max_vtx).c_str());
        } else {
            policy_rating_var << Format("VAR w %s", agent_->GetState().VertexToText(max_vtx).c_str());
        }
        policy_rating_var << policy_rating.str();

        out << GtpSuccess(policy_rating_var.str());
    } else if (const auto res = spt.Find("gogui-ownership_heatmap", 0)) {
        int playouts = 0;
        if (const auto p = spt.GetWord(1)) {
            playouts = p->Get<int>(playouts);
        }

        agent_->GetSearch().ReleaseTree();
        auto result = agent_->GetSearch().Computation(playouts, Search::kForced);

        const auto board_size = agent_->GetState().GetBoardSize();
        const auto num_intersections = board_size * board_size;
        const auto color = agent_->GetState().GetToMove();

        auto owner_map = std::ostringstream{};

        for (int idx = 0; idx < num_intersections; ++idx) {
            if (idx != 0) {
                owner_map << '\n';
            }

            const auto x = idx % board_size;
            const auto y = idx / board_size;
            const auto vtx = agent_->GetState().GetVertex(x,y);

            // map [-1 ~ 1] to [0 ~ 1]
            const auto owner_val = (result.root_ownership[idx] + 1.f) / 2.f;

            owner_map << GoguiGray(owner_val,
                                       agent_->GetState().VertexToText(vtx),
                                       color == kWhite);
        }
        out << GtpSuccess(owner_map.str());
    } else if (const auto res = spt.Find("gogui-ownership_influence", 0)) {
        int playouts = 0;
        if (const auto p = spt.GetWord(1)) {
            playouts = p->Get<int>(playouts);
        }

        agent_->GetSearch().ReleaseTree();
        auto result = agent_->GetSearch().Computation(playouts, Search::kForced);

        const auto board_size = agent_->GetState().GetBoardSize();
        const auto num_intersections = board_size * board_size;
        const auto color = agent_->GetState().GetToMove();

        auto owner_map = std::ostringstream{};
        owner_map << "INFLUENCE";

        for (int idx = 0; idx < num_intersections; ++idx) {
            const auto x = idx % board_size;
            const auto y = idx / board_size;
            const auto vtx = agent_->GetState().GetVertex(x,y);

            auto owner_val = result.root_ownership[idx];
            if (color == kWhite) {
                owner_val = -owner_val;
            }

            owner_map << Format(" %s %.1f",
                                    agent_->GetState().VertexToText(vtx).c_str(),
                                    owner_val);
        }

        out << GtpSuccess(owner_map.str());
    } else if (const auto res = spt.Find("gogui-book_rating", 0)) {
        const auto move_list = Book::Get().GetCandidateMoves(agent_->GetState());
        auto book_rating = std::ostringstream{};

        if (!move_list.empty()) {
            const auto vtx = move_list[0].second;
            if (agent_->GetState().GetToMove() == kBlack) {
                book_rating << Format("VAR b %s", agent_->GetState().VertexToText(vtx).c_str());
            } else {
                book_rating << Format("VAR w %s", agent_->GetState().VertexToText(vtx).c_str());
            }
        }

        for (int i = 0; i < (int)move_list.size(); ++i) {
            const auto prov = move_list[i].first;
            const auto vtx = move_list[i].second;

            book_rating << '\n';
            book_rating << GoguiLable(prov, agent_->GetState().VertexToText(vtx));
        }

        out << GtpSuccess(book_rating.str());
    } else if (const auto res = spt.Find("gogui-gammas_heatmap", 0)) {
        const auto board_size = agent_->GetState().GetBoardSize();
        const auto num_intersections = board_size * board_size;
        const auto color = agent_->GetState().GetToMove();

        std::vector<float> gammas = agent_->GetState().GetGammasPolicy(color);

        auto gammas_map = std::ostringstream{};
        for (int idx = 0; idx < num_intersections; ++idx) {
            if (idx != 0) {
                gammas_map << '\n';
            }

            const auto x = idx % board_size;
            const auto y = idx / board_size;
            const auto vtx = agent_->GetState().GetVertex(x,y);
            auto gnval = gammas[idx];

            if (gnval > 0.0001f) {
                // highlight
                gnval = std::sqrt(gnval);
            }
            gammas_map << GoguiColor(gnval, agent_->GetState().VertexToText(vtx));
        }
        out << GtpSuccess(gammas_map.str());
    } else if (const auto res = spt.Find("gogui-ladder_map", 0)) {
        const auto result = agent_->GetState().board_.GetLadderMap();
        const auto board_size = agent_->GetState().GetBoardSize();
        const auto num_intersections = board_size * board_size;

        auto ladder_map = std::ostringstream{};

        for (int idx = 0; idx < num_intersections; ++idx) {
            if (idx != 0) {
                ladder_map << '\n';
            }

            const auto x = idx % board_size;
            const auto y = idx / board_size;
            const auto vtx = agent_->GetState().GetVertex(x,y);

            float map_color = 0.f;

            if (result[idx] == LadderType::kLadderAtari) {
                map_color = 0.2f;
            } else if (result[idx] == LadderType::kLadderTake) {
                map_color = 0.4f;
            } else if (result[idx] == LadderType::kLadderEscapable) {
                map_color = 0.8f;
            } else if (result[idx] == LadderType::kLadderDeath) {
                map_color = 1.0f;
            }
            ladder_map << GoguiColor(map_color, agent_->GetState().VertexToText(vtx));
        }

        out << GtpSuccess(ladder_map.str());
    } else if (const auto res = spt.Find("gogui-rules_game_id", 0)) {
        out << GtpSuccess("Go");
    } else if (const auto res = spt.Find("gogui-rules_board", 0)) {
        const auto board_size = agent_->GetState().GetBoardSize();
        auto board_oss = std::ostringstream{};

        for (int y = board_size-1; y >= 0; --y) {
            for (int x = 0; x < board_size; ++x) {
                const auto s = agent_->GetState().GetState(x,y);
                if (s == kBlack) {
                    board_oss << "X";
                } else if (s == kWhite) {
                    board_oss << "O";
                } else if (s == kEmpty) {
                    board_oss << ".";
                }
                board_oss << " \n"[board_size == x+1];
            }
        }
        out << GtpSuccess(board_oss.str());
    } else if (const auto res = spt.Find("gogui-rules_board_size", 0)) {
        out << GtpSuccess(std::to_string(agent_->GetState().GetBoardSize()));
    } else if (const auto res = spt.Find("gogui-rules_legal_moves", 0)) {
        if (agent_->GetState().IsGameOver()) {
            out << GtpSuccess("");
        } else {
            const auto board_size = agent_->GetState().GetBoardSize();
            auto legal_list = std::vector<int>{kPass};

            for (int y = board_size-1; y >= 0; --y) {
                for (int x = 0; x < board_size; ++x) {
                    const auto vtx = agent_->GetState().GetVertex(x,y);
                    if (agent_->GetState().IsLegalMove(vtx)) {
                        legal_list.emplace_back(vtx);
                    }
                }
            }

            auto legal_oss = std::ostringstream{};
            for (auto v: legal_list) {
                legal_oss << agent_->GetState().VertexToText(v) << ' ';
            }
            out << GtpSuccess(legal_oss.str());
        }
    } else if (const auto res = spt.Find("gogui-rules_side_to_move", 0)) {
        if (agent_->GetState().GetToMove() == kBlack) {
            out << GtpSuccess("black");
        } else {
            out << GtpSuccess("white");
        }
    } else if (const auto res = spt.Find("gogui-rules_final_result", 0)) {
        auto score = agent_->GetState().GetFinalScore(kBlack);

        if (std::abs(score) < 1e-4f) {
            out << GtpSuccess("0");
        } else if (score < 0.f) {
            out << GtpSuccess(Format("W+%f", -score));
        } else {
            out << GtpSuccess(Format("B+%f", score));
        }
    } else {
        try_ponder = prev_pondering_;
        out << GtpFail("unknown command");
    }
    return out.str();
}

std::string GtpLoop::GtpSuccess(std::string response) {
    auto out = std::ostringstream{};
    auto prefix = std::string{"="};
    auto suffix = std::string{"\n\n"};

    while (!response.empty() && response[response.size() - 1] == '\n') {
        // remove all end line
        response.resize(response.size() - 1);
    }

    out << prefix;
    if (curr_id_ >= 0) {
        out << curr_id_ << " ";
    } else {
        out << " ";
    }
    out << response << suffix;

    return out.str();
}

std::string GtpLoop::GtpFail(std::string response) {
    auto out = std::ostringstream{};
    auto prefix = std::string{"? "};
    auto suffix = std::string{"\n\n"};

    while (!response.empty() && response[response.size() - 1] == '\n') {
        // remove all end line
        response.resize(response.size() - 1);
    }
    out << prefix << response << suffix;

    return out.str();
}

AnalysisConfig GtpLoop::ParseAnalysisConfig(Splitter &spt, int &color) {
    AnalysisConfig config;

    config.interval = 0;
    auto main = spt.GetWord(0)->Get<>();

    if (main.find("sayuri") == 0) {
        config.output_format = AnalysisConfig::kSayuri;
    } else if (main.find("kata") == 0) {
        config.output_format = AnalysisConfig::kKata;
    } else {
        config.output_format = AnalysisConfig::kLeela;
    }

    int curr_idx = 1;
    while (true) {
        auto token = spt.GetWord(curr_idx++);
        if (!token) {
            break;
        }

        if (token->IsDigit()) {
            config.interval = token->Get<int>();
            continue;
        }

        if (token->Lower() == "b" || token->Lower() == "black") {
            color = kBlack;
            continue;
        }

        if (token->Lower() == "w" || token->Lower() == "white") {
            color = kWhite;
            continue;
        }

        if (token->Lower() == "interval") {
            if (auto interval_token = spt.GetWord(curr_idx)) {
                if (interval_token->IsDigit()) {
                    config.interval = interval_token->Get<int>();
                    curr_idx += 1;
                }
            }
            continue;
        }

        if (token->Lower() == "reuse") {
            if (auto true_token = spt.GetWord(curr_idx)) {
                if (true_token->Lower() == "true") {
                    config.use_reuse_label = true;
                    config.reuse_tree = true;
                    curr_idx += 1;
                } else if (true_token->Lower() == "false") {
                    config.use_reuse_label = true;
                    config.reuse_tree = false;
                    curr_idx += 1;
                } 
            }
            continue;
        }

        if (token->Lower() == "playouts") {
            if (auto interval_token = spt.GetWord(curr_idx)) {
                if (interval_token->IsDigit()) {
                    config.use_playouts_label = true;
                    config.playouts = interval_token->Get<int>();
                    curr_idx += 1;
                }
            }
            continue;
        }

        if (token->Lower() == "ownership") {
            if (auto true_token = spt.GetWord(curr_idx)) {
                if (true_token->Lower() == "true") {
                    config.ownership = true;
                    curr_idx += 1;
                }
            }
            continue;
        }

        if (token->Lower() == "movesownership") {
            if (auto true_token = spt.GetWord(curr_idx)) {
                if (true_token->Lower() == "true") {
                    config.moves_ownership = true;
                    curr_idx += 1;
                }
            }
            continue;
        }

        if (token->Lower() == "minmoves") {
            // Current the analysis mode do not support this tag.
            if (auto num_token = spt.GetWord(curr_idx)) {
                if (num_token->IsDigit()) {
                    config.min_moves = num_token->Get<int>();
                    curr_idx += 1;
                }
            }
            continue;
        }

        if (token->Lower() == "maxmoves") {
            if (auto num_token = spt.GetWord(curr_idx)) {
                if (num_token->IsDigit()) {
                    config.max_moves = num_token->Get<int>();
                    curr_idx += 1;
                }
            }
            continue;
        }

        using MoveToAvoid = AnalysisConfig::MoveToAvoid;

        if (token->Lower() == "avoid" || token->Lower() == "allow") {
            int moves_color = kInvalid;
            int moves_movenum = -1;
            auto moves = std::vector<int>{};

            if (auto color_token = spt.GetWord(curr_idx)) {
                moves_color = agent_->GetState().TextToColor(color_token->Lower());
                curr_idx += 1;
            }
            if (auto moves_token = spt.GetWord(curr_idx)) {
                std::istringstream movestream(moves_token->Get<>());
                while (!movestream.eof()) {
                    std::string textmove;
                    getline(movestream, textmove, ',');
                    auto sepidx = textmove.find_first_of(':');
                    if (sepidx != std::string::npos) {
                        // Do not support this format.
                    } else {
                        auto move = agent_->GetState().TextToVertex(textmove);
                        if (move != kNullVertex) {
                            moves.push_back(move);
                        }
                    }
                }
                curr_idx += 1;
            }
            if (auto num_token = spt.GetWord(curr_idx)) {
                if (num_token->IsDigit()) {
                    moves_movenum = num_token->Get<int>();
                    curr_idx += 1;
                }
            }

            if (moves_color != kInvalid && moves_movenum >= 0) {
                for (const auto vtx : moves) {
                    MoveToAvoid avoid_move;
                    avoid_move.vertex     = vtx;
                    avoid_move.color      = moves_color;
                    avoid_move.until_move = moves_movenum +
                                                agent_->GetState().GetMoveNumber() - 1;
                    if (avoid_move.Valid()) {
                        if (token->Lower() == "allow") {
                            config.allow_moves.emplace_back(avoid_move);
                        } else {
                            config.avoid_moves.emplace_back(avoid_move);
                        }
                    }
                }
            }
            continue;
        }
    }

    return config;
}

bool GtpLoop::ParseOption(Splitter &spt, std::string &rep) {
    const auto GetLowerString = [](std::string val) -> std::string {
        for (auto & c: val) {
            c = std::tolower(c);
        }
        return val;
    };

    int name_idx = -1;
    int value_idx = name_idx - 1;
    std::string name, value;

    if (const auto res = spt.Find("name")) {
        name_idx = res->Index();
    }
    if (const auto res = spt.Find("value")) {
        value_idx = res->Index();
    }
    if (value_idx < name_idx) {
        // no name tag, no value tag or value tag is before name
        rep = "invalid tag";
        return false;
    }

    if (const auto res = spt.GetSlice(name_idx+1, value_idx)) {
        name = res->Get<>();
    }
    if (const auto res = spt.GetSlice(value_idx+1)) {
        value = res->Get<>();
    }
    if (name.empty() || value.empty()) {
        // no name string or no value string
        rep = "name or value is empty";
        return false;
    }

    Parameters * param = agent_->GetSearch().GetParams();
    try {
        if (name == "playouts") {
            param->playouts = std::max(0, std::stoi(value));
        } else if (name == "reuse tree") {
            if (GetLowerString(value) == "true") {
                param->reuse_tree = true;
            } else if (GetLowerString(value) == "false") {
                param->reuse_tree = false;
            } else {
                rep = "invalid value";
                return false;
            }
        } else if (name == "pondering") {
            if (GetLowerString(value) == "true") {
                param->ponder = true;
            } else if (GetLowerString(value) == "false") {
                param->ponder = false;
            } else {
                rep = "invalid value";
                return false;
            }
        } else if (name == "resign threshold") {
            param->resign_threshold =
                std::min(1.f, std::max(0.f, std::stof(value)));
        } else if (name == "scoring rule") {
            if (GetLowerString(value) == "territory") {
                agent_->GetState().SetRule(kTerritory);
            } else if (GetLowerString(value) == "area") {
                agent_->GetState().SetRule(kArea);
            } else {
                rep = "invalid rule";
                return false;
            }
        }  else if (name == "threads") {
            agent_->SetThreads(std::stoi(value));
        } else if (name == "batch size") {
            agent_->SetBatchSize(std::stoi(value));
        }  else if (name == "batch size") {
        } else {
            rep = "invalid option name";
            return false;
        }
    } catch (const std::exception& e) {
        rep = "invalid option value: " + std::string{e.what()};
        return false;
    }
    return true;
}

bool GtpLoop::NetBench(Splitter &spt, std::string &rep) {
    Parameters * param = agent_->GetSearch().GetParams();
    const auto orig_batch = param->batch_size;
    auto optrep = std::string{};
    auto batchsize_list = std::vector<int>{};
    float timelimit = 10.0f;

    int curr_idx = 1;
    while (true) {
        auto token = spt.GetWord(curr_idx++);
        if (!token) {
            break;
        }
        if (token->Lower() == "timelimit") {
           if (auto time_token = spt.GetWord(curr_idx)) {
                timelimit = time_token->Get<float>();
                curr_idx += 1;
            }
            continue;
        }
        if (token->Lower() == "batchsize") {
            while (auto bs_token = spt.GetWord(curr_idx)) {
                if (!bs_token) {
                    break;
                }
                if (!bs_token->IsDigit()) {
                    break;
                }
                batchsize_list.emplace_back(bs_token->Get<int>());
                curr_idx += 1;
            }
            continue;
        }
        
    }

    if (batchsize_list.empty()) {
        batchsize_list.emplace_back(orig_batch);
    }
    std::sort(std::begin(batchsize_list), std::end(batchsize_list));
    batchsize_list.erase(
        std::unique(std::begin(batchsize_list), std::end(batchsize_list)),
        std::end(batchsize_list));

    if (batchsize_list.empty()) {
        rep = "invalid batch size";
        return false;
    }
    if (timelimit <= 0.0f) {
        rep = "time limit should be greater than 0";
        return false;
    }

    std::atomic<bool> running{false};
    const auto Worker = [&, this]() -> void {
        while (running.load(std::memory_order_relaxed)) {
            agent_->GetNetwork().GetOutput(
                agent_->GetState(), Network::kRandom,
                Network::Query::Get().SetCache(false));
        }
    };

    for (int batch_size: batchsize_list) {
        const int threads = batch_size * 2;
        agent_->SetBatchSize(batch_size);
        agent_->GetNetwork().ResetNumQueries();

        Timer timer;
        timer.Clock();

        auto group = ThreadGroup<void>(&ThreadPool::Get("search", threads));
        running.store(true, std::memory_order_relaxed);

        for (int i = 0; i < threads; ++i) {
            group.AddTask(Worker);
        }
        while (timer.GetDuration() < timelimit) {
            std::this_thread::yield();
        }
        running.store(false, std::memory_order_relaxed);
        group.WaitToJoin();
        const auto elapsed = timer.GetDuration();
        const auto num_nn_queries = agent_->GetNetwork().GetNumQueries();

        LOGGING << Format(
                       "batch size= %d -> %d evals | %.2f evals/s\n",
                       batch_size,
                       num_nn_queries,
                       num_nn_queries/elapsed);
    }

    if (orig_batch != param->batch_size) {
        agent_->SetBatchSize(orig_batch);
    }
    return true;
}
