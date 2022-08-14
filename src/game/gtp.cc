#include "game/gtp.h"
#include "game/sgf.h"
#include "game/commands_list.h"
#include "utils/log.h"
#include "utils/komi.h"
#include "utils/format.h"
#include "utils/gogui_helper.h"
#include "version.h"

#include "data/supervised.h"
#include "neural/encoder.h"

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <array>

void GtpLoop::Loop() {
    while (true) {
        auto input = std::string{};
        if (std::getline(std::cin, input)) {

            auto parser = CommandParser(input);
            WRITING << ">>" << ' ' << input << std::endl;

            curr_id_ = -1;

            // check the command id here
            if (const auto toke = parser.GetCommand(0)) {
                auto toke_str = toke->Get<std::string>();
                bool is_digit = true;

                for (char c : toke_str) {
                    is_digit &= isdigit(c);
                }
                if (is_digit) {
                    curr_id_ = toke->Get<int>();
                    parser.RemoveCommand(toke->Index());
                }
            }

            if (!parser.Valid()) {
                continue;
            }

            auto out = std::string{};
            auto stop = false;
            auto try_ponder = false;

            if (parser.GetCount() == 1 && parser.Find("quit")) {
                agent_->Quit();
                out = GTPSuccess("");
                stop = true;
            }

            if (out.empty()) {
                out = Execute(parser, try_ponder);
            }

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

std::string GtpLoop::Execute(CommandParser &parser, bool &try_ponder) {
    if (!agent_) {
        return std::string{};
    }

    auto out = std::ostringstream{};

    if (const auto res = parser.Find("protocol_version", 0)) {
        out << GTPSuccess(std::to_string(kProtocolVerion));
    } else if (const auto res = parser.Find("name", 0)) {
        out << GTPSuccess(GetProgramName());
    } else if (const auto res = parser.Find("version", 0)) {
        out << GTPSuccess(GetProgramVersion());
    } else if (const auto res = parser.Find("showboard", 0)) {
        agent_->GetState().ShowBoard();
        out << GTPSuccess("");
    } else if (const auto res = parser.Find("boardsize", 0)){
        int bsize = -1;
        if (const auto input = parser.GetCommand(1)) {
            bsize = input->Get<int>();
        }

        if (bsize <= kBoardSize &&
                bsize <= kMaxGTPBoardSize &&
                bsize >= kMinGTPBoardSize) {
            agent_->GetState().Reset(bsize, agent_->GetState().GetKomi());
            agent_->GetNetwork().Reload(bsize);
            out << GTPSuccess("");
        } else {
            out << GTPFail("invalid board size");
        }
    } else if (const auto res = parser.Find("clear_board", 0)){
        agent_->GetState().ClearBoard();
        agent_->GetNetwork().ClearCache();
        out << GTPSuccess("");
    } else if (const auto res = parser.Find("komi", 0)) {
        auto komi = agent_->GetState().GetKomi();
        if (const auto input = parser.GetCommand(1)) {
            komi = input->Get<float>();
            out << GTPSuccess("");
        } else {
            out << GTPFail("invalid komi");
        }
        agent_->GetState().SetKomi(komi);
    } else if (const auto res = parser.Find("play", 0)) {
        const auto end = parser.GetCount() < 3 ? parser.GetCount() : 3;
        auto cmd = std::string{};

        if (const auto input = parser.GetSlice(1, end)) {
            cmd = input->Get<std::string>();
        }
        if (agent_->GetState().PlayTextMove(cmd)) {
            out << GTPSuccess("");
        } else {
            out << GTPFail("invalid play");
        }
    } else if (const auto res = parser.Find("fixed_handicap", 0)) {
        auto handicap = -1;
        if (const auto input = parser.GetCommand(1)) {
            agent_->GetState().ClearBoard();
            handicap = input->Get<int>();
        }
        if (handicap >= 1 &&
                agent_->GetState().SetFixdHandicap(handicap)) {
            out << GTPSuccess("");
        } else {
            out << GTPFail("invalid handicap");
        }
    } else if (const auto res = parser.Find("place_free_handicap", 0)) {
        auto handicaps = -1;
        if (const auto input = parser.GetCommand(1)) {
            handicaps = input->Get<int>();
        }

        int max_handicaps = agent_->GetState().GetNumIntersections() / 4;
        if (handicaps >= 1 && handicaps <= max_handicaps) {
            agent_->GetState().ClearBoard();
            agent_->GetState().SetHandicap(handicaps);
        } else {
            handicaps = -1; // disable handicap
        }

        auto stone_list = std::vector<int>{};
        for (int i = 0; i < handicaps; ++i) {
            const int vtx = agent_->GetNetwork().GetBestPolicyVertex(agent_->GetState(), false);
            agent_->GetState().AppendMove(vtx, kBlack);
            stone_list.emplace_back(vtx);
        }

        if (!stone_list.empty()) {
            auto vtx_list = std::ostringstream{};
            for (size_t i = 0; i < stone_list.size(); i++) {
                auto vtx = stone_list[i];
                vtx_list << agent_->GetState().VertexToText(vtx);
                if (i != stone_list.size() - 1) vtx_list << ' ';
            }
            out << GTPSuccess(vtx_list.str());
        } else {
            out << GTPFail("invalid handicap");
        }
    } else if (const auto res = parser.Find("set_free_handicap", 0)) {
        auto movelist = std::vector<std::string>{};
        for (auto i = size_t{1}; i < parser.GetCount(); ++i) {
            movelist.emplace_back(parser.GetCommand(i)->Get<std::string>());
        }
        if (agent_->GetState().SetFreeHandicap(movelist)) {
            out << GTPSuccess("");
        } else {
            out << GTPFail("invalid handicap");
        }
    } else if (const auto res = parser.Find("loadsgf", 0)) {
        auto movenum = 9999;
        auto filename = std::string{};
        if (const auto input = parser.GetCommand(1)) {
            filename = input->Get<std::string>();
        }
        if (const auto input = parser.GetCommand(2)) {
            movenum = input->Get<int>();
        }
        try {
            agent_->GetState() = Sgf::Get().FromFile(filename, movenum);
            out << GTPSuccess("");
        } catch (const char *err) {
            out << GTPFail(Format("invalid SGF file, cause %s.", err));
        }
    } else if (const auto res = parser.Find("printsgf", 0)) {
        auto filename = std::string{};
        if (const auto input = parser.GetCommand(1)) {
            filename = input->Get<std::string>();
        }
        if (filename.empty()) {
            out << GTPSuccess(Sgf::Get().ToString(agent_->GetState()));
        } else {
            out << GTPSuccess("");
            Sgf::Get().ToFile(filename, agent_->GetState());
        }
    } else if (const auto res = parser.Find("final_score", 0)) {
        auto result = agent_->GetSearch().Computation(400, 0, Search::kForced);
        auto color = agent_->GetState().GetToMove();
        auto final_score = result.root_final_score;

        final_score = AdjustKomi<float>(final_score);
        if (std::abs(final_score) < 1e-4f) {
            color = kEmpty;
        } else if (final_score < 0.f) {
            final_score = -final_score;
            color = !color;
        }

        auto ss = std::ostringstream{};
        if (color == kEmpty) {
            ss << "draw";
        } else if (color == kBlack) {
            ss << "b+" << final_score;
        } else if (color == kWhite) {
            ss << "w+" << final_score;
        }
        out << GTPSuccess(ss.str());
    } else if (const auto res = parser.Find("genmove", 0)) {
        auto color = agent_->GetState().GetToMove();
        if (const auto input = parser.GetCommand(1)) {
            auto color_str = input->Get<std::string>();
            if (color_str == "b" || color_str == "B" || color_str == "black") {
                color = kBlack;
            } else if (color_str == "w" || color_str == "W" || color_str == "white") {
                color = kWhite;
            }
        }
        agent_->GetState().SetToMove(color);
        auto move = agent_->GetSearch().ThinkBestMove();
        agent_->GetState().PlayMove(move);
        out << GTPSuccess(agent_->GetState().VertexToText(move));
        try_ponder = true;
    } else if (const auto res = parser.Find("kgs-game_over", 0)) {
        agent_->GetNetwork().ClearCache();
        out << GTPSuccess("");
    } else if (const auto res = parser.Find("kgs-chat", 0)) {
        auto type = std::string{};
        auto name = std::string{};
        auto message = std::string{};
        if (parser.GetCount() < 3) {
            out << GTPFail("invalid chat settings");
        } else {
            type = parser.GetCommand(1)->Get<std::string>();
            name = parser.GetCommand(2)->Get<std::string>();
            message = parser.GetCommands(3)->Get<std::string>();
            out << GTPSuccess("I'm a go bot, not a chat bot.");
        }
    } else if (const auto res = parser.Find({"analyze", "lz-analyze", "sayuri-analyze"}, 0)) {
        auto color = agent_->GetState().GetToMove();
        auto interval = 100; // one second
        if (const auto input = parser.GetCommand(1)) {
            auto color_str = input->Get<std::string>();
            if (color_str == "b" || color_str == "B" || color_str == "black") {
                color = kBlack;
            } else if (color_str == "w" || color_str == "W" || color_str == "white") {
                color = kWhite;
            }
        }
        if (const auto input = parser.GetCommand(2)) {
            interval = input->Get<int>();
        }
        if (curr_id_ >= 0) {
            DUMPING << "=" << curr_id_ << "\n";
        } else {
            DUMPING << "=\n";
        }
        // The analyze is same as lz-analyze.
        bool use_sayuri = res->Get<std::string>() == "sayuri-analyze";

        agent_->GetState().SetToMove(color);
        agent_->GetSearch().Analyze(interval, true, use_sayuri);
        DUMPING << "\n";
    } else if (const auto res = parser.Find({"genmove_analyze", "lz-genmove_analyze", "sayuri-genmove_analyze"}, 0)) {
        auto color = agent_->GetState().GetToMove();
        auto interval = 100; // one second
        if (const auto input = parser.GetCommand(1)) {
            auto color_str = input->Get<std::string>();
            if (color_str == "b" || color_str == "B" || color_str == "black") {
                color = kBlack;
            } else if (color_str == "w" || color_str == "W" || color_str == "white") {
                color = kWhite;
            }
        }
        if (const auto input = parser.GetCommand(2)) {
            interval = input->Get<int>();
        }
        if (curr_id_ >= 0) {
            DUMPING << "=" << curr_id_ << "\n";
        } else {
            DUMPING << "=\n";
        }
        // The genmove_analyze is same as lz-genmove_analyze.
        bool use_sayuri = res->Get<std::string>() == "sayuri-genmove_analyze";

        agent_->GetState().SetToMove(color);
        auto move = agent_->GetSearch().Analyze(interval, false, use_sayuri);
        agent_->GetState().PlayMove(move);
        DUMPING << "play " << agent_->GetState().VertexToText(move) << "\n\n";
        try_ponder = true;
    } else if (const auto res = parser.Find("undo", 0)) {
        if (agent_->GetState().UndoMove()) {
            out << GTPSuccess("");
        } else {
            out << GTPFail("can't do undo move");
        }
    } else if (const auto res = parser.Find("kgs-time_settings", 0)) {
        // none, absolute, byoyomi, or canadian
        int main_time = 0, byo_yomi_time = 0, byo_yomi_stones = 0, byo_yomi_periods = 0;
        bool success = true;

        if (const auto res = parser.Find("none", 1)) {
            // infinite time
            main_time = byo_yomi_time = byo_yomi_stones = byo_yomi_periods;
        } else if (const auto res = parser.Find("absolute", 1)) {
            main_time = parser.GetCommand(2)->Get<int>();
        } else if (const auto res = parser.Find("canadian", 1)) {
            main_time = parser.GetCommand(2)->Get<int>();
            byo_yomi_time = parser.GetCommand(3)->Get<int>();
            byo_yomi_stones = parser.GetCommand(4)->Get<int>();
        } else if (const auto res = parser.Find("byoyomi", 1)) {
            main_time = parser.GetCommand(2)->Get<int>();
            byo_yomi_time = parser.GetCommand(3)->Get<int>();
            byo_yomi_periods = parser.GetCommand(4)->Get<int>();
        } else {
            success = false;
        }
        if (success) {
            agent_->GetSearch().TimeSettings(main_time, byo_yomi_time,
                                                 byo_yomi_stones, byo_yomi_periods);
            out << GTPSuccess("");
        } else {
            out << GTPFail("invalid time settings");
        }

    } else if (const auto res = parser.Find("time_settings", 0)) {
        int main_time = -1, byo_yomi_time = -1, byo_yomi_stones = -1;

        if (const auto input = parser.GetCommand(1)) {
            main_time = input->Get<int>();
        }
        if (const auto input = parser.GetCommand(2)) {
            byo_yomi_time = input->Get<int>();
        }
        if (const auto input = parser.GetCommand(3)) {
            byo_yomi_stones = input->Get<int>();
        }

        if (main_time == -1 || byo_yomi_time == -1 || byo_yomi_stones == -1) {
            out << GTPFail("invalid time settings");
        } else {
            agent_->GetSearch().TimeSettings(main_time, byo_yomi_time, byo_yomi_stones, 0);
            out << GTPSuccess("");
        }
    } else if (const auto res = parser.Find("time_left", 0)) {
        int color = kInvalid, time = -1, stones = -1;

        if (const auto input = parser.GetCommand(1)) {
            auto color_str = input->Get<std::string>();
            if (color_str == "b" || color_str == "B" || color_str == "black") {
                color = kBlack;
            } else if (color_str == "w" || color_str == "W" || color_str == "white") {
                color = kWhite;
            }
        }
        if (const auto input = parser.GetCommand(2)) {
            time = input->Get<int>();
        }
        if (const auto input = parser.GetCommand(3)) {
            stones = input->Get<int>();
        }

        if (color == kInvalid || time == -1 || stones == -1) {
            out << GTPFail("invalid time settings");
        } else {
            agent_->GetSearch().TimeLeft(color, time, stones);
            out << GTPSuccess("");
        }
    } else if (const auto res = parser.Find("final_status_list", 0)) {
        auto result = agent_->GetSearch().Computation(400, 0, Search::kForced);
        auto vtx_list = std::ostringstream{};

        if (const auto input = parser.Find("alive", 1)) {
            for (size_t i = 0; i < result.alive_strings.size(); i++) {
                vtx_list << (i == 0 ? "" : "\n");
                auto &string = result.alive_strings[i];
                for (size_t j = 0; j < string.size(); j++) {
                    auto vtx = string[j];
                    vtx_list << agent_->GetState().VertexToText(vtx);
                    if (j != string.size() - 1) vtx_list << ' ';
                }
            }
            out << GTPSuccess(vtx_list.str());
        } else if (const auto input = parser.Find("dead", 1)) {
             for (size_t i = 0; i < result.dead_strings.size(); i++) {
                vtx_list << (i == 0 ? "" : "\n");
                auto &string = result.dead_strings[i];
                for (size_t j = 0; j < string.size(); j++) {
                    auto vtx = string[j];
                    vtx_list << agent_->GetState().VertexToText(vtx);
                    if (j != string.size() - 1) vtx_list << ' ';
                }
            }
            out << GTPSuccess(vtx_list.str());
        } else {
            out << GTPFail("invalid status type");
        }
    } else if (const auto res = parser.Find({"help", "list_commands"}, 0)) {
        auto list_commands = std::ostringstream{};
        auto idx = size_t{0};

        std::sort(std::begin(kGtpCommandsList), std::end(kGtpCommandsList));

        for (const auto &cmd : kGtpCommandsList) {
            list_commands << cmd;
            if (++idx != kGtpCommandsList.size()) list_commands << std::endl;
        }
        out << GTPSuccess(list_commands.str());
    } else if (const auto res = parser.Find("known_command", 0)) {
        auto cmd = std::string{};
        if (const auto input = parser.GetCommand(1)) {
            cmd = input->Get<std::string>();
        }
        auto ite = std::find(std::begin(kGtpCommandsList), std::end(kGtpCommandsList), cmd);
        if (ite != std::end(kGtpCommandsList)) {
            out << GTPSuccess("true");
        } else {
            out << GTPSuccess("false");
        }
    } else if (const auto res = parser.Find({"supervised", "sayuri-supervised"}, 0)) {
        auto sgf_file = std::string{};
        auto data_file = std::string{};

        if (const auto sgf = parser.GetCommand(1)) {
            sgf_file = sgf->Get<std::string>();
        }
        if (const auto data = parser.GetCommand(2)) {
            data_file = data->Get<std::string>();
        }

        if (!sgf_file.empty() && !data_file.empty()) {
            bool is_general = res->Get<std::string>() != "sayuri-supervised";

            Supervised::Get().FromSgfs(is_general, sgf_file, data_file);
            out << GTPSuccess("");
        } else {
            out << GTPFail("file name is empty");
        }
    } else if (const auto res = parser.Find("planes", 0)) {
        int symmetry = Symmetry::kIdentitySymmetry;

        if (const auto symm = parser.GetCommand(1)) {
            symmetry = symm->Get<int>();
        }

        if (symmetry <= 8 && symmetry >= 0) {
            out << GTPSuccess(Encoder::Get().GetPlanesString(agent_->GetState(), symmetry));
        } else {
            out << GTPFail("symmetry must be from 0 to 7");
        }
    } else if (const auto res = parser.Find("raw-nn", 0)) {
        int symmetry = Symmetry::kIdentitySymmetry;

        if (const auto symm = parser.GetCommand(1)) {
            symmetry = symm->Get<int>();
        }

        if (symmetry <= 8 && symmetry >= 0) {
            out << GTPSuccess(agent_->GetNetwork().GetOutputString(agent_->GetState(), Network::kDirect, symmetry));   
        } else {
            out << GTPFail("symmetry must be from 0 to 7");
        }
    } else if (const auto res = parser.Find("genbook", 0)) {
        auto sgf_file = std::string{};
        auto data_file = std::string{};

        if (const auto sgf = parser.GetCommand(1)) {
            sgf_file = sgf->Get<std::string>();
        }
        if (const auto data = parser.GetCommand(2)) {
            data_file = data->Get<std::string>();
        }

        if (!sgf_file.empty() && !data_file.empty()) {
            Book::Get().GenerateBook(sgf_file, data_file);
            out << GTPSuccess("");
        } else {
            out << GTPFail("file name is empty");
        }
    } else if (const auto res = parser.Find("gogui-analyze_commands", 0)) {
        auto gogui_cmds = std::ostringstream{};

        gogui_cmds << "gfx/Win-Draw-Loss Rating/gogui-wdl_rating";
        gogui_cmds << "\ngfx/Policy Heatmap/gogui-policy_heatmap";
        gogui_cmds << "\ngfx/Policy Rating/gogui-policy_rating";
        gogui_cmds << "\ngfx/Ownership Heatmap/gogui-ownership_heatmap 0";
        gogui_cmds << "\ngfx/Ownership Influence/gogui-ownership_influence 0";
        gogui_cmds << "\ngfx/MCTS Ownership Heatmap/gogui-ownership_heatmap 400";
        gogui_cmds << "\ngfx/MCTS Ownership Influence/gogui-ownership_influence 400";
        gogui_cmds << "\ngfx/Book Rating/gogui-book_rating";
        gogui_cmds << "\ngfx/Gammas Heatmap/gogui-gammas_heatmap";

        out << GTPSuccess(gogui_cmds.str());
    } else if (const auto res = parser.Find("gogui-wdl_rating", 0)) {
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

        out << GTPSuccess(wdl_rating.str());
    } else if (const auto res = parser.Find("gogui-policy_heatmap", 0)) {
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

        out << GTPSuccess(policy_map.str());
    } else if (const auto res = parser.Find("gogui-policy_rating", 0)) {
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

        out << GTPSuccess(policy_rating_var.str());
    } else if (const auto res = parser.Find("gogui-ownership_heatmap", 0)) {
        int playouts = 0;
        if (const auto p = parser.GetCommand(1)) {
            playouts = p->Get<int>();
        }

        agent_->GetSearch().ReleaseTree();
        auto result = agent_->GetSearch().Computation(playouts, 0, Search::kForced);

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
        out << GTPSuccess(owner_map.str());
    } else if (const auto res = parser.Find("gogui-ownership_influence", 0)) {
        int playouts = 0;
        if (const auto p = parser.GetCommand(1)) {
            playouts = p->Get<int>();
        }

        agent_->GetSearch().ReleaseTree();
        auto result = agent_->GetSearch().Computation(playouts, 0, Search::kForced);

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

        out << GTPSuccess(owner_map.str());
    } else if (const auto res = parser.Find("gogui-book_rating", 0)) {
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

        out << GTPSuccess(book_rating.str());
    } else if (const auto res = parser.Find("gogui-gammas_heatmap", 0)) {
        const auto board_size = agent_->GetState().GetBoardSize();
        const auto num_intersections = board_size * board_size;
        const auto color = agent_->GetState().GetToMove();

        std::vector<float> gammas;
        for (int idx = 0; idx < num_intersections; ++idx) {
            const auto x = idx % board_size;
            const auto y = idx / board_size;
            const auto vtx = agent_->GetState().GetVertex(x,y);
            gammas.emplace_back(agent_->GetState().GetGammaValue(vtx, color));
        }
        float max_gamma = *std::max_element(std::begin(gammas), std::end(gammas));

        auto gammas_map = std::ostringstream{};
        for (int idx = 0; idx < num_intersections; ++idx) {
            if (idx != 0) {
                gammas_map << '\n';
            }

            const auto x = idx % board_size;
            const auto y = idx / board_size;
            const auto vtx = agent_->GetState().GetVertex(x,y);
            const auto gnval = gammas[idx] / max_gamma;
            gammas_map << GoguiColor(gnval, agent_->GetState().VertexToText(vtx));
        }
        out << GTPSuccess(gammas_map.str());
    } else {
        out << GTPFail("unknown command");
    }
    return out.str();
}

std::string GtpLoop::GTPSuccess(std::string response) {
    auto out = std::ostringstream{};
    auto prefix = std::string{"="};
    auto suffix = std::string{"\n\n"};

    out << prefix;
    if (curr_id_ >= 0) {
        out << curr_id_ << " ";
    } else {
        out << " ";
    }
    out << response << suffix;

    return out.str();
}

std::string GtpLoop::GTPFail(std::string response) {
    auto out = std::ostringstream{};
    auto prefix = std::string{"? "};
    auto suffix = std::string{"\n\n"};

    out << prefix << response << suffix;

    return out.str();
}
