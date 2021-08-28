#include "game/gtp.h"
#include "game/sgf.h"
#include "game/commands_list.h"
#include "utils/log.h"
#include "utils/komi.h"

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

            if (!parser.Valid()) {
                continue;
            }

            if (parser.GetCount() == 1 && parser.Find("quit")) {
                agent_->Quit();
                LOGGING << GTPSuccess("");
                break;
            }

            auto out = Execute(parser);
            LOGGING << out;
        }
    }
}

std::string GtpLoop::Execute(CommandParser &parser) {
    if (!agent_) {
        return std::string{};
    }

    auto out = std::ostringstream{};

    if (const auto res = parser.Find("protocol_version", 0)) {
        out << GTPSuccess(std::to_string(kProtocolVerion));
    } else if (const auto res = parser.Find("name", 0)) {
        out << GTPSuccess(kProgram);
    } else if (const auto res = parser.Find("version", 0)) {
        out << GTPSuccess(kVersion);
    } else if (const auto res = parser.Find("showboard", 0)) {
        agent_->GetState().ShowBoard();
        out << GTPSuccess("");
    } else if (const auto res = parser.Find("boardsize", 0)){
        auto bsize = agent_->GetState().GetBoardSize();
        if (const auto input = parser.GetCommand(1)) {
            bsize = input->Get<int>();
            out << GTPSuccess("");
        } else {
            out << GTPFail("");
        }
        agent_->GetState().Reset(bsize, agent_->GetState().GetKomi());
        agent_->GetNetwork().Reload(bsize);
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
            out << GTPFail("");
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
            out << GTPFail("");
        }
    } else if (const auto res = parser.Find("fixed_handicap", 0)) {
        auto handicap = -1;
        if (const auto input = parser.GetCommand(1)) {
            agent_->GetState().ClearBoard();
            handicap = input->Get<int>();
        }
        if (agent_->GetState().SetFixdHandicap(handicap)) {
            out << GTPSuccess("");
        } else {
            out << GTPFail("");
        }
    } else if (const auto res = parser.Find("place_free_handicap", 0)) {
         auto handicap = -1;
        if (const auto input = parser.GetCommand(1)) {
            agent_->GetState().ClearBoard();
            handicap = input->Get<int>();
        }
        auto stone_list = agent_->GetState().PlaceFreeHandicap(handicap);
        if (!stone_list.empty()) {
            auto vtx_list = std::ostringstream{};
            for (size_t i = 0; i < stone_list.size(); i++) {
                auto vtx = stone_list[i];
                vtx_list << agent_->GetState().VertexToText(vtx);
                if (i != stone_list.size() - 1) vtx_list << ' ';
            }
            out << GTPSuccess(vtx_list.str());
        } else {
            out << GTPFail("");
        }
    } else if (const auto res = parser.Find("set_free_handicap", 0)) {
        auto movelist = std::vector<std::string>{};
        for (auto i = size_t{1}; i < parser.GetCount(); ++i) {
            movelist.emplace_back(parser.GetCommand(i)->Get<std::string>());
        }
        if (agent_->GetState().SetFreeHandicap(movelist)) {
            out << GTPSuccess("");
        } else {
            out << GTPFail("");
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
        if (filename.empty()) {
            out << GTPFail("");
        } else {
            agent_->GetState() = Sgf::Get().FormFile(filename, movenum);
            out << GTPSuccess("");
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
        auto result = agent_->GetSearch().Computation(400);
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
        agent_->GetState().SetColor(color);
        auto move = agent_->GetSearch().ThinkBestMove();
        agent_->GetState().PlayMove(move);
        out << GTPSuccess(agent_->GetState().VertexToText(move));

    //} else if (const auto res = parser.Find("kgs-genmove_cleanup", 0)) {
    //    out << GTPFail("");
    //} else if (const auto res = parser.Find("kgs-time_settings", 0)) {
    //    out << GTPFail("");

    } else if (const auto res = parser.Find("kgs-game_over", 0)) {
        agent_->GetNetwork().ClearCache();
        out << GTPSuccess("");
    } else if (const auto res = parser.Find("undo", 0)) {
        if (agent_->GetState().UndoMove()) {
            out << GTPSuccess("");
        } else {
            out << GTPFail("");
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
            out << GTPFail("");
        } else {
            agent_->GetSearch().TimeSettings(main_time, byo_yomi_time, byo_yomi_stones);
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
            out << GTPFail("");
        } else {
            agent_->GetSearch().TimeLeft(color, time, stones);
            out << GTPSuccess("");
        }
    } else if (const auto res = parser.Find("final_status_list", 0)) {
        int pass_cnt = 0;
        while (agent_->GetState().GetPasses() >= 2) {
            agent_->GetState().UndoMove();
            agent_->GetState().UndoMove();
            pass_cnt += 2;
        }

        static constexpr auto OWBERSHIP_THRESHOLD = 0.75f;
        auto result = agent_->GetSearch().Computation(0);
        auto bsize = agent_->GetState().GetBoardSize();
        auto color = agent_->GetState().GetToMove();

        auto alive = std::vector<std::vector<int>>{};
        auto dead = std::vector<std::vector<int>>{};

        for (int idx = 0; idx < agent_->GetState().GetNumIntersections(); ++idx) {
            auto x = idx % bsize;
            auto y = idx / bsize;
            auto vtx = agent_->GetState().GetVertex(x,y);
            auto owner = result.root_ownership[idx];
            auto state = agent_->GetState().GetState(vtx);

            if (owner > OWBERSHIP_THRESHOLD) {
                if (color == state) {
                    alive.emplace_back(agent_->GetState().GetStringList(vtx));
                } else if ((!color) == state) {
                    dead.emplace_back(agent_->GetState().GetStringList(vtx));
                }
            } else if (owner < -OWBERSHIP_THRESHOLD) {
                if ((!color) == state) {
                    alive.emplace_back(agent_->GetState().GetStringList(vtx));
                } else if (color == state) {
                    dead.emplace_back(agent_->GetState().GetStringList(vtx));
                }
            }
        }

        // remove multiple mentions of the same string
        // unique reorders and returns new iterator, erase actually deletes
        std::sort(begin(alive), end(alive));
        alive.erase(std::unique(std::begin(alive), std::end(alive)),
                    std::end(alive));

        std::sort(std::begin(dead), std::end(dead));
        dead.erase(std::unique(std::begin(dead), std::end(dead)),
                   std::end(dead));

        auto vtx_list = std::ostringstream{};

        if (const auto input = parser.Find("alive", 1)) {
            for (size_t i = 0; i < alive.size(); i++) {
                vtx_list << (i == 0 ? "" : "\n");
                auto &string = alive[i];
                for (size_t j = 0; j < string.size(); j++) {
                    auto vtx = string[j];
                    vtx_list << agent_->GetState().VertexToText(vtx);
                    if (j != string.size() - 1) vtx_list << ' ';
                }
            }
            out << GTPSuccess(vtx_list.str());
        } else if (const auto input = parser.Find("dead", 1)) {
             for (size_t i = 0; i < dead.size(); i++) {
                vtx_list << (i == 0 ? "" : "\n");
                auto &string = dead[i];
                for (size_t j = 0; j < string.size(); j++) {
                    auto vtx = string[j];
                    vtx_list << agent_->GetState().VertexToText(vtx);
                    if (j != string.size() - 1) vtx_list << ' ';
                }
            }
            out << GTPSuccess(vtx_list.str());
        } else {
            out << GTPFail("");
        }
        for (int i = 0; i < pass_cnt; ++i) {
            agent_->GetState().PlayMove(kPass);
        }
    } else if (const auto res = parser.Find({"help", "list_commands"}, 0)) {
        auto list_commands = std::ostringstream{};
        auto idx = size_t{0};
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
            out << GTPFail("false");
        }
    } else if (const auto res = parser.Find("supervised", 0)) {
        auto sgf_file = std::string{};
        auto data_file = std::string{};

        if (const auto sgf = parser.GetCommand(1)) {
            sgf_file = sgf->Get<std::string>();
        }
        if (const auto data = parser.GetCommand(2)) {
            data_file = data->Get<std::string>();
        }

        if (!sgf_file.empty() && !data_file.empty()) {
            Supervised::Get().FromSgf(sgf_file, data_file);
            out << GTPSuccess("");
        } else {
            out << GTPFail("");
        }
    } else if (const auto res = parser.Find("planes", 0)) {
        int symmetry = 0;

        if (const auto symm = parser.GetCommand(1)) {
            symmetry = symm->Get<int>();
        }

        out << GTPSuccess(Encoder::Get().GetPlanesString(agent_->GetState(), symmetry));
    } else if (const auto res = parser.Find("raw-nn", 0)) {
        int symmetry = 0;

        if (const auto symm = parser.GetCommand(1)) {
            symmetry = symm->Get<int>();
        }

        out << GTPSuccess(agent_->GetNetwork().GetOutputString(agent_->GetState(), Network::kDirect, symmetry));
    } else {
        out << GTPFail("unknown command");
    }
    return out.str();
}

std::string GtpLoop::GTPSuccess(std::string response) {
    auto out = std::ostringstream{};
    auto prefix = std::string{"= "};
    auto suffix = std::string{"\n\n"};

    out << prefix << response << suffix;

    return out.str();
}

std::string GtpLoop::GTPFail(std::string response) {
    auto out = std::ostringstream{};
    auto prefix = std::string{"? "};
    auto suffix = std::string{"\n\n"};

    out << prefix << response << suffix;

    return out.str();
}
