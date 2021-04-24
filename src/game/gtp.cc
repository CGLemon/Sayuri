#include "game/gtp.h"
#include "game/sgf.h"
#include "game/commands_list.h"
#include "utils/log.h"

#include "data/supervised.h"
#include "neural/encoder.h"
#include "search/end_game.h"

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
        out << GTPSuccess("NA");
    } else if (const auto res = parser.Find("version", 0)) {
        out << GTPSuccess("NA");
    } else if (const auto res = parser.Find("showboard", 0)) {
        agent_->GetState().ShowBoard();
        out << GTPSuccess("");
    } else if (const auto res = parser.Find("boardsize", 0)){
        auto size = agent_->GetState().GetBoardSize();
        if (const auto input = parser.GetCommand(1)) {
            size = input->Get<int>();
            out << GTPSuccess("");
        } else {
            out << GTPFail("");
        }
        agent_->GetState().Reset(size, agent_->GetState().GetKomi());
    } else if (const auto res = parser.Find("clear_board", 0)){
        agent_->GetState().ClearBoard();
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
        out << GTPFail("");
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
        out << GTPSuccess(Sgf::Get().ToString(agent_->GetState()));
    } else if (const auto res = parser.Find("final_score", 0)) {
        out << GTPFail("");
    } else if (const auto res = parser.Find("genmove", 0)) {
        out << GTPFail("");
    } else if (const auto res = parser.Find("kgs-genmove_cleanup", 0)) {
        out << GTPFail("");
    } else if (const auto res = parser.Find("kgs-time_settings", 0)) {
        out << GTPFail("");
    } else if (const auto res = parser.Find("kgs-game_over", 0)) {
        out << GTPFail("");
    } else if (const auto res = parser.Find("undo", 0)) {
        if (agent_->GetState().UndoMove()) {
            out << GTPSuccess("");
        } else {
            out << GTPFail("");
        }
    } else if (const auto res = parser.Find("time_settings", 0)) {
        out << GTPFail("");
    } else if (const auto res = parser.Find("time_left", 0)) {
        out << GTPFail("");
    } else if (const auto res = parser.Find("final_status_list", 0)) {
        out << GTPFail("");
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
    } else {
        out << GTPFail("unknown command");
    }
    return out.str();
}

std::string GtpLoop::GTPSuccess(std::string response) {
    auto out = std::ostringstream{};
    auto Prefix = std::string{"= "};
    auto suffix = std::string{"\n\n"};

    out << Prefix << response << suffix;

    return out.str();
}

std::string GtpLoop::GTPFail(std::string response) {
    auto out = std::ostringstream{};
    auto Prefix = std::string{"? "};
    auto suffix = std::string{"\n\n"};

    out << Prefix << response << suffix;

    return out.str();
}
