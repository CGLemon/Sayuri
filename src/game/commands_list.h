#pragma once

#include <vector>
#include <string>

static constexpr auto kProtocolVersion = 2;

static std::vector<std::string> kGtpCommandsList = {
    // Part of GTP version 2 standard command
    "protocol_version",

    // Part of GTP version 2 standard command
    "name",

    // Part of GTP version 2 standard command
    "version",

    // Part of GTP version 2 standard command
    "quit",

    // Part of GTP version 2 standard command
    "known_command",

    // Part of GTP version 2 standard command
    "list_commands",

    // Part of GTP version 2 standard command
    "help",

    // Part of GTP version 2 standard command
    "boardsize",

    // Part of GTP version 2 standard command
    "clear_board",

    // Part of GTP version 2 standard command
    "komi",

    // Part of GTP version 2 standard command
    "play",

    // Part of GTP version 2 standard command
    "genmove",

    // Part of GTP version 2 standard command
    "showboard",

    // Part of GTP version 2 standard command
    "undo",

    // Part of GTP version 2 standard command
    "final_score",

    // Part of GTP version 2 standard command
    "final_status_list",

    // Part of GTP version 2 standard command
    "time_settings",

    // Part of GTP version 2 standard command
    "time_left",

    // Part of GTP version 2 standard command
    "fixed_handicap",

    // Part of GTP version 2 standard command
    "place_free_handicap",

    // Part of GTP version 2 standard command
    "set_free_handicap",

    // Part of GTP version 2 standard command
    "loadsgf",

    "printsgf",

    "is_legal",

    "get_komi",

    "get_handicap",

    "query_boardsize",

    "clear_cache",

    "selfplay-genmove",      // debug

    "selfplay",              // debug

    "dump_training_buffer",  // debug

    "clear_training_buffer", // debug

    // "kgs-chat", // may crush on the KgsGtp

    "analyze",

    "genmove_analyze",

    "lz-analyze",

    "lz-genmove_analyze",

    "sayuri-analyze",

    "sayuri-genmove_analyze",

    "kgs-game_over",

    "kgs-time_settings",

    "supervised",

    "sayuri-supervised",

    "planes",

    "raw-nn",

    "benchmark",

    "genbook",

    "gogui-analyze_commands",

    "gogui-wdl_rating",

    "gogui-policy_heatmap",

    "gogui-policy_rating",

    "gogui-ownership_heatmap",

    "gogui-ownership_influence",

    "gogui-gammas_heatmap"
};
