#include "match/arena.h"
#include "utils/random.h"
#include "utils/log.h"
#include "utils/threadpool.h"

Arena::Arena() {
    pool_.Initialize();

    Run();
}

void Arena::Run() {
    MatchGames(1);
}

void Arena::MatchGames(int num_games) {
    PlayerWrapper a;
    PlayerWrapper b;

    if (!pool_.RandomMatch(&(a.player), &(b.player))) {
        return;
    }

    GameState main_game;
    main_game.Reset(GetOption<int>("defualt_boardsize"),
                        GetOption<float>("defualt_komi"));
    ThreadPool::Get(GetOption<int>("threads"));

    a.Apply(main_game);
    b.Apply(main_game);

    int a_wins = 0;
    int b_wins = 0;

    for (int i = 0; i < num_games; ++i) {
        main_game.ClearBoard();

        bool a_hold_black = (bool)(Random<kXoroShiro128Plus>::Get().Generate() % 2);

        while (!main_game.IsGameOver()) {
            auto color = main_game.GetToMove();
            int move;
            if ((color == kBlack && a_hold_black) ||
                    (color == kWhite && !a_hold_black)) {
                move = a.search->ThinkBestMove();
            } else {
                move = b.search->ThinkBestMove();
            }
            main_game.PlayMove(move);
            main_game.ShowBoard(); // debug
        }

        auto winner = main_game.GetWinner();
        if (winner == kUndecide && main_game.GetPasses() >= 2) {
            auto score = main_game.GetFinalScore();
            if (score > 1e-4f) {
                main_game.SetWinner(kBlackWon);
            } else if (score < -1e-4f) {
                main_game.SetWinner(kWhiteWon);
            } else {
                main_game.SetWinner(kDraw);
            }
            winner = main_game.GetWinner();
        }
        if (winner != kDraw) {
            if (winner == kBlackWon) { 
                if (a_hold_black) {
                    a_wins += 1;
                } else {
                    b_wins += 1;
                }
            }
            if (winner == kWhiteWon) { 
                if (!a_hold_black) {
                    a_wins += 1;
                } else {
                    b_wins += 1;
                }
            } 
        }
    }
    LOGGING << "totally played " << num_games << " games\n";
    LOGGING << a.player->name << " won " <<  a_wins << " games\n";
    LOGGING << b.player->name << " won " <<  b_wins << " games\n";
}
