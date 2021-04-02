#include <iostream>
#include "game/board.h"
#include "game/symmetry.h"
#include "game/zobrist.h"
#include "utils/random.h"

int main(int argc, char **argv) {
    auto board = Board();
    Symmetry::Initialize(9);
    Zobrist::Initialize();

    std::cout << Symmetry::Get().GetDebugString();

    board.Reset(9, 7.5);
    std::cout << board.GetBoardString(0, false);

    board.PlayMoveAssumeLegal(50);
    std::cout << board.GetBoardString(0, false);

    board.PlayMoveAssumeLegal(51);
    std::cout << board.GetBoardString(0, false);

    return 0;
}
