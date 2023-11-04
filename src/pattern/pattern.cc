#include <array>
#include <algorithm>
#include <iostream>

#include "game/types.h"
#include "pattern/pattern.h"

/* Mapping from point sequence to coordinate offsets (to determine
 * coordinates relative to pattern center). The array is ordered
 * in the gridcular metric order so that we can go through it
 * and incrementally match spatial features in nested circles.
 * Within one circle, coordinates are ordered by rows to keep
 * good cache behavior. */
std::array<PtCoord, kMaxPatternArea> kPointCoords;

/* For each radius, starting index in kPointCoords[]. */
std::array<int, kMaxPatternDist + 2> kPointIndex;

std::uint64_t PatternHash[8][4][kMaxPatternArea];

int CharToColor(char s) {
    switch (s) {
        case '.': return kEmpty;
        case 'X': return kBlack;
        case 'O': return kWhite;
        case '#': return kInvalid;
    }
    return kEmpty; // XXX
}

void PtcoordsInit() {
    int i = 0; /* Indexing ptcoords[] */

    kPointIndex[0] = kPointIndex[1] = 0;
    kPointCoords[i].x = kPointCoords[i].y = 0; i++;

    for (int d = 2; d <= kMaxPatternDist; d++) {
        kPointIndex[d] = i;
        /* For each y, examine all integer solutions
         * of d = |x| + |y| + max(|x|, |y|). */
        /* TODO: (Stern, 2006) uses a hand-modified
         * circles that are finer for small d and more
         * coarse for large d. */
        for (short y = d / 2; y >= 0; y--) {
            short x;
            if (y > d / 3) {
                /* max(|x|, |y|) = |y|, non-zero x */
                x = d - y * 2;
                if (x + y * 2 != d) continue;
            } else {
                /* max(|x|, |y|) = |x| */
                /* Or, max(|x|, |y|) = |y| and x is zero */
                x = (d - y) / 2;
                if (x * 2 + y != d) continue;
            }

            kPointCoords[i].x = x; kPointCoords[i].y = y; i++;
            if (x != 0) { kPointCoords[i].x = -x; kPointCoords[i].y = y; i++; }
            if (y != 0) { kPointCoords[i].x = x; kPointCoords[i].y = -y; i++; }
            if (x != 0 && y != 0) { kPointCoords[i].x = -x; kPointCoords[i].y = -y; i++; }
        }
    }
    kPointIndex[kMaxPatternDist + 1] = i;

#if 0

// d=0 (0)
// d=1 (0) 0,0
// d=2 (1) 0,1 0,-1 1,0 -1,0
// d=3 (5) 1,1 -1,1 1,-1 -1,-1
// d=4 (9) 0,2 0,-2 2,0 -2,0
// d=5 (13) 1,2 -1,2 1,-2 -1,-2 2,1 -2,1 2,-1 -2,-1
// d=6 (21) 0,3 0,-3 2,2 -2,2 2,-2 -2,-2 3,0 -3,0
// d=7 (29) 1,3 -1,3 1,-3 -1,-3 3,1 -3,1 3,-1 -3,-1
// d=8 (37) 0,4 0,-4 2,3 -2,3 2,-3 -2,-3 3,2 -3,2 3,-2 -3,-2 4,0 -4,0
// d=9 (49) 1,4 -1,4 1,-4 -1,-4 3,3 -3,3 3,-3 -3,-3 4,1 -4,1 4,-1 -4,-1
// d=10 (61) 0,5 0,-5 2,4 -2,4 2,-4 -2,-4 4,2 -4,2 4,-2 -4,-2 5,0 -5,0

    for (int d = 0; d <= kMaxPatternDist; d++) {
        fprintf(stderr, "d=%d (%d) ", d, kPointIndex[d]);
        for (int j = kPointIndex[d]; j < kPointIndex[d + 1]; j++) {
            fprintf(stderr, "%d,%d ", kPointCoords[j].x, kPointCoords[j].y);
        }
        fprintf(stderr, "\n");
    }
#endif
}

void PatternHashInit() {
    constexpr int kMaxPatternBoardSize = (kMaxPatternDist+1) * (kMaxPatternDist+1);

    std::uint64_t pthboard[4][kMaxPatternBoardSize];
    int pthbc = kMaxPatternBoardSize / 2; // tengen coord

    std::uint64_t h1 = 0xd6d6d6d1;
    std::uint64_t h2 = 0xd6d6d6d2;
    std::uint64_t h3 = 0xd6d6d6d3;
    std::uint64_t h4 = 0xd6d6d6d4;
    for (int i = 0; i < kMaxPatternBoardSize; i++) {
        pthboard[kEmpty][i]   = (h1 = h1 * 16787);
        pthboard[kBlack][i]   = (h2 = h2 * 16823);
        pthboard[kWhite][i]   = (h3 = h3 * 16811 - 13);
        pthboard[kInvalid][i] = (h4 = h4 * 16811);
    }

    /* Virtual board with hashes created, now fill
     * PatternHash[] with hashes for points in actual
     * sequences, also considering various rotations. */
#define PTH_VMIRROR 1
#define PTH_HMIRROR 2
#define PTH_90ROT   4
    for (int r = 0; r < 8; r++) {
        for (int i = 0; i < kMaxPatternArea; i++) {
            /* Rotate appropriately. */
            int rx = kPointCoords[i].x;
            int ry = kPointCoords[i].y;
            if (r & PTH_VMIRROR) ry = -ry;
            if (r & PTH_HMIRROR) rx = -rx;
            if (r & PTH_90ROT) {
                int rs = rx; rx = -ry; ry = rs;
            }
            int bi = pthbc + ry * (kMaxPatternDist + 1) + rx;

            /* Copy info. */
            PatternHash[r][kEmpty][i] = pthboard[kEmpty][bi];
            PatternHash[r][kBlack][i] = pthboard[kBlack][bi];
            PatternHash[r][kWhite][i] = pthboard[kWhite][bi];
            PatternHash[r][kInvalid][i] = pthboard[kInvalid][bi];
        }
    }
#undef PTH_VMIRROR
#undef PTH_HMIRROR
#undef PTH_90ROT
}

void PatternHashAndCoordsInit() {
    PtcoordsInit();
    PatternHashInit();
}
