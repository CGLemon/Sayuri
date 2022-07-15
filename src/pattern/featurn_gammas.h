#pragma once

#include <vector>

// Gamma value from the pachi

float kNoSpatial = 0.029559;

std::vector<float> kDistToBorder = {
1.02207,   // 1
1.36662,
1.25208,
1.02381,
0.866419   // 5
};

std::vector<float> kDistToLastMove = {
1,         // 0
3.23899,   // 1
2.47651,
1.5095,
1.67321,
1.21389,
1.01648,
0.832063,
0.713436,
0.608242,
0.515163,
0.448311,
0.398505,
0.360999,
0.324438,
0.321366,
0.257671,
0.179091,  // >= 17
1,
};
