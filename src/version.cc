#include "version.h"
#include "utils/format.h"

#include <vector>

const std::string kProgram = "Sayuri";
std::vector<std::string> kVersionNamesTable = {
    "",
    "Mikazuki"
};

constexpr int kVersionMajor = 0;
constexpr int kVersionMinor = 1;
constexpr int kVersionPatch = 0;

std::string GetProgramName() {
    return kProgram;
}

std::string GetProgramVersion() {
    return Format("%d.%d.%d", kVersionMajor, kVersionMinor, kVersionPatch);
}

std::string GetVersionName() {
    return kVersionNamesTable[kVersionMinor];
}
