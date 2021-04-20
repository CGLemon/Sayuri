#pragma once

#include <string>

class Supervised {
public:
    static Supervised &Get();

    void FromSgf(std::string sgf_name, std::string out_name);

private:

};
