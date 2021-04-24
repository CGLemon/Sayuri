#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

class Supervised {
public:
    static Supervised &Get();

    void FromSgf(std::string sgf_name, std::string out_name);

private:
   void SgfProcess(std::string &sgfstring, std::ostream &out_file);

};
