#pragma once

#include "game/game_state.h"

#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

class SgfNode {
public:
    void AddProperty(std::string property, std::string value);
    std::shared_ptr<SgfNode> AddChild();

    void StartPopulateState();
    void InitState();

    std::shared_ptr<SgfNode> GetChild(unsigned int index);

    GameState GetMainlineState(unsigned int movenum);

private:
    GameState& GetState();
    int GetVertexFromString(const std::string& movestring);
    void PopulateState(GameState currstate);

    std::shared_ptr<std::string> GetPropertyValue(std::string property) const;

    using PropertyMap = std::multimap<std::string, std::string>;
    PropertyMap properties_;

    std::vector<std::shared_ptr<SgfNode>> children_;

    GameState state_;
};

class SgfParser {
public:
    static SgfParser& Get();

    std::shared_ptr<SgfNode> ParseFormFile(std::string filename, size_t index=0) const;
    std::shared_ptr<SgfNode> ParseFormString(std::string sgfstring) const;

    std::vector<std::string> ChopAll(std::string filename) const;

private:
    void Parse(std::istringstream &strm, std::shared_ptr<SgfNode> node) const;

    std::string ParsePropertyValue(std::istringstream &strm, bool &success) const;
    std::string ParsePropertyName(std::istringstream &strm) const;

    std::vector<std::string> ChopStream(std::istream& ins, size_t stopat) const;
    std::vector<std::string> ChopAll(std::string filename, size_t stopat) const;
    std::string ChopFromFile(std::string filename, size_t index) const;
};

class Sgf {
public:
    static Sgf& Get();

    GameState FormFile(std::string filename, unsigned int movenum);

    GameState FormString(std::string sgfstring, unsigned int movenum);

    std::string ToString(GameState &state);

    void ToFile(std::string filename, GameState &state);
};
