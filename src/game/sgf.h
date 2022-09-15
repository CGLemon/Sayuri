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
    SgfNode* AddChild();

    void StartPopulateState();
    void InitState();

    SgfNode* GetChild(unsigned int index);

    GameState GetMainLineState(unsigned int movenum);

private:
    GameState& GetState();
    int GetVertexFromString(const std::string& movestring);
    void PopulateState(GameState currstate);

    std::unique_ptr<std::string> GetPropertyValue(std::string property) const;

    using PropertyMap = std::multimap<std::string, std::string>;
    PropertyMap properties_;

    std::vector<std::unique_ptr<SgfNode>> children_;

    GameState state_;
};

class SgfParser {
public:
    static SgfParser& Get();

    std::unique_ptr<SgfNode> ParseFromFile(std::string filename, size_t index=0) const;
    std::unique_ptr<SgfNode> ParseFromString(std::string sgfstring) const;

    std::vector<std::string> ChopAll(std::string filename) const;

private:
    void Parse(std::istringstream &strm, SgfNode* node) const;

    std::string ParsePropertyValue(std::istringstream &strm, bool &success) const;
    std::string ParsePropertyName(std::istringstream &strm) const;

    std::vector<std::string> ChopStream(std::istream& ins, size_t stopat) const;
    std::vector<std::string> ChopAll(std::string filename, size_t stopat) const;
    std::string ChopFromFile(std::string filename, size_t index) const;
};

class Sgf {
public:
    static Sgf& Get();

    // Get the main line state from the first game in the SGF file.
    GameState FromFile(std::string filename, unsigned int movenum);

    // Get the main line state from the SGF string.
    GameState FromString(std::string sgfstring, unsigned int movenum);

    // Import the game state to string.
    std::string ToString(GameState &state);

    // Import the game state as SGF file.
    void ToFile(std::string filename, GameState &state);

    // Remove unused SGF elements for this program and save it.
    void CleanSgf(std::string in, std::string out);
};
