#include "game/sgf.h"
#include "game/types.h"
#include "utils/log.h"
#include "utils/time.h"

#include <limits>

void SgfNode::AddProperty(std::string property, std::string value) {
    properties_.emplace(property, value);
}

std::shared_ptr<std::string> SgfNode::GetPropertyValue(std::string property) const {
    auto it = properties_.find(property);
    if (it != end(properties_)) {
        return std::make_shared<std::string>(it->second);
    }
    return nullptr;
}

std::shared_ptr<SgfNode> SgfNode::AddChild() {
    // first allocation is better small
    if (children_.empty()) {
        children_.reserve(1);
    }
    children_.emplace_back(std::make_shared<SgfNode>());
    return children_.back();
}

GameState& SgfNode::GetState() {
    return state_;
}

void SgfNode::StartPopulateState() {
    InitState();
    PopulateState(GetState());
}

void SgfNode::InitState() {
    GetState().ClearBoard();
}

int SgfNode::GetVertexFromString(const std::string& movestring) {
    if (movestring.empty()) {
        return kPass;
    }

    int bsize = GetState().GetBoardSize();

    if (bsize <= 19) {
        if (movestring == "tt") {
            return kPass;
        }
    }

    if (bsize <= 0) {
        throw std::runtime_error("Node has 0 sized board");
    }

    if (movestring.size() != 2) {
        throw std::runtime_error("Illegal SGF move");
    }

    char c1 = movestring[0];
    char c2 = movestring[1];

    int cc1;
    int cc2;

    if (c1 >= 'A' && c1 <= 'Z') {
        cc1 = 26 + c1 - 'A';
    } else {
        cc1 = c1 - 'a';
    }
    if (c2 >= 'A' && c2 <= 'Z') {
        cc2 = bsize - 26 - (c2 - 'A') - 1;
    } else {
        cc2 = bsize - (c2 - 'a') - 1;
    }

    // catch illegal SGF
    if (cc1 < 0 || cc1 >= bsize
        || cc2 < 0 || cc2 >= bsize) {
        throw std::runtime_error("Illegal SGF move");
    }

    int vtx = GetState().GetVertex(cc1, cc2);

    return vtx;
}

void SgfNode::PopulateState(GameState currstate) {
    state_ = currstate;

    // first check for go game setup in properties
    if (const auto res = GetPropertyValue("GM")) {
        if (std::stoi(*res) != 1) {
            throw std::runtime_error("SGF Game is not a Go game");
        }
    }

    // board size
    if (const auto res = GetPropertyValue("SZ")) {
        const auto bsize = std::stoi(*res);
        GetState().Reset(bsize, GetState().GetKomi());
    }

    // komi
    if (const auto res = GetPropertyValue("KM")) {
        const auto komi = std::stof(*res);
        GetState().SetKomi(komi);
    }

    // handicap
    if (const auto res = GetPropertyValue("HA")) {
        const auto handicap = std::stoi(*res);
        GetState().SetFixdHandicap(handicap);
    }

    // time
    if (const auto res = GetPropertyValue("TM")) {

    }

    if (const auto res = GetPropertyValue("AB")) {

    }

    // result
    if (const auto res = GetPropertyValue("RE")) {
        if (res->find("B+") != std::string::npos) {
            GetState().SetWinner(kBlackWon);
        } else if (res->find("W+") != std::string::npos) {
            GetState().SetWinner(kWhiteWon);
        } else if (res->find("0") != std::string::npos) {
            GetState().SetWinner(kDraw);
        } else {
            GetState().SetWinner(kUndecide);
        }

        auto result = *res;
        auto black_final_score = 0;
        if (result == "0") {
            black_final_score = 0;
        } else {
            if (result.size() > 2 && result.find("+") == 1) {
                result.erase(0, 2);
                black_final_score = std::stof(result);

                if (res->find("W+") != std::string::npos) {
                    black_final_score = 0 - black_final_score;
                }
            }
        }
        GetState().black_final_score = black_final_score;
    }

    if (const auto res = GetPropertyValue("PL")) {
        const auto who = *res;
        if (who == "B") {
            GetState().SetColor(kBlack);
        } else if (who == "W") {
            GetState().SetColor(kWhite);
        }
    }

    if (const auto res = GetPropertyValue("B")) {
        const auto vtx = GetVertexFromString(*res);
        GetState().PlayMove(vtx, kBlack);
    } else if (const auto res = GetPropertyValue("W")) {
        const auto vtx = GetVertexFromString(*res);
        GetState().PlayMove(vtx, kWhite);
    }

    for (auto& child : children_) {
        child->PopulateState(GetState());
    }
}

std::shared_ptr<SgfNode> SgfNode::GetChild(unsigned int index) {
    if (children_.size() <= index) {
        return nullptr;
    }
    return children_[index];
}

// This follows the entire line, and doesn't really need the intermediate
// states, just the moves. As a consequence, states that contain more than
// just moves won't have any effect.
GameState SgfNode::GetMainlineState(unsigned int movenum) {
    auto link = std::make_shared<SgfNode>(*this);

    // This initializes a starting state from a KoState and
    // sets up the game history.
    GameState main_state = GetState();

    for (unsigned int i = 0; i <= movenum && link != nullptr; i++) {
        // root position has no associated move
        if (i != 0) {
            auto vtx = kNullVertex;
            auto color = kInvalid;
            if (const auto res = link->GetPropertyValue("B")) {
                vtx = GetVertexFromString(*res);
                color = kBlack;
            } else if (const auto res = link->GetPropertyValue("W")) {
                vtx = GetVertexFromString(*res);
                color = kWhite;
            }

            if (vtx != kNullVertex && color != kInvalid) {
                if (!main_state.PlayMove(vtx, color)) {
                    return main_state;
                }
            } else {
                return main_state;
            }
        }
        link = link->GetChild(0);
    }
    return main_state;
}

std::string SgfParser::ParsePropertyName(std::istringstream & strm) const {
    auto result = std::ostringstream{};

    auto c = char{};
    while (strm >> c) {
        // SGF property names are guaranteed to be uppercase,
        // except that some implementations like IGS are retarded
        // and don't folow the spec. So allow both upper/lowercase.
        if (!std::isupper(c) && !std::islower(c)) {
            strm.unget();
            break;
        } else {
            result << c;
        }
    }

    return result.str();
}

std::string SgfParser::ParsePropertyValue(std::istringstream &strm, bool &success) const {
    strm >> std::noskipws;
    auto c = char{};

    while (strm >> c) {
        if (!std::isspace(c)) {
            strm.unget();
            break;
        }
    }

    strm >> c;

    if (c != '[') {
        strm.unget();
        success = false;
        return std::string{};
    }

    auto result = std::ostringstream{};

    while (strm >> c) {
        if (c == ']') {
            break;
        } else if (c == '\\') {
            strm >> c;
        }
        result << c;
    }

    strm >> std::skipws;
    success = true;
    return result.str();
}

void SgfParser::Parse(std::istringstream &strm, std::shared_ptr<SgfNode> node) const {
    auto splitpoint = false;
    auto c = char{};

    while (strm >> c) {
        if (strm.fail()) {
            return;
        }

        if (std::isspace(c)) {
            continue;
        }

        // parse a property
        if (std::isalpha(c) && std::isupper(c)) {
            strm.unget();

            auto propname = ParsePropertyName(strm);
            do {
                auto success = false;
                auto propval = ParsePropertyValue(strm, success);
                if (success) {
                    node->AddProperty(propname, propval);
                } else {
                    break;
                }
            } while(true);

            continue;
        }

        if (c == '(') {
            // eat first ;
            auto cc = char{};
            do {
                strm >> cc;
            } while (std::isspace(cc));
            if (cc != ';') {
                strm.unget();
            }

            // start a variation here
            splitpoint = true;

            // new node
            auto newptr = node->AddChild();
            Parse(strm, newptr);
        } else if (c == ')') {
            // variation ends, go back
            // if the variation didn't start here, then
            // push the "variation ends" mark back
            // and try again one level up the tree
            if (!splitpoint) {
                strm.unget();
                return;
            } else {
                splitpoint = false;
                continue;
            }
        } else if (c == ';') {
            // new node
            auto newptr = node->AddChild();
            node = newptr;
            continue;
        }
    }
}

std::vector<std::string> SgfParser::ChopStream(std::istream& ins, size_t stopat) const {
    auto result = std::vector<std::string>{};
    auto gamebuff = std::string{};

    ins >> std::noskipws;

    int nesting = 0;      // parentheses
    bool intag = false;   // brackets
    int line = 0;

    auto c = char{};
    while (ins >> c && result.size() <= stopat) {
        if (c == '\n') line++;

        gamebuff.push_back(c);
        if (c == '\\') {
            // read literal char
            ins >> c;
            gamebuff.push_back(c);
            // Skip special char parsing
            continue;
        }

        if (c == '(' && !intag) {
            if (nesting == 0) {
                // eat ; too
                do {
                    ins >> c;
                } while (std::isspace(c) && c != ';');
                gamebuff.clear();
            }
            nesting++;
        } else if (c == ')' && !intag) {
            nesting--;

            if (nesting == 0) {
                result.push_back(gamebuff);
            }
        } else if (c == '[' && !intag) {
            intag = true;
        } else if (c == ']') {
            if (intag == false) {
                ERROR << "Tag error on line" << ' ' << line << std::endl;
            }
            intag = false;
        }
    }

    // No game found? Assume closing tag was missing (OGS)
    if (result.empty()) {
        result.push_back(gamebuff);
    }

    return result;
}

std::vector<std::string> SgfParser::ChopAll(std::string filename,
                                            size_t stopat) const {
    std::ifstream ins(filename.c_str(), std::ifstream::binary | std::ifstream::in);

    if (ins.fail()) {
        throw std::runtime_error("Error opening file");
    }

    auto result = ChopStream(ins, stopat);
    ins.close();

    return result;
}

std::vector<std::string> SgfParser::ChopAll(std::string filename) const {
    return ChopAll(filename,  std::numeric_limits<size_t>::max());
}

// Scan the file and extract the game with number index.
std::string SgfParser::ChopFromFile(std::string filename, size_t index) const {
    auto vec = ChopAll(filename, index);
    return vec[index];
}

SgfParser& SgfParser::Get() {
    static SgfParser parser;
    return parser;
}

std::shared_ptr<SgfNode> SgfParser::ParseFormFile(std::string filename, size_t index) const {
    auto sgfstring = ChopFromFile(filename, index);
    auto node = ParseFormString(sgfstring);
    return node;
}

std::shared_ptr<SgfNode> SgfParser::ParseFormString(std::string sgfstring) const {
    auto node = std::make_shared<SgfNode>();
    auto rootnode = node;
    auto gamebuff = std::istringstream{sgfstring};
    Parse(gamebuff, node);
    rootnode->StartPopulateState();
    return rootnode;
}

Sgf& Sgf::Get() {
    static Sgf sgf;
    return sgf;
}

GameState Sgf::FormFile(std::string filename, unsigned int movenum) {
    auto node = SgfParser::Get().ParseFormFile(filename);
    return node->GetMainlineState(movenum);
}

GameState Sgf::FormString(std::string sgfstring, unsigned int movenum) {
    auto node = SgfParser::Get().ParseFormString(sgfstring);
    return node->GetMainlineState(movenum);
}

template<typename T>
std::string MakePropertyString(std::string property, T value) {
    auto out = std::ostringstream{};
    out << property << '[' << value << ']';
    return out.str();
}

std::string Sgf::ToString(GameState &state) {
    auto out = std::ostringstream{};
    auto &history = state.GetHistory();

    out << '(' << ';';
    out << MakePropertyString("GM", 1);
    out << MakePropertyString("FF", 4);
    out << MakePropertyString("SZ", state.GetBoardSize());
    out << MakePropertyString("KM", state.GetKomi());
    out << MakePropertyString("RU", "chinese");
    out << MakePropertyString("PB", "black bot");
    out << MakePropertyString("PW", "white bot");
    out << MakePropertyString("DT", CurrentDateTime());

    if (state.GetWinner() != kUndecide) {
        out << "RE" << '[';
        auto score = state.GetFinalScore();
        if (state.GetWinner() == kBlackWon) {
            out << "B+";
            if (state.GetPasses() >= 2) out << score;
        } else if (state.GetWinner() == kWhiteWon) {
            out << "W+";
            if (state.GetPasses() >= 2) out << -score;
        } else if (state.GetWinner() == kDraw) {
            out << "0";
        }
        out << ']';
    }

    for (const auto &board : history) {
        auto color = !board->GetToMove();
        auto lastmove = board->GetLastMove();
        if (lastmove != kNullVertex) {
            out << ';';
            if (color == kBlack) {
                out << 'B';
            } else if (color == kWhite) {
                out << 'W';
            }
            out << '[';
            out << state.VertexToSgf(lastmove);
            out << ']';
        }
    }
    out << ')';
    return out.str();
}
