#include "utils/simple_json.h"

std::string JsonEscape(const std::string &str) {
    const int size = str.size();
    auto output = std::string{};
    for (int i = 0; i < size; ++i) {
        const char c = str[i];
        switch (c) {
            case '\"': output += "\\\""; break;
            case '\\': output += "\\\\"; break;
            case '\b': output += "\\b";  break;
            case '\f': output += "\\f";  break;
            case '\n': output += "\\n";  break;
            case '\r': output += "\\r";  break;
            case '\t': output += "\\t";  break;
            default  : output += c; break;
        }
    }
    return output;
}

void ParseNext(Json &json, const std::string &, size_t &);

void ConsumeWhiteSpace(const std::string &str, size_t &offset) {
    while (std::isspace(str[offset])) {
        ++offset;
    }
}

void ParseArray(Json &node, const std::string &str, size_t &offset) {
    auto array = Json(Json::Type::kArray);

    ++offset;
    ConsumeWhiteSpace(str, offset);

    if (str[offset] == ']') {
        ++offset;
        node.Import(array);
        return;
    }

    int index = 0;
    while (true) {
        // Get the each item.
        Json val;
        ParseNext(val, str, offset);
        array[index++].Import(val);
        ConsumeWhiteSpace(str, offset);

        const char c = str[offset++];
        if (c == ',') {
            // do nothing...
        } else if (c == ']') {
            break;
        } else {
            throw "Array Error: Token is not ',' or ']'";
        }
    }
    node.Import(array);
}

void ParseObject(Json &node, const std::string &str, size_t &offset) {
    auto object = Json(Json::Type::kObject);

    ++offset;
    ConsumeWhiteSpace(str, offset);
    if(str[offset] == '}') {
        ++offset;
        node.Import(object);
    }

    while (true) {
        // Get key...
        Json key;
        ParseNext(key, str, offset);
        ConsumeWhiteSpace(str, offset);

        if(str[offset] != ':') {
            throw "Object Error: Token is not colon";
        }
        ConsumeWhiteSpace(str, ++offset);

        // Get value...
        Json val;
        ParseNext(val, str, offset);
        object[key.ToString()].Import(val);
        
        ConsumeWhiteSpace(str, offset);

        const char c = str[offset++];
        if (c == ',') {
            // do nothing...
        } else if (c == '}') {
            break;
        } else {
            throw "Object Error: Token is not comma";
        }
    }

    node.Import(object);
}

void ParseString(Json &node, const std::string &str, size_t &offset) {
    auto sval = std::string{};
    for (char c = str[++offset]; c != '\"' ; c = str[++offset]) {
        if (c == '\\') {
            switch (str[++offset]) {
                case '\"': sval += '\"'; break;
                case '\\': sval += '\\'; break;
                case '/' : sval += '/' ; break;
                case 'b' : sval += '\b'; break;
                case 'f' : sval += '\f'; break;
                case 'n' : sval += '\n'; break;
                case 'r' : sval += '\r'; break;
                case 't' : sval += '\t'; break;
                case 'u' :
                    {
                        sval += "\\u" ;
                        for (int i = 1; i <= 4; ++i) {
                            c = str[offset+i];
                            if (std::isdigit(c) ||
                                    (c >= 'a' && c <= 'f') ||
                                    (c >= 'A' && c <= 'F')) {
                                sval += c;
                            } else {
                                throw "String Error: Token is not hex character in unicode escape";
                            }
                        }
                        offset += 4;
                    }
                    break;
                default  : sval += '\\'; break;
            }
        } else {
            sval += c;
        }
    }
    ++offset;
    node.SetString(sval);
}

void ParseNumber(Json &node, const std::string &str, size_t &offset) {
    auto val = std::string{};
    char c = '\0';
    bool is_float = false;
    int exp10 = 0;

    // Number field.
    while (true) {
        c = str[offset++];
        if (std::isdigit(c) || c == '-') {
            val += c;
        } else if (c == '.' ) {
            val += c; 
            is_float = true;
        } else {
            break;
        }
    }

    // Exponent field.
    if (c == 'E' || c == 'e') {
        auto exp10_str = std::string{};
        c = str[offset++];
        if (c == '-') {
            exp10_str += '-';
            is_float = true;
        }
        while (true) {
            c = str[offset++];
            if (std::isdigit(c)) {
                exp10_str += c;
            } else if (!std::isspace(c) && c != ',' && c != ']' && c != '}') {
                throw "Number Error: Token is not a number for exponent";
            } else {
                break;
            }
        }
        exp10 = std::stol(exp10_str);
    } else if (!std::isspace(c) && c != ',' && c != ']' && c != '}') {
        throw "Number Error: Token is not number";
    }
    --offset;
    
    Json::Floating fval = 0;
    Json::Integral ival = 0;
    if (is_float) {
        fval = std::stod(val);
        if (exp10 != 0) {
            fval *= std::pow(10, exp10);
        }
    } else {
        ival = std::stol(val);
        if (exp10 != 0) {
            ival *= std::pow(10, exp10);
        }
    }
    if (is_float) {
        node.SetFloat(fval);
    } else {
        node.SetInt(ival);
    }
}

void ParseBool(Json &node, const std::string &str, size_t &offset) {
    Json::Boolean bval;
    if (str.substr(offset, 4) == "true") {
        bval = true;
    } else if (str.substr(offset, 5) == "false") {
        bval = false;
    } else {
        throw "Bool Error: Token is not 'true' or 'false'";
    }
    offset += bval ? 4 : 5;
    node.SetBool(bval);
}

void ParseNull(Json &node, const std::string &str, size_t &offset) {
    if (str.substr(offset, 4) != "null") {
        throw "Null Error: Token is not 'null'";
    }
    offset += 4;
    node.SetNull();
}

void ParseNext(Json &node, const std::string &str, size_t &offset) {
    ConsumeWhiteSpace(str, offset);
    char c = str[offset];

    switch (c) {
        case '[' : ParseArray(node, str, offset); return;
        case '{' : ParseObject(node, str, offset); return;
        case '\"': ParseString(node, str, offset); return;
        case 't' :
        case 'f' : ParseBool(node, str, offset); return;
        case 'n' : ParseNull(node, str, offset); return;

        default  : if (std::isdigit(c) || c == '-') {
                       ParseNumber(node, str, offset); return;
                   }
    }
    throw "Parse Next Error: Token is Unknown starting character";
}

std::shared_ptr<Json> Json::ParseAndReturn(const std::string v) {
    auto root = std::make_shared<Json>();
    size_t offset = 0;
    try {
        ParseNext(*root, v, offset);
    } catch (const char *err) {
        std::cerr << err << std::endl;
        root->SetNull();
    }
    return root;
}

void Json::Parse(const std::string v) {
    size_t offset = 0;
    try {
        this->TryReleasePointer();
        ParseNext(*this, v, offset);
    } catch (const char *err) {
        std::cerr << err << std::endl;
        this->SetNull();
    }
}
