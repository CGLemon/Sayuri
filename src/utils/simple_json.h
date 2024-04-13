#pragma once

#include <cstdint>
#include <cmath>
#include <cctype>
#include <cstring>
#include <string>
#include <sstream>
#include <vector>
#include <deque>
#include <map>
#include <type_traits>
#include <initializer_list>
#include <ostream>
#include <iostream>
#include <algorithm>
#include <memory>

std::string JsonEscape(const std::string &str);

class Json {
public:
    enum Type {
        kNull = 0,
        kObject,
        kArray,
        kString,
        kFloating,
        kIntegral,
        kBoolean
    };

    using Object = std::map<std::string, Json>;
    using Array = std::deque<Json>;
    using String = std::string;
    using Floating = double;
    using Integral = std::int64_t;
    using Boolean = bool;

    using ObjectPointer = Object*;
    using ArrayPointer = Array*;
    using StringPointer = String*;

    static std::shared_ptr<Json> ParseAndReturn(const std::string v);
    void Parse(const std::string v);

    Json() = default;

    Json(const Json &other) {
        CopyFromOther(other);
    }

    Json(const Json &&other) {
        CopyFromOther(other);
    }

    Json(std::initializer_list<Json> &&list) {
        SetType(Type::kArray);
        int index = GetArrayPointer()->size();
        for (auto i = std::begin(list), e = std::end(list); i != e; i+=1) {
            Json jj = *i;
            this->operator[](index++) = std::move(jj);
        }
    }
    Json(std::initializer_list<std::pair<std::string, Json>> &&list) {
        SetType(Type::kObject);
        for (auto i = std::begin(list), e = std::end(list); i != e; i+=1) {
            Json jj = i->second;
            this->operator[](i->first) = std::move(jj);
        }
    }

    template <typename T>
    Json(
        T val,
        typename std::enable_if_t<std::is_same<T, bool>::value> *cond = 0)
    {
        SetBool(val);
    }

    template <typename T>
    Json(
        T val,
        typename std::enable_if_t<
                     !std::is_same<T, bool>::value &&
                     std::is_integral<T>::value
                 > *cond = 0)
    {
        SetInt(val);
    }

    template <typename T>
    Json(
        T val,
        typename std::enable_if_t<std::is_floating_point<T>::value> *cond = 0)
    {
        SetFloat(val);
    }

    template <typename T>
    Json(
        T val,
        typename std::enable_if_t<std::is_convertible<T,String>::value> *cond = 0)
    {

        SetString(val);
    }

    Json(std::nullptr_t) : Json() {
        SetNull();
    }

    Json(Json::Type t) {
        SetType(t);
    }

    ~Json() {
        TryReleasePointer();
    }

    Json& operator[](const std::string &key) {
        SetType(Type::kObject);
        return GetObjectPointer()->operator[](key);
    }
    Json& operator[](const std::string &&key) {
        SetType(Type::kObject);
        return GetObjectPointer()->operator[](key);
    }
    Json& operator[](int index) {
        SetType(Type::kArray);
        auto ptr = GetArrayPointer();
        if (index >= static_cast<int>(ptr->size())) {
            ptr->resize(index + 1);
        }
        return ptr->operator[](index);
    }

    Json& operator=(Json &&other) {
        TryReleasePointer();
        type_ = other.type_;
        data_ = other.data_;
        other.type_ = Type::kNull;
        return *this;
    }

    void Import(Json &other) {
        TryReleasePointer();
        type_ = other.type_;
        data_ = other.data_;
        other.type_ = Type::kNull;
    }

    void Import(Json &&other) {
        TryReleasePointer();
        type_ = other.type_;
        data_ = other.data_;
        other.type_ = Type::kNull;
    }

    void SetString(String val) {
        SetType(Type::kString);
        *GetStringPointer() = val;
    }
    void SetFloat(Floating val) {
        SetType(Type::kFloating);
        CopyToData(val);
    }
    void SetInt(Integral val) {
        SetType(Type::kIntegral);
        CopyToData(val);
    }
    void SetBool(Boolean val) {
        SetType(Type::kBoolean);
        CopyToData(val);
    }
    void SetNull() {
        SetType(Type::kNull);
    }

    Floating ToFloat() const {
        if (type_ != Type::kFloating) {
            throw "Not the correct casting Type.";
        }
        return GetDataAs<Floating>();
    }
    Integral ToInt() const {
        if (type_ != Type::kIntegral) {
            throw "Not the correct casting Type.";
        }
        return GetDataAs<Integral>();
    }
    Boolean ToBool() const {
        if (type_ != Type::kBoolean) {
            throw "Not the correct casting Type.";
        }
        return GetDataAs<Boolean>();
    }
    std::string ToString() const {
        if (type_ != Type::kString) {
            throw "Not the correct casting Type.";
        }
        return JsonEscape(*GetStringPointer());
    }

    Json::Type GetType() {
        return type_;
    }
    bool IsNull() const {
        return type_ == Type::kNull;
    }
    size_t GetSize() const {
        switch (type_) {
            case Type::kObject: return GetObjectPointer()->size();
            case Type::kArray: return GetArrayPointer()->size();
            case Type::kString:
            case Type::kFloating:
            case Type::kIntegral:
            case Type::kBoolean: return 1;
            default: break;
        }
        return 0;
    }
    bool Find(std::string key, Json::Type type) const {
        switch (type_) {
            case Type::kObject:
                {
                    auto ptr = GetObjectPointer();
                    auto it = ptr->find(key);
                    if (it != std::end(*ptr)) {
                        return it->second.type_ == type;
                    }
                }
            case Type::kArray:
            case Type::kString:
            case Type::kFloating:
            case Type::kIntegral:
            case Type::kBoolean:;
            default: break;
        }
        return false;
    }

    std::string DumpBase(int tabs, char end, char space, int depth = 0) const {
        int nextdepth = depth+1;
        auto pad = std::string{};
        pad.resize(nextdepth * tabs);
        std::fill(std::begin(pad), std::end(pad), ' ');

        switch (type_) {
            case Type::kNull:
                return "null";
            case Type::kObject:
                {
                    std::ostringstream oss;
                    oss << "{" << end;
                    bool skip = true;
                    for (auto &it : *GetObjectPointer()) {
                        if (!skip) {
                            oss << "," << end;
                        }
                        oss << pad << "\""
                                << it.first
                                <<  "\"" << space << ":" << space
                                << it.second.DumpBase(tabs, end, space, nextdepth);
                        skip = false;
                    }
                    oss << end << pad.erase(0, tabs) << "}";
                    return oss.str();
                }
            case Type::kArray:
                {
                    std::ostringstream oss;
                    oss << "[";
                    bool skip = true;
                    for (auto &it : *GetArrayPointer()) {
                        if (!skip) {
                            oss << "," << space;
                        }
                        oss << it.DumpBase(tabs, end, space, nextdepth);
                        skip = false;
                    }
                    oss << "]";
                    return oss.str();
                }
            case Type::kString:
                return "\"" + JsonEscape(*GetStringPointer()) + "\"";
            case Type::kFloating:
                return std::to_string(ToFloat());
            case Type::kIntegral:
                return std::to_string(ToInt());
            case Type::kBoolean:
                return ToBool() ? "true" : "false";
            default:
                return std::string{};
        }
        return std::string{};
    }

    std::string Dump(int tabs = 4) const {
        return DumpBase(tabs, '\n', ' ');
    }

    std::string DumpLine(char space = ' ') const {
        return DumpBase(0, '\0', space);
    }

private:

    template<typename T>
    T GetDataAs() const {
        T dest;
        std::memcpy(&dest, &data_, sizeof(T));
        return dest;
    }

    template<typename T>
    void CopyToData(T src) {
        std::memcpy(&data_, &src, sizeof(T));
    }

    void SetType(Type type) {
        if (type_ == type) {
            return;
        }

        TryReleasePointer();

        type_ = type;
        switch (type_) {
            case Type::kObject: CopyToData(new Object()); break;
            case Type::kArray: CopyToData(new Array()); break;
            case Type::kString: CopyToData(new String()); break;
            case Type::kFloating:
            case Type::kIntegral:
            case Type::kBoolean:
            case Type::kNull: break;
        }
    }

    void TryReleasePointer() {
        switch (type_) {
            case Type::kObject: delete GetObjectPointer(); break;
            case Type::kArray: delete GetArrayPointer(); break;
            case Type::kString: delete GetStringPointer(); break;
            case Type::kFloating:
            case Type::kIntegral:
            case Type::kBoolean:
            case Type::kNull: break;
        }
        type_ = Type::kNull;
    }

    void CopyFromOther(const Json &other) {
        TryReleasePointer();

        SetType(other.type_);
        switch (type_) {
            case Type::kObject:
                {
                    auto ptr = GetObjectPointer();
                    for (auto &it: *other.GetObjectPointer()) {
                        ptr->insert(it);
                    }
                }
                break;
            case Type::kArray: 
                {
                    auto ptr = GetArrayPointer();
                    for (auto &it : *other.GetArrayPointer()) {
                        ptr->emplace_back(it);
                    }
                }
                break;
            case Type::kString:
                *GetStringPointer() = *other.GetStringPointer();
                break;
            default:
                data_ = other.data_;
                break;
        }
    }

    ObjectPointer GetObjectPointer() const {
        return reinterpret_cast<ObjectPointer>(data_);
    }
    ArrayPointer GetArrayPointer() const {
        return reinterpret_cast<ArrayPointer>(data_);
    }
    StringPointer GetStringPointer() const {
        return reinterpret_cast<StringPointer>(data_);
    }

    std::uint64_t data_{0ULL};
    Type type_{Type::kNull};
};
