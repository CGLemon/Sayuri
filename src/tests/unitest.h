#pragma once

#include <exception>
#include <iostream>
#include <sstream>
#include <string>

namespace unitest {

struct TestStats {
    int passed = 0;
    int failed = 0;
};

using TestFunction = void (*)();

class TestRegistrar {
public:
    TestRegistrar(const char* name, TestFunction function);
};

TestStats& GetTestStats();
void AddPass();
void AddFailure(const char* file, int line, const std::string& message);
int RunAllTests();

inline void ExpectTrue(bool condition, const char* file, int line, const char* expression) {
    if (condition) {
        AddPass();
        return;
    }

    std::ostringstream oss;
    oss << "EXPECT_TRUE(" << expression << ") failed";
    AddFailure(file, line, oss.str());
}

inline void ExpectFalse(bool condition, const char* file, int line, const char* expression) {
    if (!condition) {
        AddPass();
        return;
    }

    std::ostringstream oss;
    oss << "EXPECT_FALSE(" << expression << ") failed";
    AddFailure(file, line, oss.str());
}

template <typename Actual, typename Expected>
void ExpectEq(const Actual& actual,
              const Expected& expected,
              const char* actual_expression,
              const char* expected_expression,
              const char* file,
              int line) {
    if (actual == expected) {
        AddPass();
        return;
    }

    std::ostringstream oss;
    oss << "EXPECT_EQ(" << actual_expression << ", " << expected_expression
        << ") failed  actual=" << actual << "  expected=" << expected;
    AddFailure(file, line, oss.str());
}

template <typename Actual, typename Expected>
void ExpectNe(const Actual& actual,
              const Expected& expected,
              const char* actual_expression,
              const char* expected_expression,
              const char* file,
              int line) {
    if (actual != expected) {
        AddPass();
        return;
    }

    std::ostringstream oss;
    oss << "EXPECT_NE(" << actual_expression << ", " << expected_expression
        << ") failed  actual=" << actual << "  expected=" << expected;
    AddFailure(file, line, oss.str());
}

template <typename Actual, typename Expected, typename Epsilon>
void ExpectNear(const Actual& actual,
                const Expected& expected,
                const Epsilon& epsilon,
                const char* actual_expression,
                const char* expected_expression,
                const char* epsilon_expression,
                const char* file,
                int line) {
    const auto diff = (actual > expected) ? (actual - expected) : (expected - actual);
    if (diff <= epsilon) {
        AddPass();
        return;
    }

    std::ostringstream oss;
    oss << "EXPECT_NEAR(" << actual_expression << ", " << expected_expression << ", "
        << epsilon_expression << ") failed  actual=" << actual << "  expected=" << expected
        << "  diff=" << diff << "  epsilon=" << epsilon;
    AddFailure(file, line, oss.str());
}

inline void ExpectThrow(bool threw, const char* file, int line, const char* expression) {
    if (threw) {
        AddPass();
        return;
    }

    std::ostringstream oss;
    oss << "EXPECT_THROW(" << expression << ") did not throw";
    AddFailure(file, line, oss.str());
}

} // namespace unitest

#define UNITEST_CONCAT_IMPL(a, b) a##b
#define UNITEST_CONCAT(a, b) UNITEST_CONCAT_IMPL(a, b)

#define UNITEST_STRINGIFY_IMPL(x) #x
#define UNITEST_STRINGIFY(x) UNITEST_STRINGIFY_IMPL(x)

#define UNITEST(section, name) UNITEST_IMPL(section, name, __LINE__)
#define UNITEST_IMPL(section, name, line)                                                          \
    static void UNITEST_CONCAT(UnitestCase_, line)();                                              \
    static const ::unitest::TestRegistrar UNITEST_CONCAT(UnitestRegistrar_, line)(                 \
        UNITEST_STRINGIFY(section) "." UNITEST_STRINGIFY(name),                                    \
        &UNITEST_CONCAT(UnitestCase_, line));                                                      \
    static void UNITEST_CONCAT(UnitestCase_, line)()

#define EXPECT_TRUE(cond)                                                                          \
    do {                                                                                           \
        ::unitest::ExpectTrue(static_cast<bool>(cond), __FILE__, __LINE__, #cond);                 \
    } while (0)

#define EXPECT_FALSE(cond)                                                                         \
    do {                                                                                           \
        ::unitest::ExpectFalse(static_cast<bool>(cond), __FILE__, __LINE__, #cond);                \
    } while (0)

#define EXPECT_EQ(actual, expected)                                                                \
    do {                                                                                           \
        const auto& unitest_actual = (actual);                                                     \
        const auto& unitest_expected = (expected);                                                 \
        ::unitest::ExpectEq(                                                                       \
            unitest_actual, unitest_expected, #actual, #expected, __FILE__, __LINE__);             \
    } while (0)

#define EXPECT_NE(actual, expected)                                                                \
    do {                                                                                           \
        const auto& unitest_actual = (actual);                                                     \
        const auto& unitest_expected = (expected);                                                 \
        ::unitest::ExpectNe(                                                                       \
            unitest_actual, unitest_expected, #actual, #expected, __FILE__, __LINE__);             \
    } while (0)

#define EXPECT_NEAR(actual, expected, epsilon)                                                     \
    do {                                                                                           \
        const auto& unitest_actual = (actual);                                                     \
        const auto& unitest_expected = (expected);                                                 \
        const auto& unitest_epsilon = (epsilon);                                                   \
        ::unitest::ExpectNear(unitest_actual,                                                      \
                              unitest_expected,                                                    \
                              unitest_epsilon,                                                     \
                              #actual,                                                             \
                              #expected,                                                           \
                              #epsilon,                                                            \
                              __FILE__,                                                            \
                              __LINE__);                                                           \
    } while (0)

#define EXPECT_THROW(expr)                                                                         \
    do {                                                                                           \
        bool unitest_threw = false;                                                                \
        try {                                                                                      \
            (void)(expr);                                                                          \
        } catch (...) {                                                                            \
            unitest_threw = true;                                                                  \
        }                                                                                          \
        ::unitest::ExpectThrow(unitest_threw, __FILE__, __LINE__, #expr);                          \
    } while (0)

#define EXPECT_NO_THROW(expr)                                                                      \
    do {                                                                                           \
        try {                                                                                      \
            (void)(expr);                                                                          \
            ::unitest::AddPass();                                                                  \
        } catch (const std::exception& unitest_exception) {                                        \
            std::ostringstream unitest_oss;                                                        \
            unitest_oss << "EXPECT_NO_THROW(" #expr ") threw exception: "                          \
                        << unitest_exception.what();                                               \
            ::unitest::AddFailure(__FILE__, __LINE__, unitest_oss.str());                          \
        } catch (...) {                                                                            \
            ::unitest::AddFailure(__FILE__, __LINE__, "EXPECT_NO_THROW(" #expr ") threw");         \
        }                                                                                          \
    } while (0)
