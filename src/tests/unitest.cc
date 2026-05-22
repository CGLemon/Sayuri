#include "tests/unitest.h"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifndef UNITEST_OUTPUT_STREAM
#define UNITEST_OUTPUT_STREAM std::cout
#endif

namespace unitest {
namespace {

struct TestCase {
    std::string name;
    TestFunction function;
};

std::vector<TestCase>& GetRegistry() {
    static std::vector<TestCase> tests;
    return tests;
}

TestStats& MutableTestStats() {
    static TestStats stats;
    return stats;
}

} // namespace

TestRegistrar::TestRegistrar(const char* name, TestFunction function) {
    GetRegistry().push_back({name, function});
}

TestStats& GetTestStats() {
    return MutableTestStats();
}

void AddPass() {
    ++MutableTestStats().passed;
}

void AddFailure(const char* file, int line, const std::string& message) {
    ++MutableTestStats().failed;
    std::cerr << file << ":" << line << "  " << message << "\n";
}

int RunAllTests() {
    TestStats& stats = MutableTestStats();
    stats = TestStats{};

    const auto& tests = GetRegistry();
    int failed_tests = 0;

    UNITEST_OUTPUT_STREAM << "[==========] Running " << tests.size() << " unitest(s).\n";
    for (const auto& test : tests) {
        const auto failed_before = stats.failed;
        UNITEST_OUTPUT_STREAM << "[ RUN      ] " << test.name << "\n";

        try {
            test.function();
        } catch (const std::exception& e) {
            std::ostringstream oss;
            oss << "Unhandled exception in " << test.name << ": " << e.what();
            AddFailure("<unitest>", 0, oss.str());
        } catch (...) {
            std::ostringstream oss;
            oss << "Unhandled unknown exception in " << test.name;
            AddFailure("<unitest>", 0, oss.str());
        }

        if (stats.failed == failed_before) {
            UNITEST_OUTPUT_STREAM << "[       OK ] " << test.name << "\n";
        } else {
            ++failed_tests;
            UNITEST_OUTPUT_STREAM << "[  FAILED  ] " << test.name << "\n";
        }
    }

    UNITEST_OUTPUT_STREAM << "[==========] " << tests.size() << " unitest(s) ran.\n";
    UNITEST_OUTPUT_STREAM << "[  PASSED  ] " << tests.size() - failed_tests << " unitest(s), "
                          << stats.passed << " assertion(s).\n";

    if (failed_tests > 0) {
        UNITEST_OUTPUT_STREAM << "[  FAILED  ] " << failed_tests << " unitest(s), " << stats.failed
                              << " assertion(s).\n";
    }

    return stats.failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

} // namespace unitest
