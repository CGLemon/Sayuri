#include <string>
#include <vector>

#include "tests/unitest.h"
#include "utils/option.h"

namespace {

std::vector<char*> BuildArgv(std::vector<std::string>& args) {
    std::vector<char*> argv;
    argv.reserve(args.size());
    for (auto& arg : args) {
        argv.emplace_back(arg.data());
    }
    return argv;
}

UNITEST(Option, StoresDefaultValue) {
    OptionsMap options;
    options << RegisterOption({"--value"}, "value", 7);
    Option& option = options.at("value");

    EXPECT_EQ(option.Get<int>(), 7);
    EXPECT_EQ(option.Count(), 1);
    EXPECT_TRUE(option.IsDefault());
}

UNITEST(Option, SetReplacesDefaultAndKeepsHistory) {
    OptionsMap options;
    options << RegisterOption({"--value"}, "value", std::string("default"));
    Option& option = options.at("value");

    option.Set(std::string("first"));
    option.Set(std::string("second"));

    EXPECT_EQ(option.Get<std::string>(), std::string("second"));
    EXPECT_EQ(option.Get<std::string>(0), std::string("first"));
    EXPECT_EQ(option.Count(), 2);
    EXPECT_FALSE(option.IsDefault());
}

UNITEST(Option, RangeClampsNumericValues) {
    OptionsMap options;
    options << RegisterOption({"--value"}, "value", 5).Range(1, 10);
    Option& option = options.at("value");

    EXPECT_EQ(option.Get<int>(), 5);

    option.Set(99);
    EXPECT_EQ(option.Get<int>(), 10);

    option.Set(-3);
    EXPECT_EQ(option.Get<int>(), 1);
}

UNITEST(Option, ChoicesUseRegisteredNames) {
    OptionsMap options;
    options << RegisterOption({"--mode"}, "mode", std::string("gtp"))
                   .Choices<std::string>({"gtp", "test"}, {"gtp", "test"});
    Option& option = options.at("mode");

    option.SetFromString("test");

    EXPECT_EQ(option.Get<std::string>(), std::string("test"));
    EXPECT_EQ(option.ToString(), std::string("test, Choices: {gtp, test}"));
    EXPECT_THROW(option.SetFromString("benchmark"));
}

UNITEST(OptionsMap, ParsesFlagsAndBooleanNoValue) {
    OptionsMap options;
    options << RegisterOption({"--name"}, "name", std::string("default"));
    options << RegisterOption({"--enabled"}, "enabled", false);

    std::vector<std::string> args{"sayuri", "--name", "unitest", "--enabled"};
    auto argv = BuildArgv(args);
    options.ParseArgs(static_cast<int>(argv.size()), argv.data());

    EXPECT_EQ(options.at("name").Get<std::string>(), std::string("unitest"));
    EXPECT_TRUE(options.at("enabled").Get<bool>());
}

} // namespace
