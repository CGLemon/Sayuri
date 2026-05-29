#include <algorithm>
#include <iterator>
#include <string>
#include <string_view>
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

template <typename T> T TestOptionGet(const Option& option) {
    return option.Get<T>();
}

template <typename T> T TestOptionGet(const Option& option, int idx) {
    return option.Get<T>(idx);
}

template <typename T> void TestOptionSet(Option& option, const T& value) {
    option.Set<T>(value);
}

int TestOptionCount(const Option& option) {
    return option.Count();
}

bool TestOptionIsDefault(const Option& option) {
    return option.IsDefault();
}

void TestOptionSetCurrentAsDefault(Option& option) {
    option.SetCurrentAsDefault();
}

void TestOptionUnique(Option& option) {
    option.Unique();
}

void TestOptionSetFromString(Option& option, std::string_view raw) {
    option.SetFromString(raw);
}

std::string TestOptionToDebugString(const Option& option) {
    return option.ToDebugString();
}

void TestAddProfile(OptionsMap& options, std::string_view name) {
    options.AddProfile(std::string(name));
}

void TestSetProfile(OptionsMap& options, std::string_view name) {
    options.SetProfile(std::string(name));
}

// ---------------------------------------------------------------------------
// Option: default / history semantics
// ---------------------------------------------------------------------------

UNITEST(Option, FirstSetAfterDefaultResetsHistory) {
    OptionsMap options;
    options << RegisterOption({"--value"}, "value", 1);
    Option& option = options.at("value");

    EXPECT_EQ(TestOptionCount(option), 1);
    EXPECT_TRUE(TestOptionIsDefault(option));

    TestOptionSet<int>(option, 2);
    // First Set after default clears the prior history rather than appending,
    // so Count drops back to 1 (the previous default is moved to an internal slot).
    EXPECT_EQ(TestOptionCount(option), 1);
    EXPECT_FALSE(TestOptionIsDefault(option));
    EXPECT_EQ(TestOptionGet<int>(option, 0), 2);
    EXPECT_EQ(TestOptionGet<int>(option), 2);

    TestOptionSet<int>(option, 3);
    // Subsequent Sets append to history.
    EXPECT_EQ(TestOptionCount(option), 2);
    EXPECT_EQ(TestOptionGet<int>(option, 0), 2);
    EXPECT_EQ(TestOptionGet<int>(option), 3);
}

UNITEST(Option, SetCurrentAsDefaultPromotesCurrentValue) {
    OptionsMap options;
    options << RegisterOption({"--value"}, "value", 1);
    Option& option = options.at("value");
    EXPECT_TRUE(TestOptionIsDefault(option));
    EXPECT_EQ(TestOptionCount(option), 1);

    TestOptionSet<int>(option, 42);
    EXPECT_FALSE(TestOptionIsDefault(option));
    EXPECT_EQ(TestOptionCount(option), 1);

    TestOptionSetCurrentAsDefault(option);
    EXPECT_TRUE(TestOptionIsDefault(option));
    EXPECT_EQ(TestOptionGet<int>(option), 42);
    EXPECT_EQ(TestOptionCount(option), 1);

    TestOptionSet<int>(option, 7);
    EXPECT_FALSE(TestOptionIsDefault(option));
    EXPECT_EQ(TestOptionGet<int>(option), 7);
    EXPECT_EQ(TestOptionCount(option), 1);
}

UNITEST(Option, UniqueDeduplicatesHistory) {
    OptionsMap options;
    options << RegisterOption({"--value"}, "value", 1);
    Option& option = options.at("value");
    EXPECT_EQ(TestOptionCount(option), 1);

    TestOptionSet<int>(option, 2);
    EXPECT_EQ(TestOptionCount(option), 1);

    TestOptionSet<int>(option, 2);
    EXPECT_EQ(TestOptionCount(option), 2);

    TestOptionSet<int>(option, 3);
    EXPECT_EQ(TestOptionCount(option), 3);

    TestOptionSet<int>(option, 2);
    EXPECT_EQ(TestOptionCount(option), 4);

    TestOptionUnique(option);
    EXPECT_EQ(TestOptionCount(option), 2);
    EXPECT_EQ(TestOptionGet<int>(option), 2);
}

// ---------------------------------------------------------------------------
// Option: type / range / choices constraints
// ---------------------------------------------------------------------------

UNITEST(Option, GetWithWrongTypeThrows) {
    OptionsMap options;
    options << RegisterOption({"--value"}, "value", 7);
    Option& option = options.at("value");

    EXPECT_THROW(TestOptionGet<std::string>(option));
}

UNITEST(Option, GetWithIndexOutOfRangeThrows) {
    OptionsMap options;
    options << RegisterOption({"--value"}, "value", 7);
    Option& option = options.at("value");

    EXPECT_THROW(TestOptionGet<int>(option, 5));
}

UNITEST(Option, RangeClampsNumericValues) {
    OptionsMap options;
    options << RegisterOption({"--value"}, "value", 99).Range(1, 10);
    Option& option = options.at("value");

    EXPECT_EQ(TestOptionGet<int>(option), 10);

    TestOptionSet<int>(option, 5);
    EXPECT_EQ(TestOptionGet<int>(option), 5);

    TestOptionSet<int>(option, 99);
    EXPECT_EQ(TestOptionGet<int>(option), 10);

    TestOptionSet<int>(option, -3);
    EXPECT_EQ(TestOptionGet<int>(option), 1);
}

UNITEST(Option, ChoicesUseRegisteredNames) {
    OptionsMap options;
    options << RegisterOption({"--mode"}, "mode", std::string("gtp"))
                   .Choices<std::string>({"gtp", "test"}, {"gtp", "test"});
    Option& option = options.at("mode");

    TestOptionSetFromString(option, "test");

    EXPECT_EQ(TestOptionGet<std::string>(option), std::string("test"));
    EXPECT_EQ(TestOptionToDebugString(option), std::string("test, Choices: {gtp, test}"));
    EXPECT_THROW(TestOptionSetFromString(option, "benchmark"));
}

UNITEST(Option, ChoicesRejectNonChoiceValueOnSet) {
    OptionsMap options;
    options << RegisterOption({"--level"}, "level", 1).Choices<int>({"low", "high"}, {1, 9});
    Option& option = options.at("level");

    // Set with a registered choice value is accepted.
    TestOptionSet<int>(option, 9);
    EXPECT_EQ(TestOptionGet<int>(option), 9);

    // Set with a non-choice value throws via EnsureValueHasChoice in Push.
    EXPECT_THROW(TestOptionSet<int>(option, 5));
}

UNITEST(Option, RangeAndChoicesAreMutuallyExclusive) {
    OptionsMap options;

    EXPECT_THROW(
        options
        << RegisterOption({"--a"}, "a", 1).Range(0, 10).Choices<int>({"zero", "one"}, {0, 1}));

    EXPECT_THROW(
        options
        << RegisterOption({"--b"}, "b", 1).Choices<int>({"zero", "one"}, {0, 1}).Range(0, 10));
}

// ---------------------------------------------------------------------------
// Option: string parsing and custom setters
// ---------------------------------------------------------------------------

UNITEST(Option, SetFromStringParsesNumericAndBool) {
    OptionsMap options;
    options << RegisterOption({"--i"}, "i", 0);
    options << RegisterOption({"--f"}, "f", 0.0f);
    options << RegisterOption({"--d"}, "d", 0.0);
    options << RegisterOption({"--b"}, "b", false);
    Option& int_option = options.at("i");
    Option& float_option = options.at("f");
    Option& double_option = options.at("d");
    Option& bool_option = options.at("b");

    TestOptionSetFromString(int_option, "42");
    TestOptionSetFromString(float_option, "1.61803");
    TestOptionSetFromString(double_option, "3.14159");
    TestOptionSetFromString(bool_option, "true");

    EXPECT_EQ(TestOptionGet<int>(int_option), 42);
    EXPECT_EQ(TestOptionGet<float>(float_option), 1.61803f);
    EXPECT_EQ(TestOptionGet<double>(double_option), 3.14159);
    EXPECT_TRUE(TestOptionGet<bool>(bool_option));

    TestOptionSetFromString(bool_option, "false");
    EXPECT_FALSE(TestOptionGet<bool>(bool_option));
}

UNITEST(Option, CustomSetterOverridesParsing) {
    OptionsMap options;
    options << RegisterOption({"--value"}, "value", 0).Setter([](Option& o, std::string_view raw) {
        TestOptionSet<int>(o, static_cast<int>(raw.size()) * 100);
    });
    Option& option = options.at("value");

    TestOptionSetFromString(option, "ab");
    EXPECT_EQ(TestOptionGet<int>(option), 200);

    TestOptionSetFromString(option, "abcd");
    EXPECT_EQ(TestOptionGet<int>(option), 400);
}

UNITEST(OptionsMap, ParseArgsInvokesCustomSetter) {
    OptionsMap options;
    options << RegisterOption({"--count"}, "count", 0).Setter([](Option& o, std::string_view raw) {
        TestOptionSet<int>(o, static_cast<int>(raw.size()));
    });

    std::vector<std::string> args{"sayuri", "--count", "hello"};
    auto argv = BuildArgv(args);
    options.ParseArgs(static_cast<int>(argv.size()), argv.data());

    EXPECT_EQ(TestOptionGet<int>(options.at("count")), 5);
}

// ---------------------------------------------------------------------------
// OptionsMap: basic state and registration validation
// ---------------------------------------------------------------------------

UNITEST(OptionsMap, DuplicateOptionKeyAndFlagThrow) {
    OptionsMap key_options;
    key_options << RegisterOption({"--x"}, "key", 1);
    EXPECT_THROW(key_options << RegisterOption({"--y"}, "key", 2));

    OptionsMap flag_options;
    flag_options << RegisterOption({"--shared"}, "a", 1);
    EXPECT_THROW(flag_options << RegisterOption({"--shared"}, "b", 2));
}

UNITEST(OptionsMap, FlagMustStartWithDashThrows) {
    OptionsMap options;
    EXPECT_THROW(options << RegisterOption({"value"}, "key", 1));
}

// ---------------------------------------------------------------------------
// OptionsMap: ParseArgs
// ---------------------------------------------------------------------------

UNITEST(OptionsMap, ParsesFlagsAndBooleanNoValue) {
    OptionsMap options;
    options << RegisterOption({"--name"}, "name", std::string("default"));
    options << RegisterOption({"--enabled"}, "enabled", false);

    std::vector<std::string> args{"sayuri", "--name", "unitest", "--enabled"};
    auto argv = BuildArgv(args);
    options.ParseArgs(static_cast<int>(argv.size()), argv.data());
    Option& name_option = options.at("name");
    Option& enabled_option = options.at("enabled");

    EXPECT_EQ(TestOptionGet<std::string>(name_option), std::string("unitest"));
    EXPECT_TRUE(TestOptionGet<bool>(enabled_option));
}

// ---------------------------------------------------------------------------
// OptionsMap: Helpers / Groups
// ---------------------------------------------------------------------------

UNITEST(OptionsMap, HelpersToStringGroupsAndFilters) {
    OptionsMap options;
    options << RegisterOption({"--threads"}, "threads", 1).Group("engine").Helper("worker count");
    options << RegisterOption({"--komi"}, "komi", 7.5f).Group("rules").Helper("komi value");
    options << RegisterOption({"--misc"}, "misc", 0).Helper("ungrouped");

    EXPECT_EQ(options.at("threads").Group(), std::string("engine"));

    // Filtering by group lists only that group's flags and header.
    const std::string engine = options.HelpersToString("engine");
    EXPECT_TRUE(engine.find("--threads") != std::string::npos);
    EXPECT_TRUE(engine.find("--komi") == std::string::npos);
    EXPECT_TRUE(engine.find("[engine]") != std::string::npos);

    // Without a filter, every group header appears; ungrouped lands under "default".
    const std::string all = options.HelpersToString();
    EXPECT_TRUE(all.find("[engine]") != std::string::npos);
    EXPECT_TRUE(all.find("[rules]") != std::string::npos);
    EXPECT_TRUE(all.find("[default]") != std::string::npos);
}

// ---------------------------------------------------------------------------
// OptionsMap: Profiles
// ---------------------------------------------------------------------------

UNITEST(OptionsMap, DefaultProfileIsInitiallyActive) {
    OptionsMap options;
    options << RegisterOption({"--value"}, "value", 10);
    Option& option = options.at("value");

    EXPECT_EQ(options.GetProfile(), std::string("default"));
    EXPECT_EQ(options.GetDefaultProfile(), std::string("default"));
    EXPECT_EQ(TestOptionGet<int>(option), 10);
}

UNITEST(OptionsMap, ProfilesAreIndependent) {
    OptionsMap options;
    options << RegisterOption({"--value"}, "value", 1);

    TestAddProfile(options, "alt");
    TestSetProfile(options, "alt");
    Option& alt_option = options.at("value");
    TestOptionSet<int>(alt_option, 99);
    EXPECT_EQ(TestOptionGet<int>(alt_option), 99);

    TestSetProfile(options, "default");
    Option& default_option = options.at("value");
    EXPECT_EQ(TestOptionGet<int>(default_option), 1);
}

UNITEST(OptionsMap, ParseArgsAssignsPerProfileValues) {
    OptionsMap options;
    options << RegisterOption({"--value"}, "value", 0);
    options << RegisterOption({"--gamma"}, "gamma", 0.0f);
    options << RegisterOption({"--beta"}, "beta", 0.0f);
    options << RegisterOption({"--name"}, "name", std::string("base"));

    // "--value 777" (no @) broadcasts to every profile; subsequent @p1/@p2 flags
    // then override "value" for those profiles. "name" is only set per-profile,
    // so the base profile must keep its registration default.
    std::vector<std::string> args{
        "sayuri",     "--value",   "777",       "--gamma", "7.77",      "--value@p1", "11",
        "--value@p2", "22",        "--name@p1", "alice",   "--name@p2", "bob",        "--beta@p1",
        "1.1",        "--beta@p2", "2.2",       "--beta",  "3.3",       "@p1",        "--gamma",
        "0.5",        "@p2",       "--gamma",   "1.5",
    };
    auto argv = BuildArgv(args);
    options.ParseArgs(static_cast<int>(argv.size()), argv.data());

    auto profiles = options.GetProfiles();
    EXPECT_TRUE(std::find(std::begin(profiles), std::end(profiles), std::string("p1")) !=
                std::end(profiles));
    EXPECT_TRUE(std::find(std::begin(profiles), std::end(profiles), std::string("p2")) !=
                std::end(profiles));

    TestSetProfile(options, "p1");
    EXPECT_EQ(TestOptionGet<int>(options.at("value")), 11);
    EXPECT_EQ(TestOptionGet<std::string>(options.at("name")), std::string("alice"));
    EXPECT_EQ(TestOptionGet<float>(options.at("gamma")), 0.5f);
    EXPECT_EQ(TestOptionGet<float>(options.at("beta")), 3.3f);

    TestSetProfile(options, "p2");
    EXPECT_EQ(TestOptionGet<int>(options.at("value")), 22);
    EXPECT_EQ(TestOptionGet<std::string>(options.at("name")), std::string("bob"));
    EXPECT_EQ(TestOptionGet<float>(options.at("gamma")), 1.5f);
    EXPECT_EQ(TestOptionGet<float>(options.at("beta")), 3.3f);

    TestSetProfile(options, "default");
    EXPECT_EQ(TestOptionGet<int>(options.at("value")), 777);
    EXPECT_EQ(TestOptionGet<std::string>(options.at("name")), std::string("base"));
    EXPECT_EQ(TestOptionGet<float>(options.at("gamma")), 7.77f);
    EXPECT_EQ(TestOptionGet<float>(options.at("beta")), 3.3f);
}

UNITEST(OptionsMap, AddProfileInheritsCurrentBaseValues) {
    OptionsMap options;
    options << RegisterOption({"--value"}, "value", 1);
    options << RegisterOption({"--name"}, "name", std::string("base"));

    // Mutate base profile before forking a new one — the new profile must inherit
    // the *current* state, not the original registration defaults.
    TestOptionSet<int>(options.at("value"), 42);

    TestAddProfile(options, "alt");
    TestSetProfile(options, "alt");
    EXPECT_EQ(TestOptionGet<int>(options.at("value")), 42);
    EXPECT_EQ(TestOptionGet<std::string>(options.at("name")), std::string("base"));

    // Modifying a different key on "alt" must not bleed back into "default".
    TestOptionSet<std::string>(options.at("name"), std::string("alt-name"));
    EXPECT_EQ(TestOptionGet<std::string>(options.at("name")), std::string("alt-name"));

    TestSetProfile(options, "default");
    EXPECT_EQ(TestOptionGet<int>(options.at("value")), 42);
    EXPECT_EQ(TestOptionGet<std::string>(options.at("name")), std::string("base"));
}

// ---------------------------------------------------------------------------
// Global helpers operating on kOptionsMap
// ---------------------------------------------------------------------------

UNITEST(Option, SetOptionAsDefaultViaGlobalHelper) {
    // Use unique key to avoid colliding with production options registered in kOptionsMap.
    const std::string key = "__unitest_set_default_key";
    kOptionsMap << RegisterOption({"--__unitest_set_default_flag"}, key, 1);

    EXPECT_TRUE(SetOption<int>(key, 99, /*as_default=*/true));
    EXPECT_TRUE(IsOptionDefault(key));
    EXPECT_EQ(GetOption<int>(key), 99);

    EXPECT_TRUE(SetOption<int>(key, 5));
    EXPECT_FALSE(IsOptionDefault(key));
}

} // namespace
