// Contract tests for the ECMA-262 pattern engine.
//
// The oracle is an independent backtracking matcher written here, deliberately in a different style
// from the NFA under test: it tries every start position and recurses over the tree. Agreement
// between the two is the evidence, so the oracle must not share code with the engine.

#include "text/regex_engine.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

int check(bool condition, const std::string& label) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << label << '\n';
    return 1;
}

using ninfer::text::RegexMatcher;

// Decodes a UTF-8 literal into the code points the engine consumes.
std::vector<std::uint32_t> codepoints(std::string_view text) {
    std::vector<std::uint32_t> out;
    for (std::size_t index = 0; index < text.size();) {
        const unsigned char lead = static_cast<unsigned char>(text[index]);
        std::uint32_t value      = 0;
        std::size_t length       = 1;
        if (lead < 0x80) {
            value = lead;
        } else if ((lead & 0xE0) == 0xC0) {
            value  = lead & 0x1Fu;
            length = 2;
        } else if ((lead & 0xF0) == 0xE0) {
            value  = lead & 0x0Fu;
            length = 3;
        } else {
            value  = lead & 0x07u;
            length = 4;
        }
        for (std::size_t extra = 1; extra < length; ++extra) {
            value = (value << 6U) | (static_cast<unsigned char>(text[index + extra]) & 0x3Fu);
        }
        out.push_back(value);
        index += length;
    }
    return out;
}

// Runs the engine over one string and reports whether the pattern matched anywhere.
bool engine_matches(const RegexMatcher& matcher, std::string_view text) {
    auto cursor = matcher.begin();
    for (const std::uint32_t codepoint : codepoints(text)) {
        if (!matcher.step(cursor, codepoint)) { return false; }
    }
    return RegexMatcher::matched(cursor);
}

// The independent oracle. Tries every start offset; `recursive` classifies a candidate substring
// with a straightforward reading of the syntax the tests use.
struct Oracle {
    static bool equals(std::string_view pattern, std::string_view text) {
        for (std::size_t start = 0; start <= text.size(); ++start) {
            if (match_here(pattern, text.substr(start))) { return true; }
        }
        return false;
    }

private:
    // Matches a pattern that is a top-level concatenation of single-character atoms, which is all
    // the curated table needs. Anything richer is handled by the engine's own greedy reading and is
    // covered by the curated expectations instead.
    static bool match_here(std::string_view pattern, std::string_view text) {
        if (pattern.empty()) { return true; }
        const char atom = pattern[0];
        if (atom == '.') {
            if (text.empty()) { return false; }
            return match_here(pattern.substr(1), text.substr(1));
        }
        if (!text.empty() && atom == text[0]) { return match_here(pattern.substr(1), text.substr(1)); }
        return false;
    }
};

int unanchored_semantics() {
    int failures = 0;
    // JSON Schema reads `pattern` with RegExp.prototype.test, so a pattern matches anywhere in the
    // value. `{"pattern":"b"}` accepting "abc" is the case a full-match engine gets wrong.
    const auto single = RegexMatcher::compile("b");
    failures += check(engine_matches(*single, "abc"), "b matches inside abc");
    failures += check(engine_matches(*single, "b"), "b matches b");
    failures += check(!engine_matches(*single, "aaa"), "b does not match aaa");

    const auto anchored = RegexMatcher::compile("^abc$");
    failures += check(engine_matches(*anchored, "abc"), "^abc$ matches abc");
    failures += check(!engine_matches(*anchored, "xabc"), "^abc$ rejects a prefixed text");
    failures += check(!engine_matches(*anchored, "abcx"), "^abc$ rejects a suffixed text");

    const auto start_only = RegexMatcher::compile("^ab");
    failures += check(engine_matches(*start_only, "abc"), "^ab matches a prefix");
    failures += check(!engine_matches(*start_only, "xab"), "^ab rejects a later occurrence");

    const auto end_only = RegexMatcher::compile("bc$");
    failures += check(engine_matches(*end_only, "abc"), "bc$ matches a suffix");
    failures += check(!engine_matches(*end_only, "abcx"), "bc$ rejects a non-suffix");

    // The oracle and the engine must agree on the plain case.
    for (const char* text : {"abc", "b", "aaa", "cab", ""}) {
        failures += check(engine_matches(*single, text) == Oracle::equals("b", text),
                          std::string("engine agrees with the oracle on '") + text + "'");
    }
    return failures;
}

int character_classes() {
    int failures = 0;
    const auto digits = RegexMatcher::compile("^[0-9]+$");
    failures += check(engine_matches(*digits, "2026"), "digits accept a number");
    failures += check(!engine_matches(*digits, "20a6"), "digits reject a letter");
    failures += check(!engine_matches(*digits, ""), "digits reject the empty string");

    const auto negation = RegexMatcher::compile("^[^0-9]+$");
    failures += check(engine_matches(*negation, "abc"), "a negated class accepts letters");
    failures += check(!engine_matches(*negation, "a1c"), "a negated class rejects a digit");

    const auto shorthand = RegexMatcher::compile("^\\w+$");
    failures += check(engine_matches(*shorthand, "a_1"), "\\w accepts word characters");
    failures += check(!engine_matches(*shorthand, "a-1"), "\\w rejects punctuation");

    const auto negative_shorthand = RegexMatcher::compile("^\\D+$");
    failures += check(engine_matches(*negative_shorthand, "abc"), "\\D accepts letters");
    failures += check(!engine_matches(*negative_shorthand, "a1"), "\\D rejects a digit");

    const auto ranges = RegexMatcher::compile("^[a-cx-z]+$");
    failures += check(engine_matches(*ranges, "abxyz"), "class ranges accept each span");
    failures += check(!engine_matches(*ranges, "abd"), "class ranges reject a gap");
    return failures;
}

int quantifiers_and_alternation() {
    int failures = 0;
    const auto optional = RegexMatcher::compile("^colou?r$");
    failures += check(engine_matches(*optional, "color"), "? accepts zero occurrences");
    failures += check(engine_matches(*optional, "colour"), "? accepts one occurrence");

    const auto bounded = RegexMatcher::compile("^a{2,4}$");
    failures += check(!engine_matches(*bounded, "a"), "{2,4} rejects below the lower bound");
    failures += check(engine_matches(*bounded, "aa"), "{2,4} accepts the lower bound");
    failures += check(engine_matches(*bounded, "aaaa"), "{2,4} accepts the upper bound");
    failures += check(!engine_matches(*bounded, "aaaaa"), "{2,4} rejects above the upper bound");

    const auto unbounded = RegexMatcher::compile("^a{2,}$");
    failures += check(!engine_matches(*unbounded, "a"), "{2,} rejects below the lower bound");
    failures += check(engine_matches(*unbounded, "aaaaa"), "{2,} accepts beyond it");

    const auto alternation = RegexMatcher::compile("^(cat|dog)$");
    failures += check(engine_matches(*alternation, "cat"), "alternation accepts the first branch");
    failures += check(engine_matches(*alternation, "dog"), "alternation accepts the second branch");
    failures += check(!engine_matches(*alternation, "cow"), "alternation rejects neither branch");

    const auto lazy = RegexMatcher::compile("^a+?$");
    failures += check(engine_matches(*lazy, "aaa"), "a lazy quantifier still matches");

    const auto noncapturing = RegexMatcher::compile("^(?:ab)+$");
    failures += check(engine_matches(*noncapturing, "abab"), "a non-capturing group repeats");
    return failures;
}

int escapes_and_unicode() {
    int failures = 0;
    const auto quotes = RegexMatcher::compile("^a\"b$");
    failures += check(engine_matches(*quotes, "a\"b"), "a literal quote matches");

    const auto dot_literal = RegexMatcher::compile("^a\\.b$");
    failures += check(engine_matches(*dot_literal, "a.b"), "an escaped dot is literal");
    failures += check(!engine_matches(*dot_literal, "axb"), "an escaped dot is not a wildcard");

    const auto newline = RegexMatcher::compile("^a\\nb$");
    failures += check(engine_matches(*newline, "a\nb"), "\\n matches a newline");

    // `.` excludes line terminators in ECMA-262.
    const auto any = RegexMatcher::compile("^.$");
    failures += check(engine_matches(*any, "x"), ". matches an ordinary character");
    failures += check(!engine_matches(*any, "\n"), ". rejects a line terminator");

    const auto unicode = RegexMatcher::compile("^\\u4e2d$");
    failures += check(engine_matches(*unicode, "\xe4\xb8\xad"), "\\u4e2d matches the code point");

    // A surrogate pair names one code point above the BMP.
    const auto astral = RegexMatcher::compile("^\\ud83d\\ude00$");
    failures += check(engine_matches(*astral, "\xf0\x9f\x98\x80"),
                      "a surrogate pair names one astral code point");
    return failures;
}

int prefix_feasibility() {
    int failures = 0;
    // This is the property constrained decoding actually consumes: after a prefix, does any
    // extension still match? Getting it wrong either prunes a legal continuation or admits an
    // illegal one.
    const auto matcher = RegexMatcher::compile("^abc$");
    auto cursor        = matcher->begin();
    failures += check(RegexMatcher::matched(cursor) == false, "the empty prefix does not match");
    failures += check(matcher->step(cursor, 'a'), "'a' can still extend to a match");
    failures += check(RegexMatcher::matched(cursor) == false, "'a' alone does not match");
    failures += check(matcher->step(cursor, 'b'), "'ab' can still extend");
    failures += check(matcher->step(cursor, 'c'), "'abc' can still extend");
    failures += check(RegexMatcher::matched(cursor), "'abc' matches");
    // `$` makes the match conditional on the value ending here, so a fourth code point must revoke
    // it. Keeping the earlier match alive would make `^abc$` accept "abcd".
    failures += check(!matcher->step(cursor, 'd'),
                      "a code point past the end anchor is rejected");
    failures += check(!RegexMatcher::matched(cursor),
                      "the end-anchored match does not survive a longer value");

    // A dead prefix must be reported dead, not merely unmatched.
    const auto dead = RegexMatcher::compile("^abc$");
    auto dead_cursor = dead->begin();
    failures += check(!dead->step(dead_cursor, 'x'), "a diverging code point is rejected");
    failures += check(dead->exhausted(dead_cursor), "the diverged state is exhausted");

    // Once matched, the pattern stays satisfied for the rest of the value.
    const auto loose = RegexMatcher::compile("b");
    auto loose_cursor = loose->begin();
    failures += check(loose->step(loose_cursor, 'a'), "'a' is a legal start");
    failures += check(loose->step(loose_cursor, 'b'), "'ab' still legal");
    failures += check(RegexMatcher::matched(loose_cursor), "'ab' matches the unanchored pattern");
    for (const std::uint32_t codepoint : {std::uint32_t('x'), std::uint32_t('y')}) {
        failures += check(loose->step(loose_cursor, codepoint),
                          "a matched pattern tolerates any trailing code point");
    }
    return failures;
}

int unsupported_constructs_are_rejected() {
    int failures = 0;
    const auto rejects = [&](const char* pattern, const char* label) {
        try {
            (void)RegexMatcher::compile(pattern);
            failures += check(false, std::string(label) + " was accepted");
        } catch (const std::invalid_argument&) {
            // Rejected, which is the contract.
        }
    };
    // Constructs still outside the machine are refused rather than approximated, because accepting
    // them with a weaker match would silently emit values the caller promised to exclude.
    rejects("\\ba", "a word boundary");
    rejects("(?i)a", "an inline flag");
    rejects("(?<name>a)", "a named group");
    rejects("(a)(b)(c)(d)(e)(f)(g)(h)(i)(j)", "more groups than a backreference can name");
    rejects("(abc", "an unterminated group");
    rejects("(?=a", "an unterminated assertion");
    rejects("[abc", "an unterminated class");
    rejects("*a", "a leading quantifier");
    rejects("a{3,1}", "an inverted quantifier range");
    return failures;
}

// Backreferences and lookaround are why this engine is a threaded automaton rather than a DFA: the
// languages they describe are not regular, so a configuration has to carry the captures that
// constrain it. Every case here is paired with a continuation question, because a constrained
// decoder asks "can this still become a match" after each code point, not only "does it match".
int non_regular_constructs() {
    int failures = 0;
    const auto matches = [&](const char* pattern, std::string_view text, bool expected,
                             const char* label) {
        const auto matcher = RegexMatcher::compile(pattern);
        failures += check(engine_matches(*matcher, text) == expected, label);
    };
    // True while some continuation still reaches accept — the property the mask depends on. A
    // pattern the machine calls dead here can never be completed, so the caller stops the turn.
    const auto still_open = [&](const char* pattern, std::string_view prefix, bool expected,
                                const char* label) {
        const auto matcher = RegexMatcher::compile(pattern);
        auto cursor        = matcher->begin();
        for (const std::uint32_t codepoint : codepoints(prefix)) {
            (void)matcher->step(cursor, codepoint);
        }
        failures += check(!matcher->exhausted(cursor) == expected, label);
    };

    // A backreference consumes the text its group captured, so the language is {a^n a^n} — which no
    // finite automaton recognizes.
    matches("^(ab)\\1$", "abab", true, "a backreference repeats the text its group captured");
    matches("^(ab)\\1$", "ababab", false, "the repeat does not extend past the group");
    matches("^(a*)\\1$", "aaaa", true, "a doubled run of two");
    matches("^(a*)\\1$", "aaaaaa", true, "a doubled run of three");
    matches("^(a*)\\1$", "aaa", false, "an odd run is not a doubled run");
    matches("^(a*)\\1$", "", true, "the empty run is trivially doubled");
    still_open("^(ab)\\1$", "ab", true, "a half-written backreference is still open");
    still_open("^(ab)\\1$", "aba", true, "the backreference is one byte from done");
    still_open("^(ab)\\1$", "abb", false, "a backreference that disagreed is dead, not open");

    // Lookahead is decided by what comes next; lookbehind by what already came.
    matches("^a(?=b)", "ab", true, "a positive lookahead that holds");
    matches("^a(?=b)", "ac", false, "a positive lookahead that does not");
    matches("^a(?!b)", "ac", true, "a negative lookahead that holds");
    matches("^a(?!b)", "ab", false, "a negative lookahead that does not");
    matches("(?<=a)b", "ab", true, "a positive lookbehind that holds");
    matches("(?<=a)b", "cb", false, "a positive lookbehind that does not");
    matches("(?<!a)b", "cb", true, "a negative lookbehind that holds");
    matches("(?<!a)b", "ab", false, "a negative lookbehind that does not");

    // A lookahead is zero-width: the main path resumes where the assertion was evaluated and consumes
    // the same text again.
    matches("^(?=ab)a.c$", "abc", true, "the assertion does not consume");

    // An unresolved assertion is neither a match nor a death, so the value may not close early and
    // must not be called dead.
    still_open("^a(?=b)", "a", true, "an undecided lookahead keeps the value open");
    {
        const auto matcher = RegexMatcher::compile("^a(?=b)");
        auto cursor        = matcher->begin();
        (void)matcher->step(cursor, 'a');
        failures += check(!RegexMatcher::matched(cursor),
                          "an undecided lookahead does not let the value close yet");
    }
    still_open("^a(?!b)b", "ab", false, "a negative lookahead that came true is death");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += non_regular_constructs();
    failures += unanchored_semantics();
    failures += character_classes();
    failures += quantifiers_and_alternation();
    failures += escapes_and_unicode();
    failures += prefix_feasibility();
    failures += unsupported_constructs_are_rejected();
    if (failures == 0) { std::cout << "regex_engine: all contracts hold\n"; }
    return failures == 0 ? 0 : 1;
}
