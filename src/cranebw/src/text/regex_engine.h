#pragma once

// ECMA-262 pattern matching for constrained decoding.
//
// The question this engine answers is not "does this string match" but "can this *prefix* still
// become a match", because the decoder asks it after every emitted code point. A backtracking
// matcher cannot answer that without exploring every extension, so the implementation is a Thompson
// NFA whose live thread set *is* the answer: a non-empty set means some extension still reaches
// accept.
//
// Patterns are matched in JSON Schema's unanchored sense, which is `RegExp.prototype.test` and not
// a full-match: `{"pattern":"b"}` accepts "abc". The machine therefore runs in search mode, where a
// fresh start thread is seeded at every position.
//
// Alphabet: code points, not bytes. JSON Schema string keywords constrain the decoded value, so the
// caller unescapes `\uXXXX` and friends before feeding anything here, and `minLength`/`maxLength`
// count the same units.

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>
#include <vector>

namespace ninfer::text {

// A matching engine for one pattern. Immutable, so one instance serves every request that shares
// the schema.
// ECMA-262 numbers backreferences up to `\9`, and the same bound keeps the per-thread capture state a
// fixed-size array instead of a heap allocation on every step of every thread.
constexpr std::uint32_t kMaxCaptureGroups = 9;
constexpr std::uint32_t kUnsetCapture     = 0xffffffffU;

class RegexMatcher {
public:
    // Compiles `pattern`. Throws std::invalid_argument naming the construct when the pattern uses
    // something outside the implemented subset, so a caller promising a hard guarantee can reject
    // the request instead of silently enforcing less than was asked.
    //
    // Implemented: literals, `.`, `[...]` classes with ranges and negation, `\d \D \w \W \s \S`,
    // character escapes, `\uXXXX`, quantifiers `* + ? {n} {n,} {n,m}` and their lazy forms, groups
    // `(...)` and `(?:...)`, alternation, the position anchors `^` and `$`, lookaround
    // `(?=...) (?!...) (?<=...) (?<!...)`, and backreferences `\1`..`\9`.
    //
    // Lookaround and backreferences are non-regular — `(a*)\1` recognizes {a^n a^n}, which no finite
    // automaton can — so a live configuration here is a program counter *plus* the captures that
    // thread has recorded, and the code points consumed so far are retained so a lookbehind can be
    // decided against the past. A backreference consumes its group's text one code point at a time,
    // so the state stays finite and the "can this still match" question keeps its exact answer.
    //
    // Not implemented, and rejected rather than approximated: `\b` and `\B`, inline flags `(?i)`, and
    // named groups `(?<name>...)`. A word boundary is decided by the code point *after* the position,
    // and this machine computes its epsilon closure before the next code point is known.
    [[nodiscard]] static std::unique_ptr<RegexMatcher> compile(std::string_view pattern);

    RegexMatcher(RegexMatcher&&) noexcept;
    RegexMatcher& operator=(RegexMatcher&&) noexcept;
    ~RegexMatcher();

    RegexMatcher(const RegexMatcher&)            = delete;
    RegexMatcher& operator=(const RegexMatcher&) = delete;

    // The state of a search after some prefix has been consumed.
    //
    // Backreferences and lookaround make the accepted language non-regular, so a configuration is not
    // a program counter alone: it also carries the capture spans that thread has recorded, and the
    // code points consumed so far are retained so a lookbehind can be decided against the past. Both
    // are bounded by the input, which is what keeps the configuration set finite.
    class Cursor {
    public:
        Cursor() = default;

        struct Thread {
            std::uint32_t pc = 0;
            // 2n = start of group n+1, 2n+1 = its end. `kUnsetCapture` means "has not matched", and a
            // backreference to such a group matches the empty string, as ECMA-262 specifies.
            std::array<std::uint32_t, 2 * kMaxCaptureGroups> groups{};
            // Code points of a backreference already matched, so a `\1` longer than one code point is
            // consumed one at a time.
            std::uint32_t backref_progress = 0;
        };
        // A lookahead cannot be decided from the past alone, so the thread that reached it waits here
        // until the sub-match resolves. It keeps the position it was parked at, because a zero-width
        // assertion resumes the main match *there* — the code points that arrived while it waited
        // have to be consumed by the main path afterwards, not skipped over.
        struct Parked {
            std::uint32_t pc = 0;
            std::uint32_t at = 0;
            bool positive    = true;
            std::array<std::uint32_t, 2 * kMaxCaptureGroups> groups{};
        };

    private:
        std::vector<Thread> threads_;
        std::vector<Parked> parked_;
        // Code points consumed so far, by index. A backreference reads the text its group captured and
        // a lookbehind reads the text before the current position; both are already in here.
        std::vector<std::uint32_t> input_;
        // Some position at or before the current one reached accept.
        bool matched_ = false;
        // A live path reaches accept exactly if the value ends here, which is how `$` is honored
        // without knowing the length in advance.
        bool end_capable_ = false;
        // Code points consumed so far; `^` is decided against it.
        std::size_t position_ = 0;
        friend class RegexMatcher;
    };

    // The state before any code point.
    [[nodiscard]] Cursor begin() const;

    // Consumes one code point. Returns false when no live thread survives, which means the value can
    // never match however it continues.
    bool step(Cursor& cursor, std::uint32_t codepoint) const;

    // True once the pattern has matched somewhere in the consumed prefix. Sticky: a later code point
    // cannot un-match it, and JSON Schema only asks whether the pattern occurs at all.
    [[nodiscard]] static bool matched(const Cursor& cursor) noexcept;

    // True when no further code point can produce a match. Equivalent to `!step` having returned
    // false, offered so callers can avoid probing with a sentinel.
    [[nodiscard]] bool exhausted(const Cursor& cursor) const noexcept;

private:
    RegexMatcher();
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::text
