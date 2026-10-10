#pragma once

// The byte-level pushdown automaton that decides what a schema-constrained stream may emit next.
//
// Three properties of JSON Schema force the shape of this machine:
//
//   * Values nest, so recognition needs a stack, not a DFA. A recursive `$ref` grows the stack
//     instead of the schema.
//   * `anyOf`/`oneOf` branch, so a prefix can have several live readings at once. The machine
//     therefore carries a *set* of configurations and stays alive while any of them is.
//   * String constraints address the decoded value, not the wire bytes. `maxLength` counts code
//     points and `pattern` matches against the unescaped text, so the string state decodes escapes
//     before it consults either. Matching raw bytes here is the classic way to pass every hand-made
//     example and fail on real payloads.

#include "text/json_schema.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ninfer::text {

// A set of byte values, one bit per value.
class ByteSet {
public:
    void insert(std::uint8_t byte) noexcept;
    // Inclusive range, used for the ASCII classes that JSON punctuation and digits occupy.
    void insert_range(std::uint8_t low, std::uint8_t high) noexcept;
    void insert_all() noexcept;
    void remove(std::uint8_t byte) noexcept;
    void union_with(const ByteSet& other) noexcept;
    [[nodiscard]] bool contains(std::uint8_t byte) const noexcept;
    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] bool full() const noexcept;
    [[nodiscard]] std::size_t count() const noexcept;
    [[nodiscard]] const std::array<std::uint64_t, 4>& words() const noexcept { return words_; }

private:
    std::array<std::uint64_t, 4> words_{};
};

// A grammar instance is single-use state: construct one per in-flight constrained request and feed
// it the bytes that request commits.
class JsonGrammar {
public:
    // Compiles the slot views the walk needs. Throws std::invalid_argument when a node combines
    // two structural sources (see merge_slots in the implementation), which this machine cannot
    // express soundly.
    explicit JsonGrammar(const JsonSchema& schema);

    JsonGrammar(JsonGrammar&&) noexcept;
    JsonGrammar& operator=(JsonGrammar&&) noexcept;
    ~JsonGrammar();

    // Copyable, because building a token mask walks the vocabulary trie depth-first and has to
    // restore the machine at every branch point. The compiled pattern programs are shared, so a copy
    // only duplicates the live configuration set.
    JsonGrammar(const JsonGrammar& other);
    JsonGrammar& operator=(const JsonGrammar& other);

    // Consumes one byte. Returns false when no live reading admits it, in which case the machine is
    // left untouched and every later feed() also returns false. Callers holding a byte that came
    // from next_bytes() never observe a false here.
    bool feed(std::uint8_t byte);

    // The bytes any live reading admits next. Empty exactly when the machine is dead.
    [[nodiscard]] ByteSet next_bytes() const;

    // True when the bytes consumed so far are a complete value the schema admits. A complete
    // document stays complete: whitespace after the root value is allowed and does not un-complete
    // it, which is what lets the mask at completion be the stop-token set.
    [[nodiscard]] bool complete() const;

    // True when at least one reading is still alive.
    // Restarts the machine *inside* a string's content, as if the opening quote had been consumed.
    //
    // A tool argument carries its string value as raw text rather than as a JSON literal, so the same
    // keywords (`enum`, `const`, `pattern`, `minLength`, `maxLength`) have to apply to content that
    // never had a quote in front of it. Everything else about the state is what `begin()` produces;
    // only the phase and the pattern cursors differ.
    void begin_string_content();

    [[nodiscard]] bool viable() const noexcept;

    void reset();

    // Number of live readings. Exposed for tests and for diagnosing a schema whose branching
    // explodes; not part of the decoding contract.
    [[nodiscard]] std::size_t configuration_count() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::text
