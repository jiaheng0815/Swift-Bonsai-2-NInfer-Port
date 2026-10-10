#pragma once

// JSON Schema normalization for constrained decoding.
//
// A schema document is compiled into a flat node graph and then driven as a byte-level pushdown
// automaton by json_grammar.h. The two halves are deliberately separate: this file owns what a
// document *means* (keyword extraction, `$ref` resolution, cycle handling), json_grammar.h owns
// what a byte stream *may do* under it.
//
// Recursion is the reason this is a graph of node ids rather than a tree. A `$ref` back to an
// ancestor resolves to the ancestor's already-assigned id, so a recursive schema reuses one node
// and the automaton's container stack supplies the "how deep am I" state. Nothing here expands a
// recursive definition into a finite approximation.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::text {

// The JSON value kinds a schema node can admit. These mirror `type` keyword spellings one-to-one;
// `integer` is kept distinct from `number` because the automaton must reject `1.5` where only
// `integer` is admitted.
enum class JsonKind : std::uint8_t {
    Null,
    Boolean,
    Object,
    Array,
    Number,
    Integer,
    String,
};

[[nodiscard]] std::string_view json_kind_name(JsonKind kind) noexcept;

// Where a node id refers to no node. Distinct from id 0, which is always the root when a document
// compiles successfully.
inline constexpr std::uint32_t kNoSchemaNode = std::numeric_limits<std::uint32_t>::max();
using SchemaNodeId                          = std::uint32_t;

// How an object treats a member name that `properties` does not declare.
enum class AdditionalProperties : std::uint8_t {
    // `additionalProperties: false`. An undeclared name ends the object's admissible language.
    Denied,
    // `additionalProperties` omitted or true. An undeclared name admits any value; the sibling
    // `additional_schema` is kNoSchemaNode and carries no further constraint.
    Allowed,
    // `additionalProperties: <schema>`. Undeclared names admit that schema's language.
    Schema,
};

// One normalized schema node. Every keyword is resolved to an explicit form here so the automaton
// never consults raw JSON again: an absent `type` becomes an empty kind list meaning "any", an
// absent `items` becomes kNoSchemaNode meaning "any value", and so on.
//
// A field left at its default means the keyword did not constrain that axis.
struct SchemaNode {
    // Admitted value kinds. Empty means every kind is admitted.
    std::vector<JsonKind> kinds;

    // --- string ---
    // Measured in Unicode code points, not bytes: `maxLength: 1` admits "é" as one U+00E9 in either
    // its 2-byte NFC form or its 3-byte NFD form, so the automaton counts code points.
    std::optional<std::size_t> min_length;
    std::optional<std::size_t> max_length;
    // ECMA-262 source, unanchored (JSON Schema semantics: the pattern may match anywhere in the
    // value). Carried verbatim; json_grammar.h compiles it. `nullopt` means unconstrained, which is
    // distinct from an empty pattern, since an empty pattern matches every string.
    std::optional<std::string> pattern;

    // --- number / integer ---
    std::optional<double> minimum;
    std::optional<double> maximum;
    std::optional<double> exclusive_minimum;
    std::optional<double> exclusive_maximum;
    std::optional<double> multiple_of;

    // --- object ---
    std::vector<std::pair<std::string, SchemaNodeId>> properties;
    std::vector<std::string> required;
    AdditionalProperties additional = AdditionalProperties::Allowed;
    SchemaNodeId additional_schema   = kNoSchemaNode;
    std::optional<std::size_t> min_properties;
    std::optional<std::size_t> max_properties;

    // --- array ---
    // Positional element schemas, from the `prefixItems` keyword or from a literal array value.
    // Element i is governed by prefix_items[i] while i < prefix_items.size().
    std::vector<SchemaNodeId> prefix_items;
    // Governs every element at or beyond prefix_items.size(). kNoSchemaNode means `items` was
    // omitted: those elements admit any value, unless `prefix_items` already bounds the length.
    SchemaNodeId items               = kNoSchemaNode;
    std::optional<std::size_t> min_items;
    std::optional<std::size_t> max_items;

    // --- literals ---
    // `enum` and `const` are compiled into literal nodes and referenced from `exact_values`, not
    // stored as serialized text. Matching serialized text would be wrong: JSON admits `{"a": 1}`
    // and `{"a":1}`, and the string "A" may be spelled `A` or `A`. A literal node instead
    // reuses the ordinary keywords so the automaton's normal whitespace and escape handling applies.
    std::vector<SchemaNodeId> exact_values;

    // A string literal's decoded code points, compared after unescaping. Only meaningful on a
    // literal node compiled from a string `enum`/`const` entry.
    std::optional<std::string> exact_string;
    // A boolean literal's value. Only meaningful on a literal node compiled from a boolean entry,
    // where `kinds` alone would admit both `true` and `false`.
    std::optional<bool> exact_boolean;

    // --- combinators ---
    std::vector<SchemaNodeId> any_of;
    std::vector<SchemaNodeId> one_of;
    std::vector<SchemaNodeId> all_of;

    [[nodiscard]] bool admits(JsonKind kind) const noexcept {
        if (kinds.empty()) { return true; }
        // `integer` is a subset of `number`: a node admitting number admits the integer 1, and a
        // node naming both must not require the automaton to know which spelling the model chose.
        if (kind == JsonKind::Integer) {
            return std::find(kinds.begin(), kinds.end(), JsonKind::Integer) != kinds.end() ||
                   std::find(kinds.begin(), kinds.end(), JsonKind::Number) != kinds.end();
        }
        return std::find(kinds.begin(), kinds.end(), kind) != kinds.end();
    }

    [[nodiscard]] SchemaNodeId property(const std::string& name) const noexcept {
        for (const auto& entry : properties) {
            if (entry.first == name) { return entry.second; }
        }
        return kNoSchemaNode;
    }

    [[nodiscard]] bool is_required(const std::string& name) const noexcept {
        return std::find(required.begin(), required.end(), name) != required.end();
    }
};

// A compiled schema document. Immutable after compile().
class JsonSchema {
public:
    // Parses `document` as a JSON Schema, resolves every `$ref` against the document root, and
    // returns the normalized graph.
    //
    // Throws std::invalid_argument with a message naming the offending JSON Pointer when the
    // document is not an object, when a keyword has the wrong JSON type, when a `$ref` cannot be
    // resolved, or when the document exceeds a structural limit.
    [[nodiscard]] static JsonSchema compile(std::string_view document);

    JsonSchema(JsonSchema&&) noexcept;
    JsonSchema& operator=(JsonSchema&&) noexcept;
    ~JsonSchema();

    JsonSchema(const JsonSchema&)            = delete;
    JsonSchema& operator=(const JsonSchema&) = delete;

    [[nodiscard]] SchemaNodeId root() const noexcept;
    [[nodiscard]] const SchemaNode& node(SchemaNodeId id) const;
    [[nodiscard]] std::size_t node_count() const noexcept;

    // Keywords present in the document that this compiler does not model. Empty when the document
    // stays inside the implemented subset. A caller that promises a hard guarantee treats a
    // non-empty result as a reason to reject the request rather than accept a weaker constraint.
    [[nodiscard]] const std::vector<std::string>& unsupported_keywords() const noexcept;

private:
    class Impl;
    explicit JsonSchema(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::text
