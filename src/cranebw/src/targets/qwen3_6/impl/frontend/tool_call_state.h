#pragma once

// The tag-level constraint for a tool call's arguments.
//
// A `strict` tool promises that the arguments it generates satisfy its declared schema, but this
// template does not write them as JSON: it writes
//
//   <tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call>
//
// so the promise splits into two unlike constraints. The markup is not JSON at all — the only legal
// continuations in the body are `<parameter=` and `</function>`, and a parameter name is only legal
// if the schema declares it and nothing has written it yet. A parameter's *value* is JSON for every
// type but `string`, where it is raw text terminated by the newline before `</parameter>`.
//
// This machine answers the first and delegates the second, because it is the same question the JSON
// automaton already answers: which byte may come next.

#include "text/json_grammar.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::targets::qwen3_6::frontend_internal {

// The declared shape of one tool's arguments, extracted from its `input_schema`.
struct ToolArgumentShape {
    std::vector<std::string> names;
    std::vector<std::uint8_t> required;
    // Each property's own schema, compiled. A `string` property constrains raw text; every other
    // type constrains a JSON literal.
    std::vector<std::shared_ptr<const text::JsonSchema>> values;
    std::vector<std::uint8_t> value_is_text;

    [[nodiscard]] std::size_t index_of(std::string_view name) const noexcept;
};

// Builds the shape from a tool's `input_schema` text. Throws std::invalid_argument when the schema is
// not an object schema, because a `strict` promise it cannot keep is a request the caller must be
// able to reject rather than one to answer under a weaker constraint.
[[nodiscard]] ToolArgumentShape compile_tool_argument_shape(std::string_view schema);

class ToolCallState {
public:
    ToolCallState();
    ToolCallState(const ToolCallState& other);
    ToolCallState& operator=(const ToolCallState& other);
    ~ToolCallState();

    // True while the argument region is under constraint, which is the caller's signal to consult
    // next_bytes() instead of sampling freely.
    [[nodiscard]] bool constrained() const noexcept;

    // The bytes that may come next. Only meaningful while constrained(); a phase that constrains
    // nothing returns the empty set, which the caller reads as "sample freely".
    [[nodiscard]] text::ByteSet next_bytes() const;

    // Takes one byte. False means the byte is not admissible in this state, which is what makes the
    // vocabulary walk prune rather than admit a token that would strand the call.
    bool feed(std::uint8_t byte);

    // The schemas this machine can select from, by tool name.
    void register_tool(std::string name, ToolArgumentShape shape);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::targets::qwen3_6::frontend_internal
