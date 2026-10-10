#pragma once

// Turning a grammar state into the set of tokens the sampler may pick.
//
// The walk is a depth-first descent of the vocabulary trie, carrying the grammar along the edges.
// An edge is followed only when the grammar admits its byte, so the walk visits exactly the byte
// prefixes the grammar can still become, and a token is admitted exactly when its whole byte string
// is such a prefix. The grammar is copied at each followed edge and the caller's own state is never
// touched, so one mask costs one walk and leaves the request ready for its next round.

#include "targets/qwen3_6/impl/frontend/token_trie.h"
#include "targets/qwen3_6/impl/frontend/tool_call_state.h"
#include "text/json_grammar.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6::frontend_internal {

// An LSB-first bitset over the token domain, laid out exactly as SamplingConfig::token_mask
// expects: token v lives in bit v&7 of byte v>>3.
class TokenMask {
public:
    explicit TokenMask(std::size_t token_domain);

    void clear() noexcept;
    void set(int token) noexcept;
    void unset(int token) noexcept;
    [[nodiscard]] bool test(int token) const noexcept;

    [[nodiscard]] const std::uint8_t* bytes() const noexcept { return bytes_.data(); }
    [[nodiscard]] std::size_t size_bytes() const noexcept { return bytes_.size(); }
    [[nodiscard]] std::size_t count() const noexcept;

    // True when no token at all is admitted. A caller treats this as a defensive failure rather
    // than something to act on: the vocabulary carries a single-byte token for all 256 values, so a
    // state reachable by legal tokens always admits at least one.
    [[nodiscard]] bool empty() const noexcept { return count() == 0; }

private:
    std::vector<std::uint8_t> bytes_;
};

// Fills `mask` with the tokens admissible from `grammar`'s current state.
//
// `completion_tokens` is the request's stop-token set. While the document is incomplete those ids
// are removed from the mask even if the trie would admit their bytes, because a token that ends the
// turn mid-document leaves a truncated value behind; once the document is complete the mask becomes
// exactly that set, which is what lets the model stop at all.
//
// `grammar` is left unchanged. Returns the number of admissible tokens.
// The same trie descent driven by the tool-call state machine instead of a JSON grammar.
//
// A strict tool's arguments are markup, not JSON — the body admits `<parameter=` and `</function>`
// and nothing else — so the constraint cannot be expressed as a schema. The descent is otherwise
// identical: a token is admitted exactly when every byte of it is accepted by the state that the
// bytes before it reach.
std::size_t build_tool_token_mask(const TokenTrie& trie, const ToolCallState& state, TokenMask& mask);

std::size_t build_token_mask(const TokenTrie& trie, const text::JsonGrammar& grammar,
                             std::span<const int> completion_tokens, TokenMask& mask);

} // namespace ninfer::targets::qwen3_6::frontend_internal
