// Vocabulary-triewalk construction of the per-round token mask.

#include "targets/qwen3_6/impl/frontend/token_mask.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace ninfer::targets::qwen3_6::frontend_internal {
namespace {

// Depth-first descent. `state` is the grammar after the bytes that lead to `index` have been fed, so
// every token terminating at `index` is admissible by construction.
//
// Both halves of the admission test are needed, and neither alone is enough.
//
// `next_bytes()` is the stricter of the two: it is the machine's own account of which bytes lead
// somewhere, and it refuses a key byte no declared member name continues. `feed` is the authority on
// whether a byte is *taken*, but it is deliberately permissive in places where the announcement is
// already narrow, so a walk that consulted only `feed` would admit the whole vocabulary at a
// property name. Testing the set first and the machine second admits exactly the tokens both agree
// on, which is what the sampler may be promised.
void walk(const TokenTrie& trie, const text::JsonGrammar& state, std::uint32_t index,
          std::size_t depth, TokenMask& mask) {
    if (depth > 4096) {
        // The trie is acyclic and its depth is bounded by the longest token, so this only trips on a
        // corrupted structure. Refusing to recurse further keeps a bad input from exhausting the
        // stack.
        return;
    }
    const TokenTrie::Node& node = trie.node(index);
    if (node.token >= 0) { mask.set(node.token); }

    const text::ByteSet announced = state.next_bytes();
    for (const TokenTrie::Edge& edge : trie.edges(index)) {
        if (!announced.contains(edge.byte)) { continue; }
        text::JsonGrammar next = state;
        if (!next.feed(edge.byte)) { continue; }
        walk(trie, next, edge.child, depth + 1, mask);
    }
}

// The same descent, driven by the tool-call state instead of a grammar. The state is copied per
// followed edge for the same reason the grammar is: the caller's state must be exactly where it was
// when the walk returns.
void walk_tool(const TokenTrie& trie, const ToolCallState& state, std::uint32_t index,
               std::size_t depth, TokenMask& mask) {
    if (depth > 4096) { return; }
    const TokenTrie::Node& node = trie.node(index);
    if (node.token >= 0) { mask.set(node.token); }

    const text::ByteSet announced = state.next_bytes();
    for (const TokenTrie::Edge& edge : trie.edges(index)) {
        if (!announced.contains(edge.byte)) { continue; }
        ToolCallState next = state;
        if (!next.feed(edge.byte)) { continue; }
        walk_tool(trie, next, edge.child, depth + 1, mask);
    }
}

} // namespace

TokenMask::TokenMask(std::size_t token_domain) : bytes_((token_domain + 7U) / 8U, 0) {}

void TokenMask::clear() noexcept { std::fill(bytes_.begin(), bytes_.end(), std::uint8_t{0}); }

void TokenMask::set(int token) noexcept {
    if (token < 0) { return; }
    const std::size_t index = static_cast<std::size_t>(token);
    if (index / 8U >= bytes_.size()) { return; }
    bytes_[index / 8U] |= static_cast<std::uint8_t>(1U << (index & 7U));
}

void TokenMask::unset(int token) noexcept {
    if (token < 0) { return; }
    const std::size_t index = static_cast<std::size_t>(token);
    if (index / 8U >= bytes_.size()) { return; }
    bytes_[index / 8U] &= static_cast<std::uint8_t>(~(1U << (index & 7U)));
}

bool TokenMask::test(int token) const noexcept {
    if (token < 0) { return false; }
    const std::size_t index = static_cast<std::size_t>(token);
    if (index / 8U >= bytes_.size()) { return false; }
    return (bytes_[index / 8U] & static_cast<std::uint8_t>(1U << (index & 7U))) != 0;
}

std::size_t TokenMask::count() const noexcept {
    std::size_t total = 0;
    for (const std::uint8_t byte : bytes_) {
        std::uint8_t value = byte;
        while (value != 0) {
            value &= static_cast<std::uint8_t>(value - 1U);
            ++total;
        }
    }
    return total;
}

std::size_t build_tool_token_mask(const TokenTrie& trie, const ToolCallState& state,
                                  TokenMask& mask) {
    mask.clear();
    walk_tool(trie, state, trie.root(), 0, mask);
    return mask.count();
}

std::size_t build_token_mask(const TokenTrie& trie, const text::JsonGrammar& grammar,
                             std::span<const int> completion_tokens, TokenMask& mask) {
    mask.clear();

    if (grammar.complete()) {
        // The document is finished, so the only legal continuation is to stop. Anything else would
        // append to a completed value.
        for (const int token : completion_tokens) { mask.set(token); }
        return mask.count();
    }

    walk(trie, grammar, trie.root(), 0, mask);

    // A stop token reached mid-document would end the turn with a truncated value, so no
    // not-yet-complete mask may contain one even when the trie walk admitted its bytes.
    for (const int token : completion_tokens) { mask.unset(token); }

    if (mask.count() == 0) {
        // No token the vocabulary can spell continues this state. The mask is a device-side
        // bitset whose contract is that at least one bit is set — an empty mask is read as "no
        // candidate at all", not as "unconstrained" — so the fallback is the stop set: the turn
        // ends rather than the sampler inventing a token the constraint never announced.
        for (const int token : completion_tokens) { mask.set(token); }
    }
    return mask.count();
}

} // namespace ninfer::targets::qwen3_6::frontend_internal
