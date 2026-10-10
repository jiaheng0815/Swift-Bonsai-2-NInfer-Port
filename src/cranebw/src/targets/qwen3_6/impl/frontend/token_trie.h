#pragma once

// A byte trie over the vocabulary, built once per model and shared by every constrained request.
//
// Constrained decoding needs the inverse of the usual tokenizer question. The tokenizer answers
// "which token ids spell this text"; the mask builder asks "which tokens spell a text the grammar
// still admits", and it has to answer that after every emitted token. A trie is what makes that a
// single depth-first walk: the walk descends while the grammar accepts the byte on the edge, and
// every node it reaches that carries a token id contributes that token to the mask.
//
// The trie stores the *decoded* bytes of each token, which are what the model actually contributes
// to the output stream. Special tokens are excluded: they are control markers, not content, and a
// JSON document never contains them.

#include "targets/qwen3_6/impl/frontend/tokenizer.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6::frontend_internal {

class TokenTrie {
public:
    // One trie node. `token` is the id whose bytes end exactly here, or -1 for an interior node.
    // A node can be both an interior node and a terminal, which is how a token that is a prefix of a
    // longer one is represented.
    struct Node {
        std::uint32_t first_edge = 0;
        std::uint32_t edge_count = 0;
        std::int32_t token       = -1;
    };

    // Edge bytes within a node are sorted ascending, so the walk can binary-search for a byte
    // instead of scanning a fan-out that reaches 256 at the root.
    struct Edge {
        std::uint8_t byte   = 0;
        std::uint32_t child = 0;
    };

    // Builds the trie from every valid token of `tokenizer` below `token_domain`, excluding special
    // tokens. The domain is passed in rather than asked for: the tokenizer exposes predicates over
    // ids but no size, because the registered route profile owns that number.
    [[nodiscard]] static TokenTrie build(const Tokenizer& tokenizer, std::size_t token_domain);

    [[nodiscard]] std::uint32_t root() const noexcept { return 0; }

    [[nodiscard]] const Node& node(std::uint32_t index) const;

    [[nodiscard]] std::span<const Edge> edges(std::uint32_t index) const;

    // The child reached by `byte` from `index`, or kNoChild.
    [[nodiscard]] std::uint32_t child(std::uint32_t index, std::uint8_t byte) const noexcept;

    [[nodiscard]] std::size_t node_count() const noexcept { return nodes_.size(); }
    [[nodiscard]] std::size_t edge_count() const noexcept { return edges_.size(); }
    [[nodiscard]] std::size_t memory_bytes() const noexcept;

    static constexpr std::uint32_t kNoChild = static_cast<std::uint32_t>(-1);

private:
    TokenTrie() = default;

    std::vector<Node> nodes_;
    std::vector<Edge> edges_;
};

} // namespace ninfer::targets::qwen3_6::frontend_internal
