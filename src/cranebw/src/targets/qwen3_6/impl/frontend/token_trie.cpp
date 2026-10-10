// Vocabulary byte trie construction.

#include "targets/qwen3_6/impl/frontend/token_trie.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ninfer::targets::qwen3_6::frontend_internal {
namespace {

// A placeholder node id used while the trie is still a set of per-node child maps. The build
// resolves these into flat edge spans at the end.
constexpr std::uint32_t kUnassigned = static_cast<std::uint32_t>(-1);

struct BuildingNode {
    // Child byte -> node index. A map rather than a 256-entry array because the great majority of
    // nodes have a single child, and 256 pointers per node would dominate the memory.
    std::unordered_map<std::uint8_t, std::uint32_t> children;
    std::int32_t token = -1;
};

} // namespace

TokenTrie TokenTrie::build(const Tokenizer& tokenizer, std::size_t token_domain) {
    std::vector<BuildingNode> building(1);

    for (std::size_t id = 0; id < token_domain; ++id) {
        const int token = static_cast<int>(id);
        if (!tokenizer.is_valid_token(token)) { continue; }
        // Special tokens are control markers rather than content, so no document ever contains them.
        // A mask that admitted one would let the model terminate the turn inside a JSON value.
        if (tokenizer.is_special_token(token)) { continue; }
        const std::string_view bytes = tokenizer.decode_token_bytes(token, true);
        if (bytes.empty()) { continue; }

        std::uint32_t current = 0;
        for (const char raw : bytes) {
            const std::uint8_t byte = static_cast<std::uint8_t>(raw);
            const auto found        = building[current].children.find(byte);
            if (found != building[current].children.end()) {
                current = found->second;
                continue;
            }
            building.emplace_back();
            const auto child = static_cast<std::uint32_t>(building.size() - 1);
            building[current].children.emplace(byte, child);
            current = child;
        }
        // Two tokens can decode to the same bytes (byte-level BPE leaves some duplicates). Keeping
        // the lower id makes the choice deterministic and matches the sampler's tie-breaking.
        if (building[current].token < 0) { building[current].token = token; }
    }

    TokenTrie trie;
    trie.nodes_.resize(building.size());

    std::size_t total_edges = 0;
    for (const BuildingNode& node : building) { total_edges += node.children.size(); }
    trie.edges_.reserve(total_edges);

    for (std::size_t index = 0; index < building.size(); ++index) {
        Node& node    = trie.nodes_[index];
        node.token    = building[index].token;
        node.first_edge = static_cast<std::uint32_t>(trie.edges_.size());
        node.edge_count = static_cast<std::uint32_t>(building[index].children.size());

        std::vector<std::pair<std::uint8_t, std::uint32_t>> sorted(building[index].children.begin(),
                                                                   building[index].children.end());
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& left, const auto& right) { return left.first < right.first; });
        for (const auto& entry : sorted) {
            trie.edges_.push_back(Edge{.byte = entry.first, .child = entry.second});
        }
        if (trie.edges_.size() > static_cast<std::size_t>(kUnassigned)) {
            throw std::length_error("vocabulary trie exceeds the addressable edge range");
        }
    }
    return trie;
}

const TokenTrie::Node& TokenTrie::node(std::uint32_t index) const {
    if (index >= nodes_.size()) { throw std::out_of_range("trie node index is out of range"); }
    return nodes_[index];
}

std::span<const TokenTrie::Edge> TokenTrie::edges(std::uint32_t index) const {
    const Node& current = node(index);
    return std::span<const Edge>(edges_.data() + current.first_edge, current.edge_count);
}

std::uint32_t TokenTrie::child(std::uint32_t index, std::uint8_t byte) const noexcept {
    if (index >= nodes_.size()) { return kNoChild; }
    const Node& current = nodes_[index];
    std::size_t low     = current.first_edge;
    std::size_t high    = low + current.edge_count;
    while (low < high) {
        const std::size_t middle = low + (high - low) / 2;
        const Edge& edge         = edges_[middle];
        if (edge.byte == byte) { return edge.child; }
        if (edge.byte < byte) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    return kNoChild;
}

std::size_t TokenTrie::memory_bytes() const noexcept {
    return nodes_.size() * sizeof(Node) + edges_.size() * sizeof(Edge);
}

} // namespace ninfer::targets::qwen3_6::frontend_internal
