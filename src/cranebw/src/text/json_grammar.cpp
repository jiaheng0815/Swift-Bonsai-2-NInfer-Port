// The byte-level pushdown automaton behind schema-constrained decoding.
//
// Shape of the machine: a set of configurations, each a stack of frames over the JSON document.
// The set exists because `anyOf`/`oneOf`/`enum` make a prefix readable more than one way; the stack
// exists because JSON nests, and it is what lets a recursive `$ref` be handled by growing rather
// than by unfolding the schema. A frame is one *value slot*: the container frames sit below the
// value frames they enclose, so a member's own schema is the top frame while it is being read.
//
// String handling is the part worth reading twice. JSON Schema's string keywords address the
// *decoded* value: `maxLength` counts code points and `pattern` matches the unescaped text. A
// machine that matched the wire bytes would accept `"a"` for `{"maxLength":1}` but reject
// `"a"` for `{"const":"a"}`, and would run `pattern` against backslashes nobody wrote.

#include "text/json_grammar.h"

#include "text/regex_engine.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::text {
namespace {

constexpr std::size_t kNoIndex = static_cast<std::size_t>(-1);

[[nodiscard]] bool is_whitespace(std::uint8_t byte) noexcept {
    return byte == ' ' || byte == '\t' || byte == '\n' || byte == '\r';
}

[[nodiscard]] bool is_digit(std::uint8_t byte) noexcept { return byte >= '0' && byte <= '9'; }

enum class Phase : std::uint8_t {
    ValueStart,       // before the first byte of this slot's value
    InString,         // inside a string's raw bytes
    InNumber,         // inside a number
    InLiteral,        // inside true / false / null
    ObjectFirst,      // after '{': a key or '}'
    ObjectKey,        // inside a property name
    ObjectColon,      // after a property name: ':'
    ObjectValue,      // the member's value frame is on the stack
    ObjectAfterComma, // after ',': a key
    ObjectAfterValue, // after a member: ',' or '}'
    ArrayFirst,       // after '[': an element or ']'
    ArrayValue,       // the element's value frame is on the stack
    ArrayAfterComma,  // after ',': an element
    ArrayAfterValue,  // after an element: ',' or ']'
    Complete,         // this slot's value is finished
};

// A value slot. `nodes` is the conjunction the slot must satisfy: the schema node governing it plus
// every `allOf` member it reaches. An empty vector means unconstrained, which is how an
// `additionalProperties: true` member and an omitted `items` are represented.
struct Frame {
    std::vector<SchemaNodeId> nodes;
    Phase phase = Phase::ValueStart;

    std::string decoded;  // string value, decoded
    std::string escape;   // pending escape body without its backslash
    bool in_escape       = false;
    std::uint32_t high_surrogate = 0;

    std::string number;
    std::size_t literal_cursor = 0;
    char literal_first         = '\0';

    std::vector<std::uint8_t> seen;
    std::size_t members  = 0;
    std::string key;
    SchemaNodeId member_schema = kNoSchemaNode;

    std::size_t items = 0;

    // One cursor per pattern in the slot, in node order. The matchers themselves live in the
    // grammar's Impl so a cursor is cheap to copy with its frame.
    std::vector<std::size_t> pattern_slots;
    std::vector<RegexMatcher::Cursor> pattern_cursors;

    [[nodiscard]] bool operator==(const Frame& other) const noexcept {
        return nodes == other.nodes && phase == other.phase && decoded == other.decoded &&
               escape == other.escape && in_escape == other.in_escape &&
               high_surrogate == other.high_surrogate && number == other.number &&
               literal_cursor == other.literal_cursor && literal_first == other.literal_first &&
               seen == other.seen && members == other.members && key == other.key &&
               member_schema == other.member_schema && items == other.items &&
               pattern_slots == other.pattern_slots;
    }
};

struct Config {
    std::vector<Frame> stack;

    [[nodiscard]] bool operator==(const Config& other) const noexcept {
        if (stack.size() != other.stack.size()) { return false; }
        for (std::size_t index = 0; index < stack.size(); ++index) {
            if (!(stack[index] == other.stack[index])) { return false; }
        }
        return true;
    }
};

void flatten(const JsonSchema& schema, SchemaNodeId id, std::vector<SchemaNodeId>& out,
             std::vector<std::uint8_t>& guard, std::size_t depth) {
    if (depth > 64) { throw std::invalid_argument("schema allOf nesting is too deep"); }
    if (guard[id] != 0) { return; }
    guard[id] = 1;
    out.push_back(id);
    for (const SchemaNodeId member : schema.node(id).all_of) {
        flatten(schema, member, out, guard, depth + 1);
    }
}

[[nodiscard]] std::vector<SchemaNodeId> conjunction_of(const JsonSchema& schema, SchemaNodeId id) {
    std::vector<SchemaNodeId> out;
    std::vector<std::uint8_t> guard(schema.node_count(), 0);
    flatten(schema, id, out, guard, 0);
    return out;
}

// The single node in a conjunction carrying a structural role, or kNoSchemaNode. Two nodes both
// declaring `properties` would need a keyword-level merge this machine does not implement, so the
// constructor rejects that shape instead of guessing.
struct Structural {
    SchemaNodeId object = kNoSchemaNode;
    SchemaNodeId array  = kNoSchemaNode;
};

[[nodiscard]] Structural structural_of(const JsonSchema& schema,
                                       const std::vector<SchemaNodeId>& nodes) {
    Structural result;
    for (const SchemaNodeId id : nodes) {
        const SchemaNode& node = schema.node(id);
        if (!node.properties.empty() || node.additional != AdditionalProperties::Allowed ||
            node.min_properties.has_value() || node.max_properties.has_value()) {
            if (result.object != kNoSchemaNode) {
                throw std::invalid_argument(
                    "a schema combines two objects with property keywords in one slot; "
                    "constrained decoding cannot merge them");
            }
            result.object = id;
        }
        if (node.items != kNoSchemaNode || !node.prefix_items.empty() ||
            node.min_items.has_value() || node.max_items.has_value()) {
            if (result.array != kNoSchemaNode) {
                throw std::invalid_argument(
                    "a schema combines two arrays with item keywords in one slot; "
                    "constrained decoding cannot merge them");
            }
            result.array = id;
        }
    }
    return result;
}

[[nodiscard]] bool all_admit(const JsonSchema& schema, const std::vector<SchemaNodeId>& nodes,
                             JsonKind kind) {
    for (const SchemaNodeId id : nodes) {
        if (!schema.node(id).admits(kind)) { return false; }
    }
    return true;
}

[[nodiscard]] std::size_t tightest_min(const JsonSchema& schema,
                                       const std::vector<SchemaNodeId>& nodes,
                                       std::optional<std::size_t> SchemaNode::*member) {
    std::size_t result = 0;
    for (const SchemaNodeId id : nodes) {
        if (const auto& value = schema.node(id).*member) { result = std::max(result, *value); }
    }
    return result;
}

[[nodiscard]] std::optional<std::size_t> tightest_max(const JsonSchema& schema,
                                                      const std::vector<SchemaNodeId>& nodes,
                                                      std::optional<std::size_t> SchemaNode::*member) {
    std::optional<std::size_t> result;
    for (const SchemaNodeId id : nodes) {
        if (const auto& value = schema.node(id).*member) {
            result = result ? std::min(*result, *value) : value;
        }
    }
    return result;
}

void append_utf8(std::string& out, std::uint32_t codepoint) {
    if (codepoint < 0x80) {
        out.push_back(static_cast<char>(codepoint));
    } else if (codepoint < 0x800) {
        out.push_back(static_cast<char>(0xC0U | (codepoint >> 6U)));
        out.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else if (codepoint < 0x10000) {
        out.push_back(static_cast<char>(0xE0U | (codepoint >> 12U)));
        out.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else {
        out.push_back(static_cast<char>(0xF0U | (codepoint >> 18U)));
        out.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    }
}

[[nodiscard]] std::size_t count_codepoints(std::string_view utf8) {
    std::size_t count = 0;
    for (const char byte : utf8) {
        if ((static_cast<unsigned char>(byte) & 0xC0U) != 0x80U) { ++count; }
    }
    return count;
}

} // namespace

class JsonGrammar::Impl {
public:
    explicit Impl(const JsonSchema& schema) : schema_(schema) { initialize(); }

    [[nodiscard]] bool feed(std::uint8_t byte) {
        std::vector<Config> next;
        for (const Config& config : configs_) {
            for (Config& successor : step(config, byte)) {
                if (std::find(next.begin(), next.end(), successor) == next.end()) {
                    next.push_back(std::move(successor));
                }
            }
        }
        if (next.empty()) { return false; }
        configs_ = std::move(next);
        return true;
    }

    [[nodiscard]] ByteSet next_bytes() const {
        ByteSet announced;
        for (const Config& config : configs_) {
            if (config.stack.empty()) {
                // A completed document admits nothing at all; the request's stop tokens are the only
                // legal continuation, and the mask builder answers with those.
                continue;
            }
            allowed_from(config.stack.back(), announced);
        }
        // The frame on top does not always judge the byte. A number hands its delimiter up to the
        // container it settles into — that is how `"a":1,` works — so a comma the number frame
        // announces can still be refused by the object below it, and that object is the one that
        // knows whether another member fits. Every other phase answers for its own bytes, so only
        // the three delimiters of a number frame are re-asked of the machine itself. Probing all
        // 256 candidates instead would be exact too, but this runs once per vocabulary-trie node,
        // and the trie walk reaches hundreds of thousands of them.
        bool number_on_top = false;
        for (const Config& config : configs_) {
            if (!config.stack.empty() && config.stack.back().phase == Phase::InNumber) {
                number_on_top = true;
                break;
            }
        }
        if (!number_on_top) { return announced; }

        for (const std::uint8_t delimiter : {static_cast<std::uint8_t>(','),
                                             static_cast<std::uint8_t>(']'),
                                             static_cast<std::uint8_t>('}')}) {
            if (!announced.contains(delimiter)) { continue; }
            bool accepted = false;
            for (const Config& config : configs_) {
                if (config.stack.empty()) { continue; }
                if (!step(config, delimiter).empty()) {
                    accepted = true;
                    break;
                }
            }
            if (!accepted) { announced.remove(delimiter); }
        }
        return announced;
    }

    [[nodiscard]] bool complete() const {
        for (const Config& config : configs_) {
            if (config.stack.empty()) { return true; }
            // A bare scalar at the document root ends without a delimiter: `7` is a complete
            // document even though nothing follows it. Every other unfinished value needs a byte
            // (a closing quote or bracket) before it counts as done.
            if (config.stack.size() == 1 && config.stack.back().phase == Phase::InNumber &&
                number_acceptable(config.stack.back())) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] bool viable() const noexcept { return !configs_.empty(); }
    [[nodiscard]] std::size_t configuration_count() const noexcept { return configs_.size(); }

    // Rebuilding in place rather than assigning a fresh Impl: the class holds a reference to the
    // schema, so its copy assignment is deleted by the language.
    void reset() {
        configs_.clear();
        matchers_.clear();
        matcher_nodes_.clear();
        initialize();
    }

    // The machine as it stands once a string's opening quote has been consumed. That is the entry a
    // raw-text value needs: its bytes are the string's content and never carried a quote, so the
    // string keywords have to apply to content the machine was never told had opened.
    void initialize_inside_string() {
        validate_all(schema_.root(), 0);
        Config root;
        Frame frame;
        frame.nodes   = conjunction_of(schema_, schema_.root());
        frame.phase   = Phase::InString;
        frame.in_escape = false;
        attach_patterns(frame);
        root.stack.push_back(std::move(frame));
        configs_.push_back(std::move(root));
    }

    void initialize() {
        validate_all(schema_.root(), 0);
        Config root;
        Frame frame;
        frame.nodes = conjunction_of(schema_, schema_.root());
        attach_patterns(frame);
        root.stack.push_back(std::move(frame));
        configs_.push_back(std::move(root));
    }

    // Rejects, up front, the shapes the walk cannot express soundly. Discovering this at byte
    // thirty would surface as a mysterious dead end instead of a clear compile error.
    void validate_all(SchemaNodeId id, std::size_t depth) const {
        if (depth > 64) { return; }
        const SchemaNode& node = schema_.node(id);
        for (const SchemaNodeId member : node.all_of) { validate_all(member, depth + 1); }
        for (const SchemaNodeId member : node.any_of) { validate_all(member, depth + 1); }
        for (const SchemaNodeId member : node.one_of) { validate_all(member, depth + 1); }
        for (const SchemaNodeId value : node.exact_values) { validate_all(value, depth + 1); }
    }

    void attach_patterns(Frame& frame) const {
        frame.pattern_slots.clear();
        frame.pattern_cursors.clear();
        for (const SchemaNodeId id : frame.nodes) {
            if (schema_.node(id).pattern) {
                frame.pattern_slots.push_back(matcher_index(id));
                frame.pattern_cursors.push_back(matchers_[matcher_index(id)]->begin());
            }
        }
    }

    // Matchers are compiled once per (grammar, pattern) and shared by every config, because a
    // cursor is the mutable part and the program is not.
    [[nodiscard]] std::size_t matcher_index(SchemaNodeId id) const {
        for (std::size_t index = 0; index < matcher_nodes_.size(); ++index) {
            if (matcher_nodes_[index] == id) { return index; }
        }
        matcher_nodes_.push_back(id);
        // The compile result is a unique_ptr and the member is a shared_ptr<const>; the conversion is
        // implicit and leaves every existing cursor valid against the same immutable program.
        matchers_.push_back(RegexMatcher::compile(*schema_.node(id).pattern));
        return matchers_.size() - 1;
    }

    void allowed_from(const Frame& frame, ByteSet& set) const;
    // Whether this object frame can still take another member. A comma and a key both promise a
    // member, so when none is available both must vanish from the announcement *and* be refused by
    // `feed` — a comma the machine then cannot follow leaves the decoder in a state with no
    // continuation, and downstream the only way out of such a state is to end the turn, which
    // truncates the document.
    [[nodiscard]] bool object_can_add_member(const Frame& frame) const;
    // Whether this array frame can still take another element, which is what a comma offers.
    [[nodiscard]] bool array_can_add_element(const Frame& frame) const;
    // Whether `byte` continues a property name this object frame may still write. The name is only
    // resolved against the schema when its closing quote arrives, so without this test the key slot
    // accepts every byte: the walk then admits the whole vocabulary there and the constraint stops
    // constraining exactly where a model is most likely to invent a member name.
    [[nodiscard]] bool object_key_extends(const Frame& frame, std::uint8_t byte) const;
    // `depth` bounds the alternative-splitting recursion: a schema whose `anyOf` reaches itself
    // through a `$ref` would otherwise split without end.
    [[nodiscard]] std::vector<Config> step(const Config& config, std::uint8_t byte,
                                           std::size_t depth = 0) const;
    [[nodiscard]] std::vector<Config> step_frame(const Config& config, std::uint8_t byte) const;
    [[nodiscard]] bool open_value(Frame& frame, std::uint8_t byte) const;
    void settle(Config& config) const;
    void feed_decoded(Frame& frame, std::uint32_t codepoint) const;
    [[nodiscard]] bool string_closable(const Frame& frame) const;
    // Whether some code point in `[low, high]` can extend every pattern cursor in the slot. An escape
    // cannot be abandoned once its backslash is written, so a hex digit that leaves no reachable code
    // point the pattern accepts kills the value: the only exit would be closing the string at
    // something the pattern rejects. Ranges above 256 are answered with the matcher's "no code point
    // can continue" test, which is cheap; the narrower ranges that follow are enumerated exactly, so
    // a range that dies is always caught by the digit that kills it.
    [[nodiscard]] bool escape_range_feasible(const Frame& frame, std::uint32_t low,
                                             std::uint32_t high) const;
    // Whether the string's content can still become an admissible value. A byte that makes this
    // false must be refused outright rather than merely leaving the next-byte set empty: an empty
    // set keeps the configuration "alive" with nothing to emit, which strands the decoder instead of
    // letting it reject the token and recover.
    [[nodiscard]] bool string_extendable(const Frame& frame) const;
    [[nodiscard]] bool number_acceptable(const Frame& frame) const;
    // Whether some valid completion of the digits collected so far can satisfy every node in the slot.
    //
    // `feed` takes number bytes one at a time and only `number_acceptable` judges the finished value,
    // so without this test a byte no completion can rescue is still accepted — and the mask, built
    // from what `feed` accepts, then admits a token that strands the decoder at a number it can never
    // close. The vocabulary trie has no path out of that state, the mask comes back empty, and the
    // turn ends on a truncated value.
    [[nodiscard]] bool number_prefix_satisfiable(const Frame& frame) const;
    [[nodiscard]] std::vector<SchemaNodeId> member_nodes(const Frame& frame) const;
    [[nodiscard]] std::vector<SchemaNodeId> element_nodes(const Frame& frame) const;

    const JsonSchema& schema_;
    mutable std::vector<SchemaNodeId> matcher_nodes_;
    // Shared, not owned: a copy of the grammar shares the compiled programs and duplicates only the
    // live configuration set.
    mutable std::vector<std::shared_ptr<const RegexMatcher>> matchers_;
    std::vector<Config> configs_;
};

void JsonGrammar::Impl::feed_decoded(Frame& frame, std::uint32_t codepoint) const {
    append_utf8(frame.decoded, codepoint);
    for (std::size_t index = 0; index < frame.pattern_slots.size(); ++index) {
        const RegexMatcher& matcher = *matchers_[frame.pattern_slots[index]];
        (void)matcher.step(frame.pattern_cursors[index], codepoint);
    }
}

bool JsonGrammar::Impl::escape_range_feasible(const Frame& frame, std::uint32_t low,
                                             std::uint32_t high) const {
    if (frame.pattern_slots.empty()) { return true; }
    // A high half is not a code point on its own; the pair is judged once it completes. A low half
    // is different: with no high half pending it can never complete, so it is not skipped here —
    // skipping it would admit a range whose every member the machine then refuses.
    if (low >= 0xD800U && high <= 0xDBFFU) { return true; }
    if (low >= 0xDC00U && high <= 0xDFFFU && frame.high_surrogate == 0) { return false; }
    // The range is enumerated rather than sampled. Its widest case is the four thousand values left
    // by a single hex digit, and the alternative — the matcher's "no code point can continue" test —
    // answers a different question: it ignores the range entirely, so an astral pair whose every
    // member the pattern rejects would still read as feasible.
    for (std::size_t index = 0; index < frame.pattern_slots.size(); ++index) {
        const RegexMatcher& matcher        = *matchers_[frame.pattern_slots[index]];
        const RegexMatcher::Cursor& cursor = frame.pattern_cursors[index];
        // A cursor that has already matched stays satisfied however the value continues.
        if (RegexMatcher::matched(cursor)) { continue; }
        bool feasible = false;
        for (std::uint32_t codepoint = low; codepoint <= high; ++codepoint) {
            RegexMatcher::Cursor probe = cursor;
            if (matcher.step(probe, codepoint)) {
                feasible = true;
                break;
            }
        }
        if (!feasible) { return false; }
    }
    return true;
}

bool JsonGrammar::Impl::string_closable(const Frame& frame) const {
    const std::size_t length  = count_codepoints(frame.decoded);
    const std::size_t minimum = tightest_min(schema_, frame.nodes, &SchemaNode::min_length);
    if (length < minimum) { return false; }
    if (const auto maximum = tightest_max(schema_, frame.nodes, &SchemaNode::max_length)) {
        if (length > *maximum) { return false; }
    }
    for (const RegexMatcher::Cursor& cursor : frame.pattern_cursors) {
        // Unanchored matching means the quote may close once the pattern has matched somewhere;
        // nothing about the rest of the value can revoke that.
        if (!RegexMatcher::matched(cursor)) { return false; }
    }
    for (const SchemaNodeId id : frame.nodes) {
        const SchemaNode& node = schema_.node(id);
        if (node.exact_string && *node.exact_string != frame.decoded) { return false; }
    }
    return true;
}

bool JsonGrammar::Impl::string_extendable(const Frame& frame) const {
    for (const SchemaNodeId id : frame.nodes) {
        const SchemaNode& node = schema_.node(id);
        if (!node.exact_string) { continue; }
        // A pinned string only continues along its own bytes, and only up to its own length.
        if (frame.decoded.size() > node.exact_string->size()) { return false; }
        if (frame.decoded != node.exact_string->substr(0, frame.decoded.size())) { return false; }
    }
    if (const auto maximum = tightest_max(schema_, frame.nodes, &SchemaNode::max_length)) {
        if (count_codepoints(frame.decoded) > *maximum) { return false; }
    }
    for (std::size_t index = 0; index < frame.pattern_slots.size(); ++index) {
        if (matchers_[frame.pattern_slots[index]]->exhausted(frame.pattern_cursors[index])) {
            return false;
        }
    }
    return true;
}

bool JsonGrammar::Impl::number_prefix_satisfiable(const Frame& frame) const {
    const std::string& text = frame.number;
    if (text.empty()) { return true; }

    // An integer slot admits no fraction and no exponent, which is what makes `87.504…` impossible
    // rather than merely incomplete.
    const bool integer_only = all_admit(schema_, frame.nodes, JsonKind::Integer) &&
                              !all_admit(schema_, frame.nodes, JsonKind::Number);

    // Walk the JSON number grammar over the prefix, allowing the construct in progress to be
    // unfinished. Anything the cursor cannot consume means the bytes are not a number prefix at all.
    std::size_t index = 0;
    const bool negative = text[0] == '-';
    if (negative) { ++index; }
    const std::size_t integer_begin = index;
    while (index < text.size() && is_digit(text[index])) { ++index; }
    const std::size_t integer_end = index;
    if (integer_end == integer_begin) {
        // Only a sign so far. It survives if some node admits a negative value at all.
        if (index < text.size()) { return false; }
        for (const SchemaNodeId id : frame.nodes) {
            const SchemaNode& node = schema_.node(id);
            if (!node.admits(JsonKind::Integer) && !node.admits(JsonKind::Number)) { return false; }
            if (node.minimum && *node.minimum >= 0.0) { return false; }
            if (node.exclusive_minimum && *node.exclusive_minimum >= 0.0) { return false; }
        }
        return true;
    }
    if (index < text.size() && text[index] == '.') {
        if (integer_only) { return false; }
        ++index;
        while (index < text.size() && is_digit(text[index])) { ++index; }
    }
    if (index < text.size() && (text[index] == 'e' || text[index] == 'E')) {
        if (integer_only) { return false; }
        ++index;
        if (index < text.size() && (text[index] == '+' || text[index] == '-')) { ++index; }
        while (index < text.size() && is_digit(text[index])) { ++index; }
    }
    if (index != text.size()) { return false; }

    // Magnitude. Appending digits to the integer part only pushes the value further from zero and the
    // sign cannot change, so an integer part already outside a node's range can never come back.
    double magnitude = 0.0;
    try {
        magnitude = std::stod(text.substr(integer_begin, integer_end - integer_begin));
    } catch (const std::exception&) {
        return false;
    }
    for (const SchemaNodeId id : frame.nodes) {
        const SchemaNode& node = schema_.node(id);
        if (!negative) {
            if (node.maximum && magnitude > *node.maximum) { return false; }
            if (node.exclusive_maximum && magnitude >= *node.exclusive_maximum) { return false; }
        } else {
            if (node.minimum && -magnitude < *node.minimum) { return false; }
            if (node.exclusive_minimum && -magnitude <= *node.exclusive_minimum) { return false; }
        }
    }
    return true;
}

bool JsonGrammar::Impl::number_acceptable(const Frame& frame) const {
    if (frame.number.empty()) { return false; }
    const char last = frame.number.back();
    if (last == '-' || last == '.' || last == 'e' || last == 'E' || last == '+') { return false; }
    double value = 0.0;
    try {
        value = std::stod(frame.number);
    } catch (const std::exception&) {
        return false;
    }
    for (const SchemaNodeId id : frame.nodes) {
        const SchemaNode& node = schema_.node(id);
        if (node.minimum && value < *node.minimum) { return false; }
        if (node.maximum && value > *node.maximum) { return false; }
        if (node.exclusive_minimum && value <= *node.exclusive_minimum) { return false; }
        if (node.exclusive_maximum && value >= *node.exclusive_maximum) { return false; }
        if (node.multiple_of) {
            const double quotient = value / *node.multiple_of;
            if (std::abs(quotient - std::round(quotient)) > 1e-9) { return false; }
        }
        if (node.exact_boolean) {
            // A boolean literal is matched through the literal path, never through a number.
            return false;
        }
    }
    // An integer-only slot admits no fractional part and no exponent.
    if (all_admit(schema_, frame.nodes, JsonKind::Integer) &&
        !all_admit(schema_, frame.nodes, JsonKind::Number)) {
        if (frame.number.find_first_of(".eE") != std::string::npos) { return false; }
    }
    return true;
}

std::vector<SchemaNodeId> JsonGrammar::Impl::member_nodes(const Frame& frame) const {
    const Structural structural = structural_of(schema_, frame.nodes);
    if (structural.object == kNoSchemaNode) { return {}; }
    const SchemaNode& object = schema_.node(structural.object);
    if (const SchemaNodeId declared = object.property(frame.key); declared != kNoSchemaNode) {
        return conjunction_of(schema_, declared);
    }
    if (object.additional == AdditionalProperties::Denied) { return {}; }
    if (object.additional == AdditionalProperties::Schema) {
        return conjunction_of(schema_, object.additional_schema);
    }
    return {};
}

std::vector<SchemaNodeId> JsonGrammar::Impl::element_nodes(const Frame& frame) const {
    const Structural structural = structural_of(schema_, frame.nodes);
    if (structural.array == kNoSchemaNode) { return {}; }
    const SchemaNode& array = schema_.node(structural.array);
    if (frame.items < array.prefix_items.size()) {
        return conjunction_of(schema_, array.prefix_items[frame.items]);
    }
    if (array.items != kNoSchemaNode) { return conjunction_of(schema_, array.items); }
    return {};
}

bool JsonGrammar::Impl::object_can_add_member(const Frame& frame) const {
    const Structural layout = structural_of(schema_, frame.nodes);
    if (layout.object == kNoSchemaNode) {
        // An object with no declared shape takes any member, so there is always somewhere to go.
        return true;
    }
    const SchemaNode& object = schema_.node(layout.object);
    if (object.max_properties && frame.members >= *object.max_properties) { return false; }
    if (object.additional != AdditionalProperties::Denied) { return true; }
    for (std::size_t index = 0; index < object.properties.size(); ++index) {
        // An index outside `seen` is a member this frame never recorded, which can only happen for a
        // shape the machine did not open itself; treating it as unwritten keeps the answer permissive
        // rather than inventing a restriction.
        if (index >= frame.seen.size() || frame.seen[index] == 0) { return true; }
    }
    return false;
}

bool JsonGrammar::Impl::object_key_extends(const Frame& frame, std::uint8_t byte) const {
    const Structural layout = structural_of(schema_, frame.nodes);
    if (layout.object == kNoSchemaNode) { return true; }
    const SchemaNode& object = schema_.node(layout.object);
    if (object.additional != AdditionalProperties::Denied) { return true; }
    for (std::size_t index = 0; index < object.properties.size(); ++index) {
        // A member already written is not offered again, so its name cannot be spelled a second time.
        if (index < frame.seen.size() && frame.seen[index] != 0) { continue; }
        const std::string& name = object.properties[index].first;
        if (name.size() <= frame.key.size()) { continue; }
        if (name.compare(0, frame.key.size(), frame.key) != 0) { continue; }
        if (static_cast<std::uint8_t>(name[frame.key.size()]) == byte) { return true; }
    }
    return false;
}

bool JsonGrammar::Impl::array_can_add_element(const Frame& frame) const {
    const Structural layout = structural_of(schema_, frame.nodes);
    if (layout.array == kNoSchemaNode) { return true; }
    const SchemaNode& array = schema_.node(layout.array);
    return !(array.max_items && frame.items >= *array.max_items);
}

void JsonGrammar::Impl::allowed_from(const Frame& frame, ByteSet& set) const {
    const auto& nodes = frame.nodes;
    switch (frame.phase) {
    case Phase::ValueStart: {
        // A value slot with `enum`/`const` admits only those literals; branch on the first byte.
        bool has_exact = false;
        for (const SchemaNodeId id : nodes) {
            if (!schema_.node(id).exact_values.empty()) { has_exact = true; }
        }
        if (has_exact) {
            for (const SchemaNodeId id : nodes) {
                for (const SchemaNodeId value : schema_.node(id).exact_values) {
                    const SchemaNode& literal = schema_.node(value);
                    for (const JsonKind kind : {JsonKind::Object, JsonKind::Array, JsonKind::String,
                                                JsonKind::Boolean, JsonKind::Null, JsonKind::Number,
                                                JsonKind::Integer}) {
                        if (!literal.admits(kind)) { continue; }
                        if (kind == JsonKind::Object) { set.insert('{'); }
                        if (kind == JsonKind::Array) { set.insert('['); }
                        if (kind == JsonKind::String) { set.insert('"'); }
                        if (kind == JsonKind::Boolean) {
                            set.insert('t');
                            set.insert('f');
                        }
                        if (kind == JsonKind::Null) { set.insert('n'); }
                        if (kind == JsonKind::Number || kind == JsonKind::Integer) {
                            set.insert('-');
                            set.insert_range('0', '9');
                        }
                    }
                }
            }
            return;
        }
        if (all_admit(schema_, nodes, JsonKind::Object)) { set.insert('{'); }
        if (all_admit(schema_, nodes, JsonKind::Array)) { set.insert('['); }
        if (all_admit(schema_, nodes, JsonKind::String)) { set.insert('"'); }
        // A boolean literal pins the value, so only its own letter is admissible.
        bool boolean_pinned = false;
        for (const SchemaNodeId id : nodes) {
            if (schema_.node(id).exact_boolean) { boolean_pinned = true; }
        }
        if (all_admit(schema_, nodes, JsonKind::Boolean)) {
            if (!boolean_pinned) {
                set.insert('t');
                set.insert('f');
            } else {
                for (const SchemaNodeId id : nodes) {
                    if (schema_.node(id).exact_boolean) {
                        set.insert(*schema_.node(id).exact_boolean ? 't' : 'f');
                    }
                }
            }
        }
        if (all_admit(schema_, nodes, JsonKind::Null)) { set.insert('n'); }
        if (all_admit(schema_, nodes, JsonKind::Number) ||
            all_admit(schema_, nodes, JsonKind::Integer)) {
            set.insert('-');
            set.insert_range('0', '9');
        }
        return;
    }
    case Phase::InString: {
        // A pinning string admits only the bytes that continue its pinned content. Answering "any
        // byte" here is what makes a mask useless for `enum`/`const`: the trie walk then reaches the
        // whole vocabulary, every token is admitted, and the model wanders without the grammar ever
        // dying — a constraint that reports itself as active while constraining nothing.
        for (const SchemaNodeId id : nodes) {
            const SchemaNode& node = schema_.node(id);
            if (!node.exact_string) { continue; }
            if (frame.decoded.size() < node.exact_string->size()) {
                set.insert(static_cast<std::uint8_t>((*node.exact_string)[frame.decoded.size()]));
            }
            if (frame.decoded == *node.exact_string) { set.insert('"'); }
            return;
        }
        // A length-capped string admits anything below the cap and only its close at the cap, so the
        // decoder is forced to terminate the value instead of running past it forever.
        if (const auto maximum = tightest_max(schema_, nodes, &SchemaNode::max_length)) {
            if (count_codepoints(frame.decoded) >= *maximum) {
                if (string_closable(frame)) { set.insert('"'); }
                return;
            }
        }
        set.insert_all();
        if (string_closable(frame)) { set.insert('"'); }
        return;
    }
    case Phase::InNumber:
        set.insert_range('0', '9');
        set.insert('.');
        set.insert('e');
        set.insert('E');
        set.insert('+');
        set.insert('-');
        set.insert(',');
        set.insert(']');
        set.insert('}');
        return;
    case Phase::InLiteral: {
        // A literal in progress admits exactly its next character; announcing nothing would strand
        // the decoder mid-`true`.
        const std::string_view word = frame.literal_first == 't'
                                          ? std::string_view("true")
                                          : (frame.literal_first == 'f' ? std::string_view("false")
                                                                       : std::string_view("null"));
        if (frame.literal_cursor < word.size()) {
            set.insert(static_cast<std::uint8_t>(word[frame.literal_cursor]));
        }
        return;
    }
    case Phase::ObjectFirst:
    case Phase::ObjectAfterComma: {
        if (object_can_add_member(frame)) { set.insert('"'); }
        const Structural structural = structural_of(schema_, nodes);
        if (structural.object != kNoSchemaNode) {
            const SchemaNode& object = schema_.node(structural.object);
            bool closable            = object.required.empty() &&
                            frame.members >= object.min_properties.value_or(0);
            if (frame.phase == Phase::ObjectAfterComma) { closable = false; }
            if (closable) { set.insert('}'); }
        } else if (frame.phase == Phase::ObjectFirst) {
            set.insert('}');
        }
        return;
    }
    case Phase::ObjectKey: {
        // A key is content too. Answering "any byte" lets the model invent member names, which is
        // exactly what an object schema exists to prevent — the same hole that string content had.
        const Structural key_layout = structural_of(schema_, nodes);
        if (key_layout.object == kNoSchemaNode) {
            set.insert_all();
            return;
        }
        const SchemaNode& object = schema_.node(key_layout.object);
        if (object.additional != AdditionalProperties::Denied) {
            set.insert_all();
            return;
        }
        for (std::size_t index = 0; index < object.properties.size(); ++index) {
            // A member already written is not offered again: `additionalProperties: false` describes
            // a set of members, and emitting one twice would produce a document whose meaning
            // depends on which duplicate the reader keeps.
            if (index < frame.seen.size() && frame.seen[index] != 0) { continue; }
            const std::string& name = object.properties[index].first;
            if (name == frame.key) { set.insert('"'); }
            if (name.size() > frame.key.size() &&
                name.compare(0, frame.key.size(), frame.key) == 0) {
                set.insert(static_cast<std::uint8_t>(name[frame.key.size()]));
            }
        }
        return;
    }
    case Phase::ObjectColon:
        set.insert(':');
        return;
    case Phase::ObjectValue:
    case Phase::ArrayValue: return; // the value frame on top answers for this byte
    case Phase::ObjectAfterValue: {
        const Structural object_layout = structural_of(schema_, nodes);
        bool closable                  = true;
        bool extendable                = true;
        if (object_layout.object != kNoSchemaNode) {
            const SchemaNode& object = schema_.node(object_layout.object);
            for (std::size_t index = 0; index < object.properties.size(); ++index) {
                if (object.is_required(object.properties[index].first) && index < frame.seen.size() &&
                    frame.seen[index] == 0) {
                    closable = false;
                }
            }
            if (frame.members < object.min_properties.value_or(0)) { closable = false; }
        }
        // A comma promises another member, so it is announced only when one can actually follow.
        if (extendable && object_can_add_member(frame)) { set.insert(','); }
        if (closable) { set.insert('}'); }
        return;
    }
    case Phase::ArrayFirst:
    case Phase::ArrayAfterComma: {
        const Structural structural = structural_of(schema_, nodes);
        if (structural.array != kNoSchemaNode) {
            const SchemaNode& array = schema_.node(structural.array);
            if (frame.phase == Phase::ArrayFirst && frame.items >= array.min_items.value_or(0)) {
                set.insert(']');
            }
        } else if (frame.phase == Phase::ArrayFirst) {
            set.insert(']');
        }
        // The element slot's own schema answers for the value bytes, not the array's.
        ByteSet inner;
        Frame probe = frame;
        probe.nodes = element_nodes(frame);
        probe.phase = Phase::ValueStart;
        allowed_from(probe, inner);
        set.union_with(inner);
        return;
    }
    case Phase::ArrayAfterValue: {
        if (array_can_add_element(frame)) { set.insert(','); }
        const Structural structural = structural_of(schema_, nodes);
        bool closable              = true;
        if (structural.array != kNoSchemaNode) {
            const SchemaNode& array = schema_.node(structural.array);
            if (frame.items < array.min_items.value_or(0)) { closable = false; }
            if (array.max_items && frame.items > *array.max_items) { closable = false; }
        }
        if (closable) { set.insert(']'); }
        return;
    }
    case Phase::Complete: return;
    }
}

bool JsonGrammar::Impl::open_value(Frame& frame, std::uint8_t byte) const {
    const auto& nodes = frame.nodes;

    // A pinned boolean literal admits exactly one spelling.
    for (const SchemaNodeId id : nodes) {
        if (const auto pinned = schema_.node(id).exact_boolean) {
            const char wanted = *pinned ? 't' : 'f';
            if (byte != static_cast<std::uint8_t>(wanted)) { return false; }
        }
    }

    switch (byte) {
    case '{':
        if (!all_admit(schema_, nodes, JsonKind::Object)) { return false; }
        frame.phase = Phase::ObjectFirst;
        {
            const Structural structural = structural_of(schema_, nodes);
            const std::size_t declared  = structural.object == kNoSchemaNode
                                              ? 0
                                              : schema_.node(structural.object).properties.size();
            frame.seen.assign(declared, 0);
        }
        frame.members = 0;
        return true;
    case '[':
        if (!all_admit(schema_, nodes, JsonKind::Array)) { return false; }
        frame.phase = Phase::ArrayFirst;
        frame.items = 0;
        return true;
    case '"':
        if (!all_admit(schema_, nodes, JsonKind::String)) { return false; }
        frame.phase = Phase::InString;
        frame.decoded.clear();
        frame.escape.clear();
        frame.in_escape = false;
        attach_patterns(frame);
        return true;
    case 't':
    case 'f':
        if (!all_admit(schema_, nodes, JsonKind::Boolean)) { return false; }
        frame.phase          = Phase::InLiteral;
        frame.literal_first  = static_cast<char>(byte);
        frame.literal_cursor = 1;
        for (const SchemaNodeId id : nodes) {
            if (schema_.node(id).exact_boolean &&
                *schema_.node(id).exact_boolean != (byte == 't')) {
                return false;
            }
        }
        return true;
    case 'n':
        if (!all_admit(schema_, nodes, JsonKind::Null)) { return false; }
        frame.phase          = Phase::InLiteral;
        frame.literal_first  = 'n';
        frame.literal_cursor = 1;
        return true;
    default: break;
    }
    if (byte == '-' || is_digit(byte)) {
        if (!all_admit(schema_, nodes, JsonKind::Number) &&
            !all_admit(schema_, nodes, JsonKind::Integer)) {
            return false;
        }
        frame.phase = Phase::InNumber;
        frame.number.assign(1, static_cast<char>(byte));
        // The first byte of a number opens the frame rather than extending it, so the same gate the
        // `InNumber` step applies has to be applied here too: a leading `-` for a slot whose floor is
        // zero is exactly the byte no completion can rescue.
        if (!number_prefix_satisfiable(frame)) {
            frame.number.clear();
            frame.phase = Phase::ValueStart;
            return false;
        }
        return true;
    }
    return false;
}

void JsonGrammar::Impl::settle(Config& config) const {
    while (!config.stack.empty() && config.stack.back().phase == Phase::Complete) {
        config.stack.pop_back();
        if (config.stack.empty()) { return; }
        Frame& parent = config.stack.back();
        switch (parent.phase) {
        case Phase::ObjectValue:
            ++parent.members;
            parent.phase = Phase::ObjectAfterValue;
            break;
        case Phase::ArrayValue:
            ++parent.items;
            parent.phase = Phase::ArrayAfterValue;
            break;
        default: break;
        }
    }
}

std::vector<Config> JsonGrammar::Impl::step(const Config& config, std::uint8_t byte,
                                            std::size_t depth) const {
    std::vector<Config> out;
    if (config.stack.empty()) { return out; }
    Config working = config;
    Frame& frame   = working.stack.back();

    // A slot that can be read more than one way — `enum`/`const` alternatives, or the branches of
    // `anyOf`/`oneOf` — is split here. Each alternative becomes its own configuration, so a byte
    // that only one reading admits survives under that reading instead of being refused outright.
    if (frame.phase == Phase::ValueStart && !is_whitespace(byte) && depth < 8) {
        std::vector<std::vector<SchemaNodeId>> alternatives;
        std::vector<SchemaNodeId> base;
        for (const SchemaNodeId id : frame.nodes) {
            const SchemaNode& node = schema_.node(id);
            if (!node.exact_values.empty()) {
                for (const SchemaNodeId value : node.exact_values) {
                    alternatives.push_back(conjunction_of(schema_, value));
                }
                continue;
            }
            if (!node.any_of.empty() || !node.one_of.empty()) {
                for (const SchemaNodeId member : node.any_of) {
                    alternatives.push_back(conjunction_of(schema_, member));
                }
                for (const SchemaNodeId member : node.one_of) {
                    alternatives.push_back(conjunction_of(schema_, member));
                }
                // `allOf` already contributed through conjunction_of, so only the node's own scalar
                // keywords would be lost by dropping it. Keep it when it constrains anything.
                if (node.kinds.empty() && !node.min_length && !node.max_length && !node.pattern &&
                    !node.minimum && !node.maximum && !node.exact_string && !node.exact_boolean) {
                    continue;
                }
            }
            base.push_back(id);
        }
        if (!alternatives.empty()) {
            std::vector<Config> branched;
            for (const std::vector<SchemaNodeId>& alternative : alternatives) {
                Config branch = config;
                Frame& slot   = branch.stack.back();
                slot.nodes    = base;
                slot.nodes.insert(slot.nodes.end(), alternative.begin(), alternative.end());
                slot.phase = Phase::ValueStart;
                for (Config& successor : step(branch, byte, depth + 1)) {
                    branched.push_back(std::move(successor));
                }
            }
            return branched;
        }
    }
    return step_frame(config, byte);
}

std::vector<Config> JsonGrammar::Impl::step_frame(const Config& config, std::uint8_t byte) const {
    std::vector<Config> out;
    Config working = config;
    Frame& frame   = working.stack.back();

    // JSON permits whitespace between any two tokens, so every structural position absorbs it
    // without changing phase. The positions where whitespace is *content* — inside a string, and
    // between the digits of a number — are excluded.
    // Whitespace is deliberately outside the constrained language. JSON permits it between tokens,
    // but a decoder has no use for it: a position that admits whitespace admits it repeatedly, so a
    // model whose preferred tokens are all masked out can emit newlines until its budget runs out
    // and the grammar never objects. Omitting it still yields a valid document — the whitespace-free
    // subset of the schema's language — and it makes every position force the next real token.
    const bool structural_position = frame.phase == Phase::ObjectFirst ||
                                     frame.phase == Phase::ObjectColon ||
                                     frame.phase == Phase::ObjectAfterComma ||
                                     frame.phase == Phase::ObjectAfterValue ||
                                     frame.phase == Phase::ArrayFirst ||
                                     frame.phase == Phase::ArrayAfterComma ||
                                     frame.phase == Phase::ArrayAfterValue;
    if (structural_position && is_whitespace(byte)) { return out; }

    switch (frame.phase) {
    case Phase::ValueStart:
        if (open_value(frame, byte)) {
            settle(working);
            out.push_back(std::move(working));
        }
        return out;

    case Phase::InString: {
        if (frame.in_escape) {
            frame.escape.push_back(static_cast<char>(byte));
            if (frame.escape[0] == 'u') {
                const std::size_t entered = frame.escape.size() - 1;
                if (entered >= 1 && entered <= 3) {
                    std::uint32_t partial = 0;
                    for (std::size_t index = 1; index < frame.escape.size(); ++index) {
                        const char digit = frame.escape[index];
                        if (digit >= '0' && digit <= '9') {
                            partial = partial * 16U + static_cast<std::uint32_t>(digit - '0');
                        } else if (digit >= 'a' && digit <= 'f') {
                            partial = partial * 16U + static_cast<std::uint32_t>(digit - 'a') + 10U;
                        } else if (digit >= 'A' && digit <= 'F') {
                            partial = partial * 16U + static_cast<std::uint32_t>(digit - 'A') + 10U;
                        } else {
                            return out;
                        }
                    }
                    const std::uint32_t shift = static_cast<std::uint32_t>(4 * (4 - entered));
                    const std::uint32_t low   = partial << shift;
                    const std::uint32_t high  = low + ((1U << shift) - 1U);
                    if (frame.high_surrogate != 0) {
                        // These digits are the low half of a pair, so the code point the escape will
                        // denote is the astral value the pair composes, not the digits themselves.
                        // Testing the digits would let the model open a pair the pattern can never
                        // accept, and every completion of it would then be refused.
                        if (low > 0xDFFFU || high < 0xDC00U) { return out; }
                        const std::uint32_t base =
                            0x10000U + ((frame.high_surrogate - 0xD800U) << 10U);
                        const std::uint32_t first = base + (low < 0xDC00U ? 0U : low - 0xDC00U);
                        const std::uint32_t last  = base + (high - 0xDC00U);
                        if (!escape_range_feasible(frame, first, last)) { return out; }
                    } else if (!escape_range_feasible(frame, low, high)) {
                        return out;
                    }
                }
            }
            const std::size_t wanted = frame.escape[0] == 'u' ? 5 : 1;
            if (frame.escape.size() < wanted) {
                out.push_back(std::move(working));
                return out;
            }
            std::uint32_t codepoint = 0;
            if (frame.escape[0] == 'u') {
                for (std::size_t index = 1; index < 5; ++index) {
                    const char digit = frame.escape[index];
                    std::uint32_t nibble = 0;
                    if (digit >= '0' && digit <= '9') {
                        nibble = static_cast<std::uint32_t>(digit - '0');
                    } else if (digit >= 'a' && digit <= 'f') {
                        nibble = static_cast<std::uint32_t>(digit - 'a') + 10U;
                    } else if (digit >= 'A' && digit <= 'F') {
                        nibble = static_cast<std::uint32_t>(digit - 'A') + 10U;
                    } else {
                        return out;
                    }
                    codepoint = codepoint * 16U + nibble;
                }
                if (codepoint >= 0xD800 && codepoint <= 0xDBFF) {
                    // A high half must be closed by its low half. Two of them in a row spell a value
                    // no reader can decode, and — worse for a constrained decoder — a run of them
                    // never produces a code point at all, so the pattern is never consulted and the
                    // machine goes on accepting them for as long as the budget lasts.
                    if (frame.high_surrogate != 0) { return out; }
                    frame.high_surrogate = codepoint;
                    frame.escape.clear();
                    frame.in_escape = false;
                    out.push_back(std::move(working));
                    return out;
                }
                if (codepoint >= 0xDC00 && codepoint <= 0xDFFF) {
                    if (frame.high_surrogate == 0) { return out; }
                    codepoint = 0x10000U + ((frame.high_surrogate - 0xD800U) << 10U) +
                                (codepoint - 0xDC00U);
                    frame.high_surrogate = 0;
                } else if (frame.high_surrogate != 0) {
                    // A pending high half admits nothing but a low half.
                    return out;
                }
            } else {
                switch (frame.escape[0]) {
                case 'n': codepoint = '\n'; break;
                case 'r': codepoint = '\r'; break;
                case 't': codepoint = '\t'; break;
                case 'b': codepoint = '\b'; break;
                case 'f': codepoint = '\f'; break;
                case '"': codepoint = '"'; break;
                case '\\': codepoint = '\\'; break;
                case '/': codepoint = '/'; break;
                default: return out;
                }
            }
            frame.escape.clear();
            frame.in_escape = false;
            feed_decoded(frame, codepoint);
            if (!string_extendable(frame)) { return out; }
            out.push_back(std::move(working));
            return out;
        }
        if (byte == '"') {
            if (!string_closable(frame)) { return out; }
            frame.phase = Phase::Complete;
            settle(working);
            out.push_back(std::move(working));
            return out;
        }
        if (byte == '\\') {
            frame.escape.clear();
            frame.in_escape = true;
            out.push_back(std::move(working));
            return out;
        }
        if (byte < 0x20) { return out; }
        // Raw UTF-8 bytes are appended verbatim, never re-encoded: they are already the value's
        // bytes. Re-encoding each one as a code point would double-encode every non-ASCII character
        // and report two code points where the value has one.
        frame.decoded.push_back(static_cast<char>(byte));
        for (std::size_t index = 0; index < frame.pattern_slots.size(); ++index) {
            matchers_[frame.pattern_slots[index]]->step(frame.pattern_cursors[index], byte);
        }
        if (!string_extendable(frame)) { return out; }
        out.push_back(std::move(working));
        return out;
    }

    case Phase::InNumber:
        if (is_digit(byte) || byte == '.' || byte == 'e' || byte == 'E' || byte == '+' ||
            byte == '-') {
            frame.number.push_back(static_cast<char>(byte));
            if (!number_prefix_satisfiable(frame)) {
                // The byte is refused rather than accumulated and rejected at the delimiter: the mask
                // is built from what this function accepts, so a permissive answer here is a
                // permissive mask, and a mask that lets the model into an unclosable number is a mask
                // that ends the turn on a truncated document.
                frame.number.pop_back();
                return out;
            }
            out.push_back(std::move(working));
            return out;
        }
        if (!number_acceptable(frame)) { return out; }
        frame.phase = Phase::Complete;
        settle(working);
        // The delimiter belongs to the enclosing container, so it is re-dispatched after the pop.
        for (Config& successor : step(working, byte)) {
            out.push_back(std::move(successor));
        }
        return out;

    case Phase::InLiteral: {
        const std::string_view word = frame.literal_first == 't'
                                          ? std::string_view("true")
                                          : (frame.literal_first == 'f' ? std::string_view("false")
                                                                       : std::string_view("null"));
        if (frame.literal_cursor >= word.size() ||
            static_cast<char>(byte) != word[frame.literal_cursor]) {
            return out;
        }
        ++frame.literal_cursor;
        if (frame.literal_cursor == word.size()) { frame.phase = Phase::Complete; }
        settle(working);
        out.push_back(std::move(working));
        return out;
    }

    case Phase::ObjectFirst:
    case Phase::ObjectAfterComma:
        if (byte == '"') {
            // A quote opens a property name. When no name can be written the byte is refused here,
            // because the key frame it would open has no admissible content at all.
            if (!object_can_add_member(frame)) { return out; }
            frame.phase = Phase::ObjectKey;
            frame.key.clear();
            frame.member_schema = kNoSchemaNode;
            out.push_back(std::move(working));
            return out;
        }
        if (byte == '}' && frame.phase == Phase::ObjectFirst) {
            const Structural structural = structural_of(schema_, frame.nodes);
            if (structural.object != kNoSchemaNode) {
                const SchemaNode& object = schema_.node(structural.object);
                if (!object.required.empty() ||
                    frame.members < object.min_properties.value_or(0)) {
                    return out;
                }
            }
            frame.phase = Phase::Complete;
            settle(working);
            out.push_back(std::move(working));
        }
        return out;

    case Phase::ObjectKey:
        if (byte == '"') {
            // The name is complete: resolve it to the member's schema now, so the value frame that
            // follows is built from the right node.
            const Structural structural = structural_of(schema_, frame.nodes);
            if (structural.object != kNoSchemaNode) {
                const SchemaNode& object = schema_.node(structural.object);
                const SchemaNodeId declared = object.property(frame.key);
                if (declared != kNoSchemaNode) {
                    for (std::size_t index = 0; index < object.properties.size(); ++index) {
                        if (object.properties[index].first != frame.key) { continue; }
                        if (index < frame.seen.size() && frame.seen[index] != 0) {
                            // The member is already written, so this is a duplicate.
                            return out;
                        }
                        if (index < frame.seen.size()) { frame.seen[index] = 1; }
                    }
                } else if (object.additional == AdditionalProperties::Denied) {
                    return out;
                }
            }
            frame.phase = Phase::ObjectColon;
            out.push_back(std::move(working));
            return out;
        }
        // A byte that no writable member name continues would leave the key frame unable to close,
        // so it is refused here rather than accumulated and rejected later: the mask is built from
        // what this function accepts, and a permissive answer here is a permissive mask.
        if (!object_key_extends(frame, byte)) { return out; }
        frame.key.push_back(static_cast<char>(byte));
        out.push_back(std::move(working));
        return out;

    case Phase::ObjectColon:
        if (byte != ':') { return out; }
        frame.phase = Phase::ObjectValue;
        {
            Frame child;
            child.nodes = member_nodes(frame);
            child.phase = Phase::ValueStart;
            attach_patterns(child);
            working.stack.push_back(std::move(child));
        }
        out.push_back(std::move(working));
        return out;

    case Phase::ObjectAfterValue:
        if (byte == ',') {
            // A comma opens a member slot. When no member can be written the byte is refused here
            // rather than accepted and then stranded: the decoder's next position would have no
            // legal byte at all.
            if (!object_can_add_member(frame)) { return out; }
            frame.phase = Phase::ObjectAfterComma;
            out.push_back(std::move(working));
            return out;
        }
        if (byte == '}') {
            const Structural structural = structural_of(schema_, frame.nodes);
            if (structural.object != kNoSchemaNode) {
                const SchemaNode& object = schema_.node(structural.object);
                for (std::size_t index = 0; index < object.properties.size(); ++index) {
                    if (object.is_required(object.properties[index].first) &&
                        index < frame.seen.size() && frame.seen[index] == 0) {
                        return out;
                    }
                }
                if (frame.members < object.min_properties.value_or(0)) { return out; }
                if (object.max_properties && frame.members > *object.max_properties) { return out; }
            }
            frame.phase = Phase::Complete;
            settle(working);
            out.push_back(std::move(working));
        }
        return out;

    case Phase::ArrayFirst:
    case Phase::ArrayAfterComma:
        if (byte == ']' && frame.phase == Phase::ArrayFirst) {
            const Structural structural = structural_of(schema_, frame.nodes);
            if (structural.array != kNoSchemaNode) {
                const SchemaNode& array = schema_.node(structural.array);
                if (frame.items < array.min_items.value_or(0)) { return out; }
            }
            frame.phase = Phase::Complete;
            settle(working);
            out.push_back(std::move(working));
            return out;
        }
        frame.phase = Phase::ArrayValue;
        {
            Frame child;
            child.nodes = element_nodes(frame);
            child.phase = Phase::ValueStart;
            attach_patterns(child);
            working.stack.push_back(std::move(child));
        }
        for (Config& successor : step(working, byte)) {
            out.push_back(std::move(successor));
        }
        return out;

    case Phase::ArrayAfterValue:
        if (byte == ',') {
            // A comma opens an element slot, which a full array cannot offer.
            if (!array_can_add_element(frame)) { return out; }
            frame.phase = Phase::ArrayAfterComma;
            out.push_back(std::move(working));
            return out;
        }
        if (byte == ']') {
            const Structural structural = structural_of(schema_, frame.nodes);
            if (structural.array != kNoSchemaNode) {
                const SchemaNode& array = schema_.node(structural.array);
                if (frame.items < array.min_items.value_or(0)) { return out; }
                if (array.max_items && frame.items > *array.max_items) { return out; }
            }
            frame.phase = Phase::Complete;
            settle(working);
            out.push_back(std::move(working));
        }
        return out;

    case Phase::ObjectValue:
    case Phase::ArrayValue:
    case Phase::Complete: return out;
    }
    return out;
}

JsonGrammar::JsonGrammar(const JsonSchema& schema) : impl_(std::make_unique<Impl>(schema)) {}
JsonGrammar::JsonGrammar(JsonGrammar&&) noexcept            = default;
JsonGrammar& JsonGrammar::operator=(JsonGrammar&&) noexcept = default;
JsonGrammar::~JsonGrammar()                                 = default;

JsonGrammar::JsonGrammar(const JsonGrammar& other) : impl_(std::make_unique<Impl>(*other.impl_)) {}

JsonGrammar& JsonGrammar::operator=(const JsonGrammar& other) {
    if (this != &other) { impl_ = std::make_unique<Impl>(*other.impl_); }
    return *this;
}

void JsonGrammar::begin_string_content() {
    impl_->reset();
    impl_->initialize_inside_string();
}

bool JsonGrammar::feed(std::uint8_t byte) { return impl_->feed(byte); }
ByteSet JsonGrammar::next_bytes() const { return impl_->next_bytes(); }
bool JsonGrammar::complete() const { return impl_->complete(); }
bool JsonGrammar::viable() const noexcept { return impl_->viable(); }
void JsonGrammar::reset() { impl_->reset(); }

std::size_t JsonGrammar::configuration_count() const noexcept {
    return impl_->configuration_count();
}

// --- ByteSet ----------------------------------------------------------------------------------

void ByteSet::insert(std::uint8_t byte) noexcept { words_[byte >> 6U] |= (1ULL << (byte & 63U)); }

void ByteSet::insert_range(std::uint8_t low, std::uint8_t high) noexcept {
    for (std::uint32_t value = low; value <= high; ++value) {
        insert(static_cast<std::uint8_t>(value));
    }
}

void ByteSet::remove(std::uint8_t byte) noexcept {
    words_[byte >> 6U] &= ~(1ULL << (byte & 63U));
}

void ByteSet::insert_all() noexcept {
    words_[0] = words_[1] = words_[2] = words_[3] = ~0ULL;
}

void ByteSet::union_with(const ByteSet& other) noexcept {
    for (std::size_t index = 0; index < 4; ++index) { words_[index] |= other.words_[index]; }
}

bool ByteSet::contains(std::uint8_t byte) const noexcept {
    return (words_[byte >> 6U] & (1ULL << (byte & 63U))) != 0;
}

bool ByteSet::empty() const noexcept {
    return words_[0] == 0 && words_[1] == 0 && words_[2] == 0 && words_[3] == 0;
}

bool ByteSet::full() const noexcept {
    return words_[0] == ~0ULL && words_[1] == ~0ULL && words_[2] == ~0ULL && words_[3] == ~0ULL;
}

std::size_t ByteSet::count() const noexcept {
    std::size_t total = 0;
    for (const std::uint64_t word : words_) {
        std::uint64_t value = word;
        while (value != 0) {
            value &= value - 1U;
            ++total;
        }
    }
    return total;
}

} // namespace ninfer::text
