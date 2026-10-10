// ECMA-262 pattern matching for constrained decoding.
//
// The compiler is continuation-passing: emit(node, next) appends the instructions that match `node`
// and then fall into `next`, returning the entry pc. Thompson's construction falls out of it — a
// concatenation emits its right side first and hands the result down as the left side's
// continuation, and a repeat's back edge is the split it already emitted. Simulation is the usual
// parallel thread walk, with the unanchored-search seed added at every position.

#include "text/regex_engine.h"

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
namespace {

constexpr std::uint32_t kUnbounded = std::numeric_limits<std::uint32_t>::max();
// A bounded repeat is expanded per repetition, so `a{1,100000}` would otherwise be a way to make
// compilation allocate without bound.
constexpr std::size_t kMaxProgramInstructions = 1u << 16u;

struct Range {
    std::uint32_t low  = 0;
    std::uint32_t high = 0;
};

struct CharClass {
    std::vector<Range> ranges;
    bool negate = false;

    [[nodiscard]] bool matches(std::uint32_t codepoint) const noexcept {
        bool inside = false;
        for (const Range& range : ranges) {
            if (codepoint >= range.low && codepoint <= range.high) {
                inside = true;
                break;
            }
        }
        return inside != negate;
    }
};

[[nodiscard]] bool is_word(std::uint32_t codepoint) noexcept {
    return (codepoint >= 'a' && codepoint <= 'z') || (codepoint >= 'A' && codepoint <= 'Z') ||
           (codepoint >= '0' && codepoint <= '9') || codepoint == '_';
}

[[nodiscard]] bool is_line_terminator(std::uint32_t codepoint) noexcept {
    return codepoint == '\n' || codepoint == '\r' || codepoint == 0x2028 || codepoint == 0x2029;
}

[[nodiscard]] CharClass digit_class() {
    return CharClass{.ranges = {{'0', '9'}}, .negate = false};
}

[[nodiscard]] CharClass word_class() {
    return CharClass{.ranges = {{'0', '9'}, {'A', 'Z'}, {'_', '_'}, {'a', 'z'}}, .negate = false};
}

[[nodiscard]] CharClass space_class() {
    return CharClass{.ranges = {{0x09, 0x0D}, {0x20, 0x20}, {0xA0, 0xA0}, {0x2028, 0x2029}},
                     .negate  = false};
}

// --- syntax tree ------------------------------------------------------------------------------

enum class NodeKind : std::uint8_t {
    Empty,
    Char,
    Any,
    Class,
    Concat,
    Alt,
    Repeat,
    AssertStart,
    AssertEnd,
    // A capturing group. `group` is its 1-based index, which a Backref names.
    Capture,
    // `\1`..`\9`: the text the named group matched, matched again literally.
    Backref,
    // `(?=)` `(?!)` `(?<=)` `(?<!)`. `left` holds the assertion body, `negated` and `behind` say which
    // of the four forms this is.
    Lookaround,
};

struct Node {
    NodeKind kind = NodeKind::Empty;
    std::uint32_t codepoint = 0;
    CharClass char_class;
    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;
    std::uint32_t min_repeat = 0;
    std::uint32_t max_repeat = 0;
    bool greedy              = true;
    std::uint32_t group      = 0;
    bool negated             = false;
    bool behind              = false;
};

[[nodiscard]] std::unique_ptr<Node> make_node(NodeKind kind) {
    auto node  = std::make_unique<Node>();
    node->kind = kind;
    return node;
}

// Recursive-descent parser over the pattern. `\uXXXX` is the only escape that produces a code point
// above the BMP; surrogate pairs are combined here so the automaton deals in code points
// everywhere, which is also the unit minLength/maxLength count.
class Parser {
public:
    explicit Parser(std::string_view pattern) : pattern_(pattern) {}

    // Capture groups are numbered by the order their '(' opens, which is the order ECMA-262 uses and
    // what a backreference names. Counted here rather than after parsing so a group nested inside an
    // earlier one still takes the higher number.
    std::uint32_t group_count_ = 0;

    [[nodiscard]] std::unique_ptr<Node> parse() {
        auto node = parse_alternation();
        if (position_ != pattern_.size()) {
            throw std::invalid_argument("pattern has an unmatched ')' at offset " +
                                        std::to_string(position_));
        }
        return node;
    }

private:
    [[nodiscard]] bool at_end() const noexcept { return position_ >= pattern_.size(); }
    [[nodiscard]] char peek() const { return at_end() ? '\0' : pattern_[position_]; }

    [[nodiscard]] std::unique_ptr<Node> parse_alternation() {
        auto left = parse_concatenation();
        while (!at_end() && peek() == '|') {
            ++position_;
            auto right = parse_concatenation();
            auto node  = make_node(NodeKind::Alt);
            node->left  = std::move(left);
            node->right = std::move(right);
            left        = std::move(node);
        }
        return left;
    }

    [[nodiscard]] std::unique_ptr<Node> parse_concatenation() {
        std::unique_ptr<Node> result;
        while (!at_end() && peek() != '|' && peek() != ')') {
            auto piece = parse_quantified();
            if (result == nullptr) {
                result = std::move(piece);
            } else {
                auto node   = make_node(NodeKind::Concat);
                node->left  = std::move(result);
                node->right = std::move(piece);
                result      = std::move(node);
            }
        }
        return result == nullptr ? make_node(NodeKind::Empty) : std::move(result);
    }

    [[nodiscard]] std::unique_ptr<Node> parse_quantified() {
        auto atom = parse_atom();
        if (at_end()) { return atom; }
        switch (peek()) {
        case '*': {
            ++position_;
            return wrap_repeat(std::move(atom), 0, kUnbounded);
        }
        case '+': {
            ++position_;
            return wrap_repeat(std::move(atom), 1, kUnbounded);
        }
        case '?': {
            ++position_;
            return wrap_repeat(std::move(atom), 0, 1);
        }
        case '{': {
            const std::size_t saved = position_;
            if (const auto bounds = try_parse_braces()) {
                return wrap_repeat(std::move(atom), bounds->first, bounds->second);
            }
            // Not a quantifier — ECMA-262 treats a `{` that does not open one as a literal.
            position_ = saved;
            return atom;
        }
        default: return atom;
        }
    }

    [[nodiscard]] std::unique_ptr<Node> wrap_repeat(std::unique_ptr<Node> child, std::uint32_t min,
                                                    std::uint32_t max) {
        bool greedy = true;
        if (!at_end() && peek() == '?') {
            greedy = false;
            ++position_;
        }
        auto node       = make_node(NodeKind::Repeat);
        node->left      = std::move(child);
        node->min_repeat = min;
        node->max_repeat = max;
        node->greedy     = greedy;
        return node;
    }

    [[nodiscard]] std::optional<std::pair<std::uint32_t, std::uint32_t>> try_parse_braces() {
        if (peek() != '{') { return std::nullopt; }
        ++position_;
        std::uint32_t min = 0;
        if (!parse_decimal(min)) { return std::nullopt; }
        std::uint32_t max = min;
        if (!at_end() && peek() == ',') {
            ++position_;
            if (!at_end() && peek() == '}') {
                max = kUnbounded;
            } else if (!parse_decimal(max)) {
                return std::nullopt;
            }
        }
        if (at_end() || peek() != '}') { return std::nullopt; }
        ++position_;
        if (max != kUnbounded && max < min) {
            throw std::invalid_argument("pattern quantifier has an upper bound below its lower");
        }
        return std::make_pair(min, max);
    }

    bool parse_decimal(std::uint32_t& out) {
        if (at_end() || peek() < '0' || peek() > '9') { return false; }
        std::uint64_t value = 0;
        while (!at_end() && peek() >= '0' && peek() <= '9') {
            value = value * 10U + static_cast<std::uint64_t>(peek() - '0');
            if (value > std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument("pattern quantifier is out of range");
            }
            ++position_;
        }
        out = static_cast<std::uint32_t>(value);
        return true;
    }

    [[nodiscard]] std::unique_ptr<Node> parse_atom() {
        if (at_end()) { return make_node(NodeKind::Empty); }
        const char character = pattern_[position_];
        switch (character) {
        case '^': {
            ++position_;
            return make_node(NodeKind::AssertStart);
        }
        case '$': {
            ++position_;
            return make_node(NodeKind::AssertEnd);
        }
        case '.': {
            ++position_;
            return make_node(NodeKind::Any);
        }
        case '(': return parse_group();
        case '[': return parse_class();
        case '\\': return parse_escape();
        case '*':
        case '+':
        case '?':
            throw std::invalid_argument(
                std::string("pattern has a quantifier with nothing to repeat at offset ") +
                std::to_string(position_));
        default: {
            ++position_;
            auto node        = make_node(NodeKind::Char);
            node->codepoint  = static_cast<std::uint32_t>(static_cast<unsigned char>(character));
            return node;
        }
        }
    }

    [[nodiscard]] std::unique_ptr<Node> parse_group() {
        ++position_; // consume '('
        // `Plain` is a non-capturing group, which adds no behaviour of its own; the other three are
        // the assertions, which do.
        enum class Form { Plain, Capture, LookaheadForm, LookbehindForm };
        Form form      = Form::Capture;
        bool negated   = false;
        if (!at_end() && peek() == '?') {
            if (position_ + 1 >= pattern_.size()) {
                throw std::invalid_argument("pattern has an unterminated group");
            }
            const char marker = pattern_[position_ + 1];
            if (marker == ':') {
                position_ += 2;
                form = Form::Plain;
            } else if (marker == '=' || marker == '!') {
                position_ += 2;
                form    = Form::LookaheadForm;
                negated = marker == '!';
            } else if (marker == '<' && position_ + 2 < pattern_.size() &&
                       (pattern_[position_ + 2] == '=' || pattern_[position_ + 2] == '!')) {
                negated = pattern_[position_ + 2] == '!';
                form    = Form::LookbehindForm;
                position_ += 3;
            } else {
                throw std::invalid_argument(
                    "pattern uses a group form that constrained decoding does not implement "
                    "(inline flags and named groups are unsupported)");
            }
        }
        auto inner = parse_alternation();
        if (at_end() || peek() != ')') {
            throw std::invalid_argument("pattern has an unterminated group");
        }
        ++position_; // consume ')'

        if (form == Form::Plain) { return inner; }
        if (form == Form::Capture) {
            if (group_count_ >= kMaxCaptureGroups) {
                throw std::invalid_argument(
                    "pattern has more capture groups than backreferences can name");
            }
            auto node   = make_node(NodeKind::Capture);
            node->group = ++group_count_;
            node->left  = std::move(inner);
            return node;
        }
        auto node     = make_node(NodeKind::Lookaround);
        node->negated = negated;
        node->behind  = form == Form::LookbehindForm;
        node->left    = std::move(inner);
        return node;
    }

    [[nodiscard]] std::unique_ptr<Node> parse_class() {
        ++position_; // consume '['
        CharClass char_class;
        if (!at_end() && peek() == '^') {
            char_class.negate = true;
            ++position_;
        }
        bool first = true;
        while (true) {
            if (at_end()) { throw std::invalid_argument("pattern has an unterminated class"); }
            if (peek() == ']' && !first) { break; }
            first = false;

            if (peek() == '\\') {
                ++position_;
                const std::string_view shorthand = read_escape_letter();
                if (shorthand == "d" || shorthand == "D" || shorthand == "w" ||
                    shorthand == "W" || shorthand == "s" || shorthand == "S") {
                    const bool negated = shorthand[0] >= 'A' && shorthand[0] <= 'Z';
                    const CharClass source =
                        shorthand == "d" || shorthand == "D"
                            ? digit_class()
                            : (shorthand == "w" || shorthand == "W" ? word_class() : space_class());
                    if (negated) {
                        // `\D` inside a class cannot be expressed by adding ranges, so it is
                        // refused rather than silently loosened.
                        throw std::invalid_argument(
                            "pattern uses a negated shorthand inside a class, which is not "
                            "implemented");
                    }
                    for (const Range& range : source.ranges) { char_class.ranges.push_back(range); }
                    continue;
                }
                if (shorthand == "u") {
                    const std::uint32_t low = read_unicode_escape();
                    std::uint32_t high      = low;
                    if (!at_end() && peek() == '-' && position_ + 2 < pattern_.size() &&
                        pattern_[position_ + 1] == '\\' && pattern_[position_ + 2] == 'u') {
                        position_ += 2;
                        high = read_unicode_escape();
                    }
                    char_class.ranges.push_back(Range{low, high});
                    continue;
                }
                if (shorthand.size() != 1) {
                    throw std::invalid_argument("pattern has an unsupported class escape");
                }
                const std::uint32_t low =
                    static_cast<std::uint32_t>(static_cast<unsigned char>(shorthand[0]));
                std::uint32_t high = low;
                if (!at_end() && peek() == '-' && position_ + 1 < pattern_.size() &&
                    pattern_[position_ + 1] != ']') {
                    ++position_;
                    high = static_cast<std::uint32_t>(read_class_literal());
                }
                char_class.ranges.push_back(Range{low, high});
                continue;
            }

            const std::uint32_t low = static_cast<std::uint32_t>(read_class_literal());
            std::uint32_t high      = low;
            if (!at_end() && peek() == '-' && position_ + 1 < pattern_.size() &&
                pattern_[position_ + 1] != ']') {
                ++position_;
                high = static_cast<std::uint32_t>(read_class_literal());
            }
            char_class.ranges.push_back(Range{low, high});
        }
        ++position_; // consume ']'
        auto node         = make_node(NodeKind::Class);
        node->char_class  = std::move(char_class);
        return node;
    }

    [[nodiscard]] char read_class_literal() {
        const char character = pattern_[position_];
        ++position_;
        return character;
    }

    [[nodiscard]] std::string_view read_escape_letter() {
        if (at_end()) { throw std::invalid_argument("pattern ends in a bare backslash"); }
        const char character = pattern_[position_];
        if (character == 'u') {
            ++position_;
            return "u";
        }
        ++position_;
        if (character == 'b' || character == 'B') {
            throw std::invalid_argument(
                "pattern uses a word boundary, which constrained decoding does not implement");
        }
        return std::string_view(pattern_.data() + position_ - 1, 1);
    }

    [[nodiscard]] std::uint32_t read_unicode_escape() {
        if (position_ + 4 > pattern_.size()) {
            throw std::invalid_argument("pattern has a truncated \\u escape");
        }
        std::uint32_t value = 0;
        for (int index = 0; index < 4; ++index) {
            const char digit = pattern_[position_ + static_cast<std::size_t>(index)];
            std::uint32_t nibble = 0;
            if (digit >= '0' && digit <= '9') {
                nibble = static_cast<std::uint32_t>(digit - '0');
            } else if (digit >= 'a' && digit <= 'f') {
                nibble = static_cast<std::uint32_t>(digit - 'a') + 10U;
            } else if (digit >= 'A' && digit <= 'F') {
                nibble = static_cast<std::uint32_t>(digit - 'A') + 10U;
            } else {
                throw std::invalid_argument("pattern has a malformed \\u escape");
            }
            value = value * 16U + nibble;
        }
        position_ += 4;
        // A high surrogate is only half a code point; combine it with the low surrogate that must
        // follow, so the automaton never sees an unpaired half.
        if (value >= 0xD800 && value <= 0xDBFF && position_ + 6 <= pattern_.size() &&
            pattern_[position_] == '\\' && pattern_[position_ + 1] == 'u') {
            position_ += 2;
            const std::uint32_t low = read_unicode_escape();
            if (low >= 0xDC00 && low <= 0xDFFF) {
                return 0x10000U + ((value - 0xD800U) << 10U) + (low - 0xDC00U);
            }
            throw std::invalid_argument("pattern has an unpaired surrogate escape");
        }
        return value;
    }

    [[nodiscard]] std::unique_ptr<Node> parse_escape() {
        ++position_; // consume '\'
        // A backreference is not a character escape: it names a capture group and consumes that
        // group's text, which is what makes the language non-regular and why it is handled here
        // rather than among the letter escapes.
        if (!at_end() && pattern_[position_] >= '1' && pattern_[position_] <= '9') {
            auto node   = make_node(NodeKind::Backref);
            node->group = static_cast<std::uint32_t>(pattern_[position_] - '0');
            ++position_;
            return node;
        }
        const std::string_view letter = read_escape_letter();
        if (letter == "u") {
            auto node       = make_node(NodeKind::Char);
            node->codepoint = read_unicode_escape();
            return node;
        }
        auto node = make_node(NodeKind::Char);
        switch (letter[0]) {
        case 'n': node->codepoint = '\n'; return node;
        case 'r': node->codepoint = '\r'; return node;
        case 't': node->codepoint = '\t'; return node;
        case 'f': node->codepoint = '\f'; return node;
        case 'v': node->codepoint = '\v'; return node;
        case '0': node->codepoint = '\0'; return node;
        case 'd':
        case 'D':
        case 'w':
        case 'W':
        case 's':
        case 'S': {
            const bool negated = letter[0] >= 'A' && letter[0] <= 'Z';
            const char lower   = static_cast<char>(letter[0] | 0x20);
            CharClass base     = lower == 'd' ? digit_class()
                                             : (lower == 'w' ? word_class() : space_class());
            base.negate        = negated;
            auto cls           = make_node(NodeKind::Class);
            cls->char_class    = std::move(base);
            return cls;
        }
        default:
            // An escaped punctuation character stands for itself; ECMA-262 allows `\.`, `\/`, `\\`.
            node->codepoint = static_cast<std::uint32_t>(static_cast<unsigned char>(letter[0]));
            return node;
        }
    }

    std::string_view pattern_;
    std::size_t position_ = 0;
};

// --- NFA --------------------------------------------------------------------------------------

enum class Op : std::uint8_t {
    Char,
    Any,
    Class,
    Split,
    Jump,
    AssertStart,
    AssertEnd,
    Match,
    // Capture group boundaries. Zero-width; they record the current position in the thread state.
    CaptureBegin,
    CaptureEnd,
    // Consumes the text its group captured, one code point per step.
    Backref,
    // Zero-width assertions. `x` is the entry of the body's own fragment, `y` the continuation.
    Lookahead,
    NegativeLookahead,
    Lookbehind,
    NegativeLookbehind,
    // Ends an assertion body. Reached only inside a sub-run, never by the main path.
    SubMatch,
};

struct Inst {
    Op op          = Op::Match;
    std::uint32_t x = 0;
    std::uint32_t y = 0;
    std::uint32_t codepoint = 0;
    std::size_t char_class  = 0;
    std::uint32_t group     = 0;
};

class Compiler {
public:
    [[nodiscard]] std::uint32_t compile(const Node& node, std::uint32_t next) {
        switch (node.kind) {
        case NodeKind::Empty: return next;
        case NodeKind::Char: {
            const std::uint32_t pc = push(Inst{.op = Op::Char, .x = next, .codepoint = node.codepoint});
            return pc;
        }
        case NodeKind::Any: {
            return push(Inst{.op = Op::Any, .x = next});
        }
        case NodeKind::Class: {
            const std::size_t index = classes_.size();
            classes_.push_back(node.char_class);
            return push(Inst{.op = Op::Class, .x = next, .char_class = index});
        }
        case NodeKind::AssertStart: return push(Inst{.op = Op::AssertStart, .x = next});
        case NodeKind::AssertEnd: return push(Inst{.op = Op::AssertEnd, .x = next});
        case NodeKind::Capture: {
            const std::uint32_t end =
                push(Inst{.op = Op::CaptureEnd, .x = next, .group = node.group});
            const std::uint32_t inner = compile(*node.left, end);
            return push(Inst{.op = Op::CaptureBegin, .x = inner, .group = node.group});
        }
        case NodeKind::Backref:
            return push(Inst{.op = Op::Backref, .x = next, .group = node.group});
        case NodeKind::Lookaround: {
            // The body gets its own fragment ending in SubMatch, so it can be run as an independent
            // match; `x` is that fragment's entry and `y` the continuation.
            const std::uint32_t done = push(Inst{.op = Op::SubMatch});
            const std::uint32_t body = compile(*node.left, done);
            const Op op = node.behind ? (node.negated ? Op::NegativeLookbehind : Op::Lookbehind)
                                      : (node.negated ? Op::NegativeLookahead : Op::Lookahead);
            return push(Inst{.op = op, .x = body, .y = next});
        }
        case NodeKind::Concat: {
            const std::uint32_t right = compile(*node.right, next);
            return compile(*node.left, right);
        }
        case NodeKind::Alt: {
            const std::uint32_t left  = compile(*node.left, next);
            const std::uint32_t right = compile(*node.right, next);
            return push(Inst{.op = Op::Split, .x = left, .y = right});
        }
        case NodeKind::Repeat: return compile_repeat(node, next);
        }
        throw std::logic_error("unknown regex node kind");
    }

    // Emitted before the tree so the accept state has a fixed pc that every compiled fragment can
    // use as its continuation.
    [[nodiscard]] std::uint32_t push_match() { return push(Inst{.op = Op::Match}); }

    [[nodiscard]] std::vector<Inst> take_program() { return std::move(program_); }
    [[nodiscard]] std::vector<CharClass> take_classes() { return std::move(classes_); }

private:
    [[nodiscard]] std::uint32_t push(Inst inst) {
        if (program_.size() >= kMaxProgramInstructions) {
            throw std::invalid_argument(
                "pattern expands past the instruction limit; a large bounded repeat is the usual "
                "cause");
        }
        program_.push_back(inst);
        return static_cast<std::uint32_t>(program_.size() - 1);
    }

    [[nodiscard]] std::uint32_t compile_repeat(const Node& node, std::uint32_t next) {
        std::uint32_t cursor = next;
        if (node.max_repeat == kUnbounded) {
            // Reserve the split first so the body can jump back to it, then patch both arms.
            const std::uint32_t split = push(Inst{.op = Op::Split});
            const std::uint32_t body  = compile(*node.left, split);
            program_[split] = Inst{.op = Op::Split,
                                   .x  = node.greedy ? body : cursor,
                                   .y  = node.greedy ? cursor : body};
            cursor = split;
        } else {
            for (std::uint32_t index = node.max_repeat; index > node.min_repeat; --index) {
                const std::uint32_t body = compile(*node.left, cursor);
                cursor = push(Inst{.op = Op::Split,
                                   .x  = node.greedy ? body : cursor,
                                   .y  = node.greedy ? cursor : body});
            }
        }
        for (std::uint32_t index = node.min_repeat; index > 0; --index) {
            cursor = compile(*node.left, cursor);
        }
        return cursor;
    }

    std::vector<Inst> program_;
    std::vector<CharClass> classes_;
};

} // namespace

class RegexMatcher::Impl {
public:
    std::vector<Inst> program;
    std::vector<CharClass> classes;
    std::uint32_t start = 0;

    // Epsilon closure from `pc`. `position` is the number of code points already consumed, which is
    // all `^` needs to decide.
    //
    // Reaching accept does not by itself mean the value matched. A path through `$` only completes
    // if the input *ends* at the current position, and the input does not end until the caller says
    // so. That distinction is why two flags come out of here: `matched` for a path that accepts
    // wherever it stands, `end_capable` for one that accepts only if nothing more follows. Collapsing
    // them makes `^abc$` accept "abcx", because the match recorded at position 3 stays recorded
    // after a fourth code point arrives.
    //
    // `visited` carries one bit per arrival kind, so a pc reachable both through `$` and without it
    // is explored under both.
    using GroupSpans = std::array<std::uint32_t, 2 * kMaxCaptureGroups>;
    using Thread     = RegexMatcher::Cursor::Thread;
    using Parked     = RegexMatcher::Cursor::Parked;

    // What a sub-run of an assertion body concluded. `Alive` is not a failure: the body has neither
    // matched nor died, so more input could still decide it.
    enum class Probe : std::uint8_t { Dead, Matched, Alive };

    struct Seen {
        std::uint32_t pc = 0;
        GroupSpans groups{};
        std::uint8_t kind = 0;
    };

    // Epsilon closure from `state`. Deduction is on (pc, captures, arrival kind): two threads at the
    // same instruction with different captures are genuinely different configurations, and collapsing
    // them would lose the one whose backreference still matches.
    void add_thread(std::vector<Thread>& out, std::vector<Seen>& seen, Thread state,
                    std::size_t position, bool requires_end, const std::vector<std::uint32_t>& input,
                    std::vector<Parked>& parked, bool& end_capable, bool& matched) const {
        while (true) {
            const std::uint8_t kind = requires_end ? 2U : 1U;
            bool known              = false;
            for (const Seen& entry : seen) {
                if (entry.pc == state.pc && entry.kind == kind && entry.groups == state.groups) {
                    known = true;
                    break;
                }
            }
            if (known) { return; }
            seen.push_back(Seen{.pc = state.pc, .groups = state.groups, .kind = kind});

            const Inst& inst = program[state.pc];
            switch (inst.op) {
            case Op::Jump:
                state.pc = inst.x;
                continue;
            case Op::Split:
                add_thread(out, seen, Thread{.pc = inst.x,
                                             .groups = state.groups,
                                             .backref_progress = state.backref_progress},
                           position, requires_end, input, parked, end_capable, matched);
                state.pc = inst.y;
                continue;
            case Op::AssertStart:
                if (position != 0) { return; }
                state.pc = inst.x;
                continue;
            case Op::AssertEnd:
                // Past `$` the path can only close, so nothing further is pushed as a consuming
                // thread.
                add_thread(out, seen, Thread{.pc = inst.x,
                                             .groups = state.groups,
                                             .backref_progress = state.backref_progress},
                           position, true, input, parked, end_capable, matched);
                return;
            case Op::CaptureBegin:
                state.groups[2 * inst.group] = static_cast<std::uint32_t>(position);
                state.pc                     = inst.x;
                continue;
            case Op::CaptureEnd:
                state.groups[2 * inst.group + 1] = static_cast<std::uint32_t>(position);
                state.pc                         = inst.x;
                continue;
            case Op::Backref: {
                const std::uint32_t begin = state.groups[2 * inst.group];
                const std::uint32_t end   = state.groups[2 * inst.group + 1];
                // A group that never matched, or matched nothing, makes the backreference empty —
                // ECMA-262 defines it that way rather than failing the match.
                if (begin == kUnsetCapture || end == kUnsetCapture || end <= begin) {
                    state.pc = inst.x;
                    continue;
                }
                state.backref_progress = 0;
                out.push_back(state);
                return;
            }
            case Op::Lookbehind:
            case Op::NegativeLookbehind: {
                const bool satisfied = matches_before(inst.x, position, input, state.groups);
                if (satisfied == (inst.op == Op::Lookbehind)) {
                    state.pc = inst.y;
                    continue;
                }
                return;
            }
            case Op::Lookahead:
            case Op::NegativeLookahead:
                // Deciding this needs input that has not arrived. The thread waits, keeping the
                // position, because a zero-width assertion resumes the main match *there*.
                parked.push_back(Parked{.pc       = state.pc,
                                        .at       = static_cast<std::uint32_t>(position),
                                        .positive = inst.op == Op::Lookahead,
                                        .groups   = state.groups});
                return;
            case Op::SubMatch:
                matched     = true;
                end_capable = true;
                return;
            case Op::Match:
                if (requires_end) {
                    end_capable = true;
                } else {
                    matched     = true;
                    end_capable = true;
                }
                return;
            default:
                out.push_back(state);
                return;
            }
        }
    }

    // Advances every live configuration over one code point. A backreference consumes the text its
    // group captured, one code point per step, which is what lets a capture of any length be matched
    // by an automaton whose state is still finite.
    [[nodiscard]] std::vector<Thread> advance(const std::vector<Thread>& live, std::uint32_t codepoint,
                                              std::size_t next_position,
                                              const std::vector<std::uint32_t>& input,
                                              std::vector<Parked>& parked, bool& end_capable,
                                              bool& matched) const {
        std::vector<Thread> out;
        std::vector<Seen> seen;
        for (const Thread& thread : live) {
            const Inst& inst = program[thread.pc];
            if (inst.op == Op::Backref) {
                const std::uint32_t begin = thread.groups[2 * inst.group];
                const std::uint32_t end   = thread.groups[2 * inst.group + 1];
                if (begin == kUnsetCapture || end == kUnsetCapture || end <= begin) { continue; }
                const std::uint32_t wanted = begin + thread.backref_progress;
                if (wanted >= end || input[wanted] != codepoint) { continue; }
                if (wanted + 1 == end) {
                    add_thread(out, seen, Thread{.pc = inst.x, .groups = thread.groups},
                               next_position, false, input, parked, end_capable, matched);
                } else {
                    out.push_back(Thread{.pc               = thread.pc,
                                         .groups           = thread.groups,
                                         .backref_progress = thread.backref_progress + 1});
                }
                continue;
            }
            if (consumes(inst, codepoint)) {
                add_thread(out, seen, Thread{.pc = inst.x, .groups = thread.groups}, next_position,
                           false, input, parked, end_capable, matched);
            }
        }
        return out;
    }

    // Runs the fragment at `entry` over `input[from .. to)` and reports whether it reached SubMatch
    // exactly at `to`. A lookbehind body has to end where the main match stands, so the end position
    // is checked rather than searched for.
    [[nodiscard]] bool matches_between(std::uint32_t entry, std::size_t from, std::size_t to,
                                       const std::vector<std::uint32_t>& input,
                                       const GroupSpans& groups) const {
        std::vector<Thread> live;
        std::vector<Seen> seen;
        std::vector<Parked> ignored;
        bool end_capable = false;
        bool matched     = false;
        add_thread(live, seen, Thread{.pc = entry, .groups = groups}, from, false, input, ignored,
                   end_capable, matched);
        if (matched) { return from == to; }
        for (std::size_t index = from; index < to; ++index) {
            bool step_end     = false;
            bool step_matched = false;
            live = advance(live, input[index], index + 1, input, ignored, step_end, step_matched);
            if (step_matched) { return index + 1 == to; }
            if (live.empty()) { return false; }
        }
        return false;
    }

    // True when the fragment at `entry` matches a suffix of `input[0 .. position)` that ends exactly
    // at `position`.
    [[nodiscard]] bool matches_before(std::uint32_t entry, std::size_t position,
                                      const std::vector<std::uint32_t>& input,
                                      const GroupSpans& groups) const {
        for (std::size_t start = 0; start <= position; ++start) {
            if (matches_between(entry, start, position, input, groups)) { return true; }
        }
        return false;
    }

    // Whether the assertion body at `entry`, evaluated at `at`, has matched, died, or is still open.
    [[nodiscard]] Probe probe_assertion(std::uint32_t entry, std::size_t at,
                                        const std::vector<std::uint32_t>& input,
                                        const GroupSpans& groups) const {
        std::vector<Thread> live;
        std::vector<Seen> seen;
        std::vector<Parked> ignored;
        bool end_capable = false;
        bool matched     = false;
        add_thread(live, seen, Thread{.pc = entry, .groups = groups}, at, false, input, ignored,
                   end_capable, matched);
        if (matched) { return Probe::Matched; }
        for (std::size_t index = at; index < input.size(); ++index) {
            bool step_end     = false;
            bool step_matched = false;
            live = advance(live, input[index], index + 1, input, ignored, step_end, step_matched);
            if (step_matched) { return Probe::Matched; }
            if (live.empty()) { return Probe::Dead; }
        }
        return Probe::Alive;
    }

    [[nodiscard]] bool consumes(const Inst& inst, std::uint32_t codepoint) const {
        switch (inst.op) {
        case Op::Char: return inst.codepoint == codepoint;
        case Op::Any: return !is_line_terminator(codepoint);
        case Op::Class: return classes[inst.char_class].matches(codepoint);
        default: return false;
        }
    }
};

RegexMatcher::RegexMatcher() : impl_(std::make_unique<Impl>()) {}
RegexMatcher::RegexMatcher(RegexMatcher&&) noexcept            = default;
RegexMatcher& RegexMatcher::operator=(RegexMatcher&&) noexcept = default;
RegexMatcher::~RegexMatcher()                                  = default;

std::unique_ptr<RegexMatcher> RegexMatcher::compile(std::string_view pattern) {
    Parser parser(pattern);
    auto tree = parser.parse();
    Compiler compiler;
    // The implicit `.*` in front of an unanchored pattern is not emitted: search mode seeds a fresh
    // start thread at every position instead, which is equivalent and keeps `^` meaningful.
    const std::uint32_t match_pc = compiler.push_match();
    const std::uint32_t start    = compiler.compile(*tree, match_pc);

    auto matcher       = std::unique_ptr<RegexMatcher>(new RegexMatcher());
    matcher->impl_->program = compiler.take_program();
    matcher->impl_->classes = compiler.take_classes();
    matcher->impl_->start   = start;
    return matcher;
}

RegexMatcher::Cursor RegexMatcher::begin() const {
    using Thread = Cursor::Thread;
    Cursor cursor;
    std::vector<Impl::Seen> seen;
    std::vector<Cursor::Parked> parked;
    bool end_capable = false;
    bool matched     = false;
    impl_->add_thread(cursor.threads_, seen, Thread{.pc = impl_->start, .groups = {}}, 0, false,
                      cursor.input_, parked, end_capable, matched);
    cursor.parked_      = std::move(parked);
    cursor.matched_     = matched;
    cursor.end_capable_ = end_capable;
    cursor.position_    = 0;
    return cursor;
}

bool RegexMatcher::step(Cursor& cursor, std::uint32_t codepoint) const {
    using Thread = Cursor::Thread;
    const std::size_t next_position = cursor.position_ + 1;
    bool end_capable                = false;
    bool matched                    = false;
    std::vector<Cursor::Parked> parked = std::move(cursor.parked_);

    // The code point joins the retained input *before* anything is closed over it. A lookbehind and
    // the search-mode start thread are both evaluated at this position, and both read the text up to
    // and including it — appending afterwards would have them index one past the end.
    cursor.input_.push_back(codepoint);
    cursor.position_ = next_position;

    std::vector<Thread> threads =
        impl_->advance(cursor.threads_, codepoint, next_position, cursor.input_, parked,
                       end_capable, matched);
    // Search mode: a match may begin at this position, which is the `.*` prefix an unanchored
    // pattern carries.
    std::vector<Impl::Seen> seen;
    impl_->add_thread(threads, seen, Thread{.pc = impl_->start, .groups = {}}, next_position, false,
                      cursor.input_, parked, end_capable, matched);

    // A parked lookahead is decided against the input that has arrived since. Until it resolves the
    // value can neither close nor be called dead, so it stays pending rather than being guessed at.
    std::vector<Cursor::Parked> unresolved;
    for (const Cursor::Parked& entry : parked) {
        const Inst& inst         = impl_->program[entry.pc];
        const bool positive      = inst.op == Op::Lookahead;
        const Impl::Probe probe  = impl_->probe_assertion(inst.x, entry.at, cursor.input_,
                                                          entry.groups);
        if (probe == Impl::Probe::Alive) {
            unresolved.push_back(entry);
            continue;
        }
        if ((probe == Impl::Probe::Matched) != positive) { continue; }

        // Satisfied. The assertion is zero-width, so the main path resumes where it was parked and
        // re-consumes everything that arrived while it waited — which is why the parked entry keeps
        // its position instead of the current one.
        std::vector<Impl::Seen> resumed_seen;
        std::vector<Thread> resumed;
        impl_->add_thread(resumed, resumed_seen, Thread{.pc = inst.y, .groups = entry.groups},
                          entry.at, false, cursor.input_, unresolved, end_capable, matched);
        for (std::size_t index = entry.at; index < cursor.input_.size() && !resumed.empty();
             ++index) {
            resumed = impl_->advance(resumed, cursor.input_[index], index + 1, cursor.input_,
                                     unresolved, end_capable, matched);
        }
        for (const Thread& thread : resumed) { threads.push_back(thread); }
    }

    cursor.threads_ = std::move(threads);
    cursor.parked_  = std::move(unresolved);
    // `end_capable_` describes the current position only: the value may close here. It is not
    // carried over, because a code point that arrives after the end-anchored path closed means the
    // value did not end there after all.
    cursor.end_capable_ = end_capable;
    // `matched_` is carried over: a pattern that matched without an end anchor stays satisfied for
    // every continuation.
    cursor.matched_ = cursor.matched_ || matched;
    return cursor.matched_ || cursor.end_capable_ || !cursor.threads_.empty() ||
           !cursor.parked_.empty();
}

bool RegexMatcher::matched(const Cursor& cursor) noexcept {
    // A parked lookahead has not resolved, so nothing that depends on it can be called satisfied.
    return cursor.matched_ || (cursor.end_capable_ && cursor.parked_.empty());
}

bool RegexMatcher::exhausted(const Cursor& cursor) const noexcept {
    // `end_capable_` counts: a cursor whose only live path closed past an end anchor has no consuming
    // thread left and no `matched_` flag either, because a match that needs the value to end is not
    // a match until it does. Reading that as "no continuation can match" refuses the very code point
    // that would let the value close — `^(cat|dog|bird)$` could never be given its final letter.
    //
    // A parked lookahead counts as alive: its assertion may still be satisfied by what comes next, so
    // the value is not dead merely because the main path is waiting.
    return !impl_->program.empty() && cursor.threads_.empty() && cursor.parked_.empty() &&
           !cursor.matched_ && !cursor.end_capable_;
}

} // namespace ninfer::text
