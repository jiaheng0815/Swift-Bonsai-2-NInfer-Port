#include "targets/qwen3_6/impl/frontend/tool_call_state.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace ninfer::targets::qwen3_6::frontend_internal {
namespace {

using Json = nlohmann::json;

constexpr std::string_view kToolOpen      = "<tool_call>";
constexpr std::string_view kFunctionClose = "</function>";
constexpr std::string_view kParameterOpen = "<parameter=";
constexpr std::size_t kNoTool             = static_cast<std::size_t>(-1);

enum class Phase : std::uint8_t {
    // Before the opening marker. Nothing is constrained yet: the model may still be answering in
    // prose, and an argument schema governs nothing until the call has begun.
    Outside,
    // Inside `<function=...>`, reading the tool name. The schema that governs the arguments is not
    // known until this ends, so nothing is constrained here either.
    FunctionName,
    // Inside the function body, between parameters.
    Body,
    // Inside `<parameter=...>`, reading the property name.
    ParameterName,
    // Inside a parameter's value.
    Value,
};

// The longest suffix of `text` that is a prefix of `marker`, which is how a marker split across two
// tokens is still recognized without buffering the whole stream.
std::size_t longest_suffix_prefix(std::string_view text, std::string_view marker) {
    const std::size_t limit = std::min(text.size(), marker.size());
    for (std::size_t length = limit; length > 0; --length) {
        if (text.compare(text.size() - length, length, marker, 0, length) == 0) { return length; }
    }
    return 0;
}

} // namespace

std::size_t ToolArgumentShape::index_of(std::string_view name) const noexcept {
    for (std::size_t index = 0; index < names.size(); ++index) {
        if (names[index] == name) { return index; }
    }
    return names.size();
}

ToolArgumentShape compile_tool_argument_shape(std::string_view schema) {
    Json parsed;
    try {
        parsed = Json::parse(schema);
    } catch (const Json::exception& error) {
        throw std::invalid_argument(std::string("tool input_schema is not valid JSON: ") +
                                    error.what());
    }
    if (!parsed.is_object()) {
        throw std::invalid_argument("tool input_schema must be a JSON object");
    }

    ToolArgumentShape shape;
    const Json properties = parsed.contains("properties") && parsed.at("properties").is_object()
                                ? parsed.at("properties")
                                : Json::object();
    std::vector<std::string> required;
    if (parsed.contains("required") && parsed.at("required").is_array()) {
        for (const Json& name : parsed.at("required")) {
            if (name.is_string()) { required.push_back(name.get<std::string>()); }
        }
    }
    for (const auto& [name, declared] : properties.items()) {
        const bool is_required =
            std::find(required.begin(), required.end(), name) != required.end();
        // A `string` argument is written as raw text and terminated by the newline before
        // `</parameter>`; every other type is written as the JSON literal it denotes.
        const bool text_value = declared.is_object() && declared.contains("type") &&
                                declared.at("type").is_string() &&
                                declared.at("type").get<std::string>() == "string";
        shape.names.push_back(name);
        shape.required.push_back(is_required ? 1U : 0U);
        shape.value_is_text.push_back(text_value ? 1U : 0U);
        shape.values.push_back(std::make_shared<const text::JsonSchema>(
            text::JsonSchema::compile(declared.dump())));
    }
    return shape;
}

struct ToolCallState::Impl {
    Phase phase = Phase::Outside;
    // Bytes held while a marker is being matched. Both markers in the body open with `<`, so the
    // decision between them waits until they differ.
    std::string pending;
    std::string name;
    std::string parameter;
    std::vector<std::pair<std::string, ToolArgumentShape>> tools;
    std::size_t selected = kNoTool;
    std::vector<std::uint8_t> seen;
    std::size_t active = 0;
    // The value's own constraint. A `string` argument is constrained as content that never had an
    // opening quote, which is what the JSON automaton is told when it is armed.
    std::unique_ptr<text::JsonGrammar> value;

    Impl() = default;
    Impl(const Impl& other)
        : phase(other.phase), pending(other.pending), name(other.name),
          parameter(other.parameter), tools(other.tools), selected(other.selected),
          seen(other.seen), active(other.active) {
        // The value grammar is owned, so it is duplicated rather than shared: a rolled-back round
        // must leave the committed machine exactly where it was.
        if (other.value != nullptr) {
            value = std::make_unique<text::JsonGrammar>(*other.value);
        }
    }

    [[nodiscard]] const ToolArgumentShape* shape() const {
        return selected == kNoTool ? nullptr : &tools[selected].second;
    }

    [[nodiscard]] bool all_required_seen() const {
        const ToolArgumentShape* declared = shape();
        for (std::size_t index = 0; index < declared->names.size(); ++index) {
            if (declared->required[index] != 0 && seen[index] == 0) { return false; }
        }
        return true;
    }

    [[nodiscard]] bool any_unseen() const {
        const ToolArgumentShape* declared = shape();
        for (std::size_t index = 0; index < declared->names.size(); ++index) {
            if (seen[index] == 0) { return true; }
        }
        return false;
    }

    void arm_value() {
        const ToolArgumentShape* declared = shape();
        value = std::make_unique<text::JsonGrammar>(*declared->values[active]);
        if (declared->value_is_text[active] != 0) { value->begin_string_content(); }
    }
};

ToolCallState::ToolCallState() : impl_(std::make_unique<Impl>()) {}
ToolCallState::~ToolCallState() = default;

ToolCallState::ToolCallState(const ToolCallState& other)
    : impl_(std::make_unique<Impl>(*other.impl_)) {}

ToolCallState& ToolCallState::operator=(const ToolCallState& other) {
    if (this != &other) {
        ToolCallState copy(other);
        std::swap(impl_, copy.impl_);
    }
    return *this;
}

void ToolCallState::register_tool(std::string name, ToolArgumentShape shape) {
    impl_->tools.emplace_back(std::move(name), std::move(shape));
}

bool ToolCallState::constrained() const noexcept {
    switch (impl_->phase) {
    case Phase::Body:
    case Phase::ParameterName: return impl_->shape() != nullptr;
    case Phase::Value: return impl_->value != nullptr;
    default: return false;
    }
}

text::ByteSet ToolCallState::next_bytes() const {
    text::ByteSet set;
    const ToolArgumentShape* declared = impl_->shape();
    switch (impl_->phase) {
    case Phase::Body: {
        if (!impl_->pending.empty()) {
            // Mid-marker: only the byte that continues a still-possible marker may follow.
            const std::string_view text = impl_->pending;
            if (declared != nullptr && impl_->any_unseen() &&
                kParameterOpen.size() > text.size() &&
                kParameterOpen.compare(0, text.size(), text) == 0) {
                set.insert(static_cast<std::uint8_t>(kParameterOpen[text.size()]));
            }
            if (declared != nullptr && impl_->all_required_seen() &&
                kFunctionClose.size() > text.size() &&
                kFunctionClose.compare(0, text.size(), text) == 0) {
                set.insert(static_cast<std::uint8_t>(kFunctionClose[text.size()]));
            }
            return set;
        }
        set.insert('\n');
        // A parameter slot opens only while a declared property is still unwritten, and the body
        // closes only once every required one has been written.
        if (impl_->any_unseen() || impl_->all_required_seen()) { set.insert('<'); }
        return set;
    }
    case Phase::ParameterName: {
        // Only a declared, unwritten property may be named, so the name slot admits exactly the
        // bytes that continue one — the rule the JSON automaton applies to an object key.
        for (std::size_t index = 0; index < declared->names.size(); ++index) {
            if (impl_->seen[index] != 0) { continue; }
            const std::string& name = declared->names[index];
            if (name.size() <= impl_->parameter.size()) { continue; }
            if (name.compare(0, impl_->parameter.size(), impl_->parameter) != 0) { continue; }
            set.insert(static_cast<std::uint8_t>(name[impl_->parameter.size()]));
        }
        return set;
    }
    case Phase::Value:
        if (impl_->value->complete()) { set.insert('\n'); }
        return set;
    default: return set;
    }
}

bool ToolCallState::feed(std::uint8_t byte) {
    switch (impl_->phase) {
    case Phase::Outside:
        impl_->pending.push_back(static_cast<char>(byte));
        if (impl_->pending.size() >= kToolOpen.size() && impl_->pending.ends_with(kToolOpen)) {
            impl_->pending.clear();
            impl_->phase = Phase::FunctionName;
            return true;
        }
        impl_->pending.erase(0, impl_->pending.size() -
                                    longest_suffix_prefix(impl_->pending, kToolOpen));
        return true;

    case Phase::FunctionName:
        if (byte != '>') {
            impl_->name.push_back(static_cast<char>(byte));
            return true;
        }
        // The name selects the schema governing everything that follows. An unregistered name leaves
        // the machine unconstrained rather than guessing at a shape.
        impl_->selected = kNoTool;
        for (std::size_t index = 0; index < impl_->tools.size(); ++index) {
            if (impl_->tools[index].first == impl_->name) {
                impl_->selected = index;
                break;
            }
        }
        impl_->name.clear();
        if (impl_->selected == kNoTool) {
            impl_->phase = Phase::Outside;
            return true;
        }
        impl_->seen.assign(impl_->shape()->names.size(), 0);
        impl_->phase = Phase::Body;
        return true;

    case Phase::Body: {
        if (impl_->pending.empty()) {
            if (byte == '\n') { return true; }
            if (byte != '<') { return false; }
            impl_->pending.assign(1, '<');
            return true;
        }
        impl_->pending.push_back(static_cast<char>(byte));
        const std::string_view text = impl_->pending;
        const ToolArgumentShape* declared = impl_->shape();
        const bool parameter_possible =
            declared != nullptr && impl_->any_unseen() && kParameterOpen.size() >= text.size() &&
            kParameterOpen.compare(0, text.size(), text) == 0;
        const bool close_possible = declared != nullptr && impl_->all_required_seen() &&
                                    kFunctionClose.size() >= text.size() &&
                                    kFunctionClose.compare(0, text.size(), text) == 0;
        if (!parameter_possible && !close_possible) {
            impl_->pending.clear();
            return false;
        }
        if (text.size() == kParameterOpen.size()) {
            impl_->pending.clear();
            impl_->parameter.clear();
            impl_->phase = Phase::ParameterName;
        } else if (text.size() == kFunctionClose.size()) {
            impl_->pending.clear();
            impl_->selected = kNoTool;
            impl_->phase    = Phase::Outside;
        }
        return true;
    }

    case Phase::ParameterName:
        if (byte == '>') {
            const std::size_t index = impl_->shape()->index_of(impl_->parameter);
            if (index >= impl_->shape()->names.size() || impl_->seen[index] != 0) {
                impl_->parameter.clear();
                return false;
            }
            impl_->seen[index] = 1;
            impl_->active      = index;
            impl_->parameter.clear();
            impl_->arm_value();
            impl_->phase = Phase::Value;
            return true;
        }
        impl_->parameter.push_back(static_cast<char>(byte));
        return true;

    case Phase::Value:
        if (byte == '\n') {
            // The value ends at the newline before `</parameter>`, so the byte is admissible only
            // once the value satisfies its schema. Refusing it otherwise is what keeps a half-written
            // argument from being closed.
            if (!impl_->value->complete()) { return false; }
            impl_->value.reset();
            impl_->phase = Phase::Body;
            return true;
        }
        return impl_->value->feed(byte);
    }
    return false;
}

} // namespace ninfer::targets::qwen3_6::frontend_internal
