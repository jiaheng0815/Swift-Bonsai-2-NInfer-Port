// JSON Schema normalization for constrained decoding.
//
// The walk in this file is location-driven, not reference-driven. Every schema *location* in the
// document (a JSON Pointer) is assigned an id at most once, and the id is recorded before the
// location's children are compiled. A `$ref` that points back at an ancestor therefore finds an
// existing id instead of recursing again, which is the whole reason a recursive schema compiles to
// a finite graph. Nothing here unfolds a recursive definition.

#include "text/json_schema.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <deque>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace ninfer::text {
namespace {

using Json = nlohmann::ordered_json;

// Structural limits. A schema document is attacker-controlled when it arrives over HTTP, so the
// walk is bounded rather than trusting the input to be shallow.
constexpr std::size_t kMaxSchemaNodes = 4096;
constexpr std::size_t kMaxSchemaDepth = 64;

// Keywords whose meaning this compiler reproduces with the normalized node fields.
const std::unordered_set<std::string>& modeled_keywords() {
    static const std::unordered_set<std::string> keywords{
        "$ref",         "type",          "properties",        "required",
        "additionalProperties",          "minProperties",     "maxProperties",
        "prefixItems",  "items",         "minItems",          "maxItems",
        "minLength",    "maxLength",     "pattern",           "minimum",
        "maximum",      "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
        "enum",         "const",         "anyOf",             "oneOf",
        "allOf"};
    return keywords;
}

// Keywords that *would* narrow the admitted language if they were honored. They are not modeled,
// so a document using one is compiled to a strictly weaker constraint than it asked for. The
// caller learns about it through unsupported_keywords() and decides whether that is acceptable;
// silently dropping them is the one outcome that must not happen quietly.
const std::unordered_set<std::string>& unmodeled_constraint_keywords() {
    static const std::unordered_set<std::string> keywords{
        "not",            "if",               "then",             "else",
        "dependentRequired",                  "dependentSchemas", "propertyNames",
        "patternProperties",                  "contains",         "minContains",
        "maxContains",    "uniqueItems",      "unevaluatedProperties",
        "unevaluatedItems"};
    // `format` is deliberately absent: JSON Schema 2020-12 makes it an annotation unless an
    // implementation opts into assertion, so not enforcing it is the specification's own default
    // and reporting it would be a false alarm on ordinary documents.
    return keywords;
}

// Keywords that carry no constraint: annotations, identifiers, and the container that holds
// reusable definitions. `$defs`/`definitions` are reached only through `$ref`, never compiled as
// schemas in their own right, so they are neither modeled nor reported.
bool is_annotation_keyword(const std::string& name) {
    static const std::unordered_set<std::string> keywords{
        "$schema", "$id",    "$anchor", "$comment", "$defs",  "definitions",
        "title",   "description", "examples", "default", "deprecated", "readOnly",
        "writeOnly"};
    return keywords.count(name) != 0;
}

[[nodiscard]] JsonKind parse_kind(const std::string& text) {
    if (text == "null") { return JsonKind::Null; }
    if (text == "boolean") { return JsonKind::Boolean; }
    if (text == "object") { return JsonKind::Object; }
    if (text == "array") { return JsonKind::Array; }
    if (text == "number") { return JsonKind::Number; }
    if (text == "integer") { return JsonKind::Integer; }
    if (text == "string") { return JsonKind::String; }
    throw std::invalid_argument("unknown JSON Schema type '" + text + "'");
}

// Appends one JSON Pointer reference token, escaping `~` and `/` per RFC 6901.
[[nodiscard]] std::string pointer_child(const std::string& parent, const std::string& token) {
    std::string escaped;
    escaped.reserve(token.size());
    for (const char character : token) {
        if (character == '~') {
            escaped += "~0";
        } else if (character == '/') {
            escaped += "~1";
        } else {
            escaped.push_back(character);
        }
    }
    return parent + "/" + escaped;
}

} // namespace

std::string_view json_kind_name(JsonKind kind) noexcept {
    switch (kind) {
    case JsonKind::Null: return "null";
    case JsonKind::Boolean: return "boolean";
    case JsonKind::Object: return "object";
    case JsonKind::Array: return "array";
    case JsonKind::Number: return "number";
    case JsonKind::Integer: return "integer";
    case JsonKind::String: return "string";
    }
    return "unknown";
}

class JsonSchema::Impl {
public:
    Json document;
    // JSON Pointer -> node id. Populated *before* a location's children are compiled so a `$ref`
    // cycle resolves to the id of the location already being built.
    std::unordered_map<std::string, SchemaNodeId> by_pointer;
    // A deque, not a vector: compile_at holds a reference to the node it is filling while it
    // recurses into children, and a vector's growth would reallocate that node out from under the
    // reference. Deque element addresses survive push_back.
    std::deque<SchemaNode> nodes;
    // The id compile_at("") produced. Held rather than assumed to be zero so the accessor cannot
    // silently disagree with the walk if the reservation order ever changes.
    SchemaNodeId root = kNoSchemaNode;
    std::vector<std::string> unsupported;
    std::unordered_set<std::string> unsupported_seen;

    [[nodiscard]] const Json& at(const std::string& pointer) const {
        try {
            return document.at(Json::json_pointer(pointer));
        } catch (const std::exception&) {
            throw std::invalid_argument("schema reference does not resolve: '#" + pointer + "'");
        }
    }

    [[nodiscard]] SchemaNodeId reserve() {
        if (nodes.size() >= kMaxSchemaNodes) {
            throw std::invalid_argument("schema exceeds the node limit");
        }
        nodes.emplace_back();
        return static_cast<SchemaNodeId>(nodes.size() - 1);
    }

    void note_unsupported(const std::string& keyword) {
        if (unsupported_seen.insert(keyword).second) { unsupported.push_back(keyword); }
    }

    [[nodiscard]] SchemaNodeId compile_at(const std::string& pointer, std::size_t depth);
    [[nodiscard]] SchemaNodeId build_literal(const Json& value, std::size_t depth);

private:
    void parse_string_keywords(const Json& value, SchemaNode& node) const;
    void parse_number_keywords(const Json& value, SchemaNode& node) const;
    void parse_object_keywords(const std::string& pointer, const Json& value, SchemaNode& node,
                               std::size_t depth);
    void parse_array_keywords(const std::string& pointer, const Json& value, SchemaNode& node,
                              std::size_t depth);
    void parse_combinators(const std::string& pointer, const Json& value, SchemaNode& node,
                           std::size_t depth);
    void parse_enumeration(const Json& value, SchemaNode& node, std::size_t depth);
    void scan_unmodeled(const Json& value);
};

void JsonSchema::Impl::parse_string_keywords(const Json& value, SchemaNode& node) const {
    const auto length_keyword = [&](const char* key, std::optional<std::size_t>& target) {
        if (!value.contains(key)) { return; }
        const Json& entry = value.at(key);
        if (!entry.is_number_unsigned()) {
            throw std::invalid_argument(std::string(key) + " must be a non-negative integer");
        }
        target = entry.get<std::size_t>();
    };
    length_keyword("minLength", node.min_length);
    length_keyword("maxLength", node.max_length);
    if (value.contains("pattern")) {
        const Json& entry = value.at("pattern");
        if (!entry.is_string()) { throw std::invalid_argument("pattern must be a string"); }
        node.pattern = entry.get<std::string>();
    }
}

void JsonSchema::Impl::parse_number_keywords(const Json& value, SchemaNode& node) const {
    const auto numeric_keyword = [&](const char* key, std::optional<double>& target) {
        if (!value.contains(key)) { return; }
        const Json& entry = value.at(key);
        if (!entry.is_number()) {
            throw std::invalid_argument(std::string(key) + " must be a number");
        }
        target = entry.get<double>();
    };
    numeric_keyword("minimum", node.minimum);
    numeric_keyword("maximum", node.maximum);
    numeric_keyword("exclusiveMinimum", node.exclusive_minimum);
    numeric_keyword("exclusiveMaximum", node.exclusive_maximum);
    numeric_keyword("multipleOf", node.multiple_of);
    if (node.multiple_of && !(*node.multiple_of > 0.0)) {
        throw std::invalid_argument("multipleOf must be greater than zero");
    }
}

void JsonSchema::Impl::parse_object_keywords(const std::string& pointer, const Json& value,
                                             SchemaNode& node, std::size_t depth) {
    if (value.contains("properties")) {
        const Json& properties = value.at("properties");
        if (!properties.is_object()) { throw std::invalid_argument("properties must be an object"); }
        for (auto entry = properties.begin(); entry != properties.end(); ++entry) {
            if (!entry.value().is_object()) {
                throw std::invalid_argument("each properties entry must be a schema object");
            }
            const std::string child = pointer_child(pointer_child(pointer, "properties"),
                                                    entry.key());
            node.properties.emplace_back(entry.key(), compile_at(child, depth + 1));
        }
    }
    if (value.contains("required")) {
        const Json& required = value.at("required");
        if (!required.is_array()) { throw std::invalid_argument("required must be an array"); }
        for (const Json& entry : required) {
            if (!entry.is_string()) {
                throw std::invalid_argument("required entries must be strings");
            }
            node.required.push_back(entry.get<std::string>());
        }
    }
    if (value.contains("additionalProperties")) {
        const Json& additional = value.at("additionalProperties");
        if (additional.is_boolean()) {
            node.additional = additional.get<bool>() ? AdditionalProperties::Allowed
                                                     : AdditionalProperties::Denied;
        } else if (additional.is_object()) {
            node.additional        = AdditionalProperties::Schema;
            node.additional_schema = compile_at(pointer_child(pointer, "additionalProperties"),
                                                depth + 1);
        } else {
            throw std::invalid_argument("additionalProperties must be a boolean or a schema");
        }
    }
    const auto count_keyword = [&](const char* key, std::optional<std::size_t>& target) {
        if (!value.contains(key)) { return; }
        const Json& entry = value.at(key);
        if (!entry.is_number_unsigned()) {
            throw std::invalid_argument(std::string(key) + " must be a non-negative integer");
        }
        target = entry.get<std::size_t>();
    };
    count_keyword("minProperties", node.min_properties);
    count_keyword("maxProperties", node.max_properties);
}

void JsonSchema::Impl::parse_array_keywords(const std::string& pointer, const Json& value,
                                            SchemaNode& node, std::size_t depth) {
    const auto prefix_keyword = [&](const char* key) {
        if (!value.contains(key)) { return; }
        const Json& items = value.at(key);
        if (!items.is_array()) { throw std::invalid_argument(std::string(key) + " must be an array"); }
        std::size_t index = 0;
        for (const Json& entry : items) {
            if (!entry.is_object()) {
                throw std::invalid_argument(std::string(key) + " entries must be schema objects");
            }
            const std::string child =
                pointer_child(pointer_child(pointer, key), std::to_string(index));
            node.prefix_items.push_back(compile_at(child, depth + 1));
            ++index;
        }
    };
    // Draft-07 spells tuple schemas as an array-valued `items`; `prefixItems` is the same idea under
    // its modern name. Both land in prefix_items, so the automaton sees one representation.
    prefix_keyword("prefixItems");
    if (value.contains("items")) {
        const Json& items = value.at("items");
        if (items.is_object()) {
            node.items = compile_at(pointer_child(pointer, "items"), depth + 1);
        } else if (items.is_array()) {
            prefix_keyword("items");
        } else {
            throw std::invalid_argument("items must be a schema object or an array of them");
        }
    }
    const auto count_keyword = [&](const char* key, std::optional<std::size_t>& target) {
        if (!value.contains(key)) { return; }
        const Json& entry = value.at(key);
        if (!entry.is_number_unsigned()) {
            throw std::invalid_argument(std::string(key) + " must be a non-negative integer");
        }
        target = entry.get<std::size_t>();
    };
    count_keyword("minItems", node.min_items);
    count_keyword("maxItems", node.max_items);
}

void JsonSchema::Impl::parse_combinators(const std::string& pointer, const Json& value,
                                         SchemaNode& node, std::size_t depth) {
    const auto combination = [&](const char* key, std::vector<SchemaNodeId>& target) {
        if (!value.contains(key)) { return; }
        const Json& members = value.at(key);
        if (!members.is_array() || members.empty()) {
            throw std::invalid_argument(std::string(key) + " must be a non-empty array");
        }
        std::size_t index = 0;
        for (const Json& entry : members) {
            if (!entry.is_object()) {
                throw std::invalid_argument(std::string(key) + " entries must be schema objects");
            }
            const std::string child =
                pointer_child(pointer_child(pointer, key), std::to_string(index));
            target.push_back(compile_at(child, depth + 1));
            ++index;
        }
    };
    combination("anyOf", node.any_of);
    combination("oneOf", node.one_of);
    combination("allOf", node.all_of);
}

void JsonSchema::Impl::parse_enumeration(const Json& value, SchemaNode& node, std::size_t depth) {
    if (value.contains("enum")) {
        const Json& entries = value.at("enum");
        if (!entries.is_array() || entries.empty()) {
            throw std::invalid_argument("enum must be a non-empty array");
        }
        for (const Json& entry : entries) { node.exact_values.push_back(build_literal(entry, depth)); }
    }
    if (value.contains("const")) { node.exact_values.push_back(build_literal(value.at("const"), depth)); }
}

void JsonSchema::Impl::scan_unmodeled(const Json& value) {
    for (auto entry = value.begin(); entry != value.end(); ++entry) {
        const std::string& key = entry.key();
        if (modeled_keywords().count(key) != 0) { continue; }
        if (is_annotation_keyword(key)) { continue; }
        // Unknown vendor extensions (`x-...`) and unknown keywords are not constraint promises, so
        // they are not reported. Only keywords that *do* promise a narrower language are.
        if (unmodeled_constraint_keywords().count(key) != 0) { note_unsupported(key); }
    }
}

// Builds the node for one literal JSON value, as produced by an `enum` entry or `const`. Strings and
// booleans carry their decoded value rather than a serialized spelling, so the automaton's ordinary
// whitespace and escape handling decides what bytes are acceptable.
SchemaNodeId JsonSchema::Impl::build_literal(const Json& value, std::size_t depth) {
    if (depth > kMaxSchemaDepth) {
        throw std::invalid_argument("schema nesting exceeds the depth limit");
    }
    const SchemaNodeId id = reserve();
    SchemaNode& node      = nodes[id];
    if (value.is_null()) {
        node.kinds = {JsonKind::Null};
        return id;
    }
    if (value.is_boolean()) {
        node.kinds         = {JsonKind::Boolean};
        node.exact_boolean = value.get<bool>();
        return id;
    }
    if (value.is_number()) {
        node.kinds = {value.is_number_integer() || value.is_number_unsigned() ? JsonKind::Integer
                                                                             : JsonKind::Number};
        node.minimum = node.maximum = value.get<double>();
        return id;
    }
    if (value.is_string()) {
        node.kinds        = {JsonKind::String};
        node.exact_string = value.get<std::string>();
        return id;
    }
    if (value.is_array()) {
        node.kinds = {JsonKind::Array};
        node.min_items = node.max_items = value.size();
        for (const Json& entry : value) {
            node.prefix_items.push_back(build_literal(entry, depth + 1));
        }
        return id;
    }
    node.kinds = {JsonKind::Object};
    node.min_properties = node.max_properties = value.size();
    // Literal objects admit exactly the members present, so every property is required and nothing
    // else is allowed. Serializing as a schema would lose the distinction between a literal `{"a":1}`
    // and the schema that merely admits it.
    node.additional = AdditionalProperties::Denied;
    for (auto entry = value.begin(); entry != value.end(); ++entry) {
        node.properties.emplace_back(entry.key(), build_literal(entry.value(), depth + 1));
        node.required.push_back(entry.key());
    }
    return id;
}

SchemaNodeId JsonSchema::Impl::compile_at(const std::string& pointer, std::size_t depth) {
    if (const auto found = by_pointer.find(pointer); found != by_pointer.end()) {
        return found->second;
    }
    if (depth > kMaxSchemaDepth) {
        throw std::invalid_argument("schema nesting exceeds the depth limit");
    }
    const Json& value = at(pointer);
    if (!value.is_object()) {
        throw std::invalid_argument("schema at '#" + pointer + "' must be an object");
    }

    // Reserve the id before descending, so a `$ref` that points back at this location resolves to
    // this node rather than compiling the location again.
    const SchemaNodeId id = reserve();
    by_pointer.emplace(pointer, id);

    SchemaNode& node = nodes[id];
    scan_unmodeled(value);

    if (value.contains("$ref")) {
        const Json& reference = value.at("$ref");
        if (!reference.is_string()) { throw std::invalid_argument("$ref must be a string"); }
        const std::string target = reference.get<std::string>();
        if (target.empty() || target[0] != '#') {
            throw std::invalid_argument("only same-document $ref is supported, got '" + target + "'");
        }
        // The fragment is a JSON Pointer; `#` alone names the document root.
        const std::string fragment = target.substr(1);
        // A reference with siblings applies both, which is exactly allOf's meaning.
        node.all_of.push_back(compile_at(fragment, depth + 1));
    }

    if (value.contains("type")) {
        const Json& type = value.at("type");
        if (type.is_string()) {
            node.kinds.push_back(parse_kind(type.get<std::string>()));
        } else if (type.is_array()) {
            if (type.empty()) { throw std::invalid_argument("type array must not be empty"); }
            for (const Json& entry : type) {
                if (!entry.is_string()) {
                    throw std::invalid_argument("type array entries must be strings");
                }
                node.kinds.push_back(parse_kind(entry.get<std::string>()));
            }
        } else {
            throw std::invalid_argument("type must be a string or an array of strings");
        }
    }

    parse_string_keywords(value, node);
    parse_number_keywords(value, node);
    parse_object_keywords(pointer, value, node, depth);
    parse_array_keywords(pointer, value, node, depth);
    parse_combinators(pointer, value, node, depth);
    parse_enumeration(value, node, depth);
    return id;
}

JsonSchema::JsonSchema(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

JsonSchema::JsonSchema(JsonSchema&&) noexcept            = default;
JsonSchema& JsonSchema::operator=(JsonSchema&&) noexcept = default;
JsonSchema::~JsonSchema()                                = default;

JsonSchema JsonSchema::compile(std::string_view document) {
    auto impl = std::make_unique<Impl>();
    try {
        impl->document = Json::parse(document);
    } catch (const nlohmann::json::parse_error& error) {
        throw std::invalid_argument(std::string("schema is not valid JSON: ") + error.what());
    }
    if (!impl->document.is_object()) {
        throw std::invalid_argument("schema document must be a JSON object");
    }
    impl->root = impl->compile_at("", 0);
    return JsonSchema(std::move(impl));
}

SchemaNodeId JsonSchema::root() const noexcept { return impl_->root; }

const SchemaNode& JsonSchema::node(SchemaNodeId id) const {
    if (id >= impl_->nodes.size()) { throw std::out_of_range("schema node id is out of range"); }
    return impl_->nodes[id];
}

std::size_t JsonSchema::node_count() const noexcept { return impl_->nodes.size(); }

const std::vector<std::string>& JsonSchema::unsupported_keywords() const noexcept {
    return impl_->unsupported;
}

} // namespace ninfer::text
