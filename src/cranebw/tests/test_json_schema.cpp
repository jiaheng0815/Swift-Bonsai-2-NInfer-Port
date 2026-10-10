// Contract tests for the JSON Schema normalizer.
//
// The subject here is what a document *means*, so every expectation is written against the
// normalized node graph rather than against any downstream automaton.

#include "text/json_schema.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int check(bool condition, const std::string& label) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << label << '\n';
    return 1;
}

using ninfer::text::AdditionalProperties;
using ninfer::text::JsonKind;
using ninfer::text::JsonSchema;

// The shape hermes sends for session titles: the document that currently gets a 400.
constexpr const char* kTitleSchema = R"({
  "type": "object",
  "properties": {"title": {"type": "string"}},
  "required": ["title"],
  "additionalProperties": false
})";

int title_schema_normalizes() {
    int failures          = 0;
    const JsonSchema schema = JsonSchema::compile(kTitleSchema);
    const auto& root        = schema.node(schema.root());

    failures += check(root.kinds.size() == 1 && root.kinds[0] == JsonKind::Object,
                      "root admits exactly object");
    failures += check(root.properties.size() == 1 && root.properties[0].first == "title",
                      "one property named title");
    failures += check(root.required.size() == 1 && root.required[0] == "title", "title is required");
    failures += check(root.additional == AdditionalProperties::Denied,
                      "additionalProperties:false is normalized to Denied");
    failures += check(root.is_required("title"), "is_required reports the required member");

    const auto& title = schema.node(root.properties[0].second);
    failures += check(title.kinds.size() == 1 && title.kinds[0] == JsonKind::String,
                      "title admits exactly string");
    failures += check(schema.unsupported_keywords().empty(),
                      "the title schema stays inside the implemented subset");
    return failures;
}

int integer_is_a_subset_of_number() {
    int failures = 0;
    // A node naming both kinds must accept the integer spelling, which is the case a naive
    // "does the list contain this kind" test gets wrong.
    const JsonSchema schema = JsonSchema::compile(R"({"type": ["integer", "string"]})");
    const auto& root        = schema.node(schema.root());

    failures += check(root.admits(JsonKind::Integer), "integer is admitted by name");
    failures += check(root.admits(JsonKind::String), "string is admitted by name");
    failures += check(!root.admits(JsonKind::Number), "number alone is not admitted");
    failures += check(!root.admits(JsonKind::Object), "object is not admitted");

    const JsonSchema numeric = JsonSchema::compile(R"({"type": "number"})");
    failures += check(numeric.node(numeric.root()).admits(JsonKind::Integer),
                      "a number-only node admits the integer 1");
    return failures;
}

int absent_type_admits_everything() {
    int failures = 0;
    const JsonSchema schema = JsonSchema::compile(R"({})");
    const auto& root        = schema.node(schema.root());

    failures += check(root.kinds.empty(), "an absent type leaves the kind list empty");
    for (const JsonKind kind : {JsonKind::Null, JsonKind::Boolean, JsonKind::Object,
                                JsonKind::Array, JsonKind::Number, JsonKind::Integer,
                                JsonKind::String}) {
        failures += check(root.admits(kind),
                          std::string("an absent type admits ") + std::string(ninfer::text::json_kind_name(kind)));
    }
    return failures;
}

int recursion_compiles_to_a_finite_graph() {
    int failures = 0;
    // A linked list whose `next` points back at the definition. If the compiler unfolded this, it
    // would not terminate; the node count staying small is the evidence that it shares one node.
    const JsonSchema schema = JsonSchema::compile(R"({
      "$defs": {"Node": {
        "type": "object",
        "properties": {"value": {"type": "integer"}, "next": {"$ref": "#/$defs/Node"}},
        "required": ["value"],
        "additionalProperties": false
      }},
      "$ref": "#/$defs/Node"
    })");
    const auto& root = schema.node(schema.root());

    failures += check(root.all_of.size() == 1, "the root delegates through allOf to $defs/Node");
    const auto& node = schema.node(root.all_of[0]);
    failures += check(node.properties.size() == 2, "Node has value and next");
    failures += check(schema.node_count() <= 8,
                      "a recursive schema stays a small graph instead of unfolding");

    // `next` must resolve back to the same Node id, which is what makes the graph cyclic.
    std::uint32_t next_id = ninfer::text::kNoSchemaNode;
    for (const auto& entry : node.properties) {
        if (entry.first == "next") { next_id = entry.second; }
    }
    failures += check(next_id != ninfer::text::kNoSchemaNode, "next resolves");
    if (next_id != ninfer::text::kNoSchemaNode) {
        const auto& next = schema.node(next_id);
        failures += check(next.all_of.size() == 1 && next.all_of[0] == root.all_of[0],
                          "next points back at the same Node node");
    }
    return failures;
}

int enum_values_become_literal_nodes() {
    int failures = 0;
    const JsonSchema schema = JsonSchema::compile(R"({
      "type": "object",
      "properties": {"mode": {"enum": ["fast", "safe", 3, true, null]}},
      "required": ["mode"]
    })");
    const auto& root = schema.node(schema.root());
    const auto& mode = schema.node(root.properties[0].second);

    failures += check(mode.exact_values.size() == 5, "five literal alternatives");
    failures += check(schema.node(mode.exact_values[0]).exact_string == std::string("fast"),
                      "a string alternative keeps its decoded value");
    failures += check(schema.node(mode.exact_values[1]).exact_string == std::string("safe"),
                      "the second string alternative is distinct");
    failures += check(schema.node(mode.exact_values[2]).kinds.size() == 1 &&
                          schema.node(mode.exact_values[2]).kinds[0] == JsonKind::Integer,
                      "3 becomes an integer literal");
    failures += check(schema.node(mode.exact_values[2]).minimum == 3.0 &&
                          schema.node(mode.exact_values[2]).maximum == 3.0,
                      "the integer literal pins minimum and maximum");
    failures += check(schema.node(mode.exact_values[3]).exact_boolean == true,
                      "true becomes a boolean literal carrying its value");
    failures += check(schema.node(mode.exact_values[4]).kinds.size() == 1 &&
                          schema.node(mode.exact_values[4]).kinds[0] == JsonKind::Null,
                      "null becomes a null literal");
    return failures;
}

int object_literals_require_every_member() {
    int failures = 0;
    const JsonSchema schema = JsonSchema::compile(R"({"const": {"a": 1, "b": [true]}})");
    const auto& root        = schema.node(schema.root());

    failures += check(root.exact_values.size() == 1, "one const alternative");
    const auto& literal = schema.node(root.exact_values[0]);
    failures += check(literal.properties.size() == 2, "the literal object keeps both members");
    failures += check(literal.required.size() == 2, "every literal member is required");
    failures += check(literal.additional == AdditionalProperties::Denied,
                      "a literal object admits nothing beyond its members");

    const auto& array = schema.node(literal.properties[1].second);
    failures += check(array.prefix_items.size() == 1, "the nested array keeps one element schema");
    failures += check(array.min_items == 1 && array.max_items == 1,
                      "the nested array is pinned to length one");
    return failures;
}

int tuple_form_populates_prefix_items() {
    int failures = 0;
    // Draft-07 spells tuples as an array-valued `items`; prefixItems is the modern spelling. Both
    // must land in the same field so the automaton has one representation to reason about.
    const JsonSchema draft07 = JsonSchema::compile(R"({"items": [{"type": "integer"}, {"type": "string"}]})");
    const JsonSchema modern  = JsonSchema::compile(R"({"prefixItems": [{"type": "integer"}, {"type": "string"}]})");

    for (const JsonSchema* schema : {&draft07, &modern}) {
        const auto& root = schema->node(schema->root());
        failures += check(root.prefix_items.size() == 2, "two positional element schemas");
        if (root.prefix_items.size() == 2) {
            failures += check(schema->node(root.prefix_items[0]).kinds[0] == JsonKind::Integer,
                              "position 0 is the integer");
            failures += check(schema->node(root.prefix_items[1]).kinds[0] == JsonKind::String,
                              "position 1 is the string");
        }
    }
    return failures;
}

int unmodeled_keywords_are_reported() {
    int failures = 0;
    // `uniqueItems` narrows the language and is not implemented, so a caller promising a hard
    // guarantee has to be able to see that. Annotations must not trigger the same alarm.
    const JsonSchema constrained =
        JsonSchema::compile(R"({"type": "array", "uniqueItems": true})");
    const auto& reported = constrained.unsupported_keywords();
    failures += check(reported.size() == 1 && reported[0] == "uniqueItems",
                      "uniqueItems is reported as unmodeled");

    const JsonSchema annotated = JsonSchema::compile(
        R"({"title": "t", "description": "d", "$comment": "c", "$schema": "2020-12", "type": "string"})");
    failures += check(annotated.unsupported_keywords().empty(),
                      "annotations are not reported as unmodeled constraints");
    return failures;
}

int malformed_documents_are_rejected() {
    int failures = 0;
    const auto rejects = [&](const char* document, const char* label) {
        try {
            (void)JsonSchema::compile(document);
            failures += check(false, std::string(label) + " was accepted");
        } catch (const std::invalid_argument&) {
            // Rejected, which is the contract.
        }
    };
    rejects("[]", "an array document");
    rejects("{", "malformed JSON");
    rejects(R"({"type": "strng"})", "a misspelled type");
    rejects(R"({"type": []})", "an empty type array");
    rejects(R"({"required": "title"})", "a non-array required");
    rejects(R"({"enum": []})", "an empty enum");
    rejects(R"({"$ref": "#/nope"})", "an unresolvable reference");
    rejects(R"({"properties": {"a": 1}})", "a non-object property schema");
    rejects(R"({"items": 3})", "a non-schema items");
    rejects(R"({"anyOf": []})", "an empty anyOf");
    rejects(R"({"multipleOf": 0})", "a zero multipleOf");
    return failures;
}

int external_references_are_rejected() {
    int failures = 0;
    // Only same-document references are implemented; an external one must fail loudly rather than
    // silently admitting any value.
    try {
        (void)JsonSchema::compile(R"({"$ref": "https://example.com/schema.json"})");
        failures += check(false, "an external $ref was accepted");
    } catch (const std::invalid_argument&) {
        // Rejected, which is the contract.
    }
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += title_schema_normalizes();
    failures += integer_is_a_subset_of_number();
    failures += absent_type_admits_everything();
    failures += recursion_compiles_to_a_finite_graph();
    failures += enum_values_become_literal_nodes();
    failures += object_literals_require_every_member();
    failures += tuple_form_populates_prefix_items();
    failures += unmodeled_keywords_are_reported();
    failures += malformed_documents_are_rejected();
    failures += external_references_are_rejected();
    if (failures == 0) { std::cout << "json_schema: all contracts hold\n"; }
    return failures == 0 ? 0 : 1;
}
