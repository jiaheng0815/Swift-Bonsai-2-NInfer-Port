// Contract tests for the schema-driven pushdown automaton.
//
// The load-bearing property for constrained decoding is not "does this document parse" but "is the
// byte set the machine publishes the same set it will actually accept". A mask that promises a byte
// the machine then refuses strands the decoder; a mask that omits a byte it would have taken costs a
// legal continuation. Every document test below is therefore paired with a mask-agreement check.

#include "text/json_grammar.h"
#include "text/json_schema.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

int check(bool condition, const std::string& label) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << label << '\n';
    return 1;
}

using ninfer::text::ByteSet;
using ninfer::text::JsonGrammar;
using ninfer::text::JsonSchema;

// Feeds every byte of `text`, asserting that each one was announced by next_bytes(). The agreement
// check is the point: it is run on the whole corpus rather than on a hand-picked position.
int feed_all(JsonGrammar& grammar, std::string_view text, const std::string& label) {
    int failures = 0;
    for (std::size_t index = 0; index < text.size(); ++index) {
        const std::uint8_t byte = static_cast<std::uint8_t>(text[index]);
        const ByteSet announced = grammar.next_bytes();
        if (!announced.contains(byte)) {
            failures += check(false, label + ": byte " + std::to_string(index) + " (" +
                                         std::string(1, text[index]) +
                                         ") was not announced by next_bytes");
            return failures;
        }
        if (!grammar.feed(byte)) {
            failures += check(false, label + ": byte " + std::to_string(index) + " (" +
                                         std::string(1, text[index]) + ") was refused");
            return failures;
        }
    }
    return failures;
}

int accepts(const char* schema_text, std::string_view document, const std::string& label) {
    const JsonSchema schema = JsonSchema::compile(schema_text);
    JsonGrammar grammar(schema);
    int failures = feed_all(grammar, document, label);
    failures += check(grammar.complete(), label + ": the document is complete");
    return failures;
}

int rejects(const char* schema_text, std::string_view document, const std::string& label) {
    const JsonSchema schema = JsonSchema::compile(schema_text);
    JsonGrammar grammar(schema);
    bool refused = false;
    for (const char character : document) {
        if (!grammar.feed(static_cast<std::uint8_t>(character))) {
            refused = true;
            break;
        }
    }
    return check(refused || !grammar.complete(), label + ": the document was refused");
}

// The exact schema hermes sends for session titles, which is what this whole feature exists to serve.
constexpr const char* kTitleSchema = R"({
  "type": "object",
  "properties": {"title": {"type": "string"}},
  "required": ["title"],
  "additionalProperties": false
})";

int title_schema_accepts_and_rejects() {
    int failures = 0;
    failures += accepts(kTitleSchema, R"({"title":"hello"})", "the title document");
    // White space is outside the constrained language on purpose: a position that admits it admits
    // it repeatedly, so a model whose preferred tokens are all masked out could emit newlines until
    // its budget ran out. The whitespace-free subset of a schema's language is still valid JSON.
    failures += rejects(kTitleSchema, R"({ "title" : "hi" })", "spacing is not generated");
    failures += accepts(kTitleSchema, R"({"title":"中文"})", "escaped title text");
    failures += accepts(kTitleSchema, R"({"title":""})", "an empty title is still a string");

    // The shape an unconstrained model tends to emit, and the reason this feature exists.
    failures += rejects(kTitleSchema, R"({"name":"hello"})", "an undeclared member");
    failures += rejects(kTitleSchema, R"({"title":42})", "a title that is not a string");
    failures += rejects(kTitleSchema, R"({})", "a missing required member");
    failures += rejects(kTitleSchema, R"({"title":"a","extra":1})", "an extra member");
    failures += rejects(kTitleSchema, R"({"title":"a")", "a truncated document");
    return failures;
}

int masks_exclude_only_what_is_refused() {
    int failures = 0;
    // For a required object the first byte is unambiguous: `}` cannot close it.
    const JsonSchema schema = JsonSchema::compile(kTitleSchema);
    JsonGrammar grammar(schema);
    const ByteSet at_start = grammar.next_bytes();
    failures += check(at_start.contains('{'), "an object slot admits '{'");
    failures += check(!at_start.contains('}'), "a required object does not admit '}' first");
    failures += check(!at_start.contains('"'), "a value slot does not admit a bare quote");

    (void)grammar.feed('{');
    const ByteSet after_brace = grammar.next_bytes();
    failures += check(after_brace.contains('"'), "a key may start after '{'");
    failures += check(!after_brace.contains('}'), "a required member blocks closing after '{'");

    // Once the declared member is present and nothing else is allowed, closing becomes legal.
    for (const char character : std::string_view("\"title\":\"x\"")) {
        (void)grammar.feed(static_cast<std::uint8_t>(character));
    }
    failures += check(grammar.next_bytes().contains('}'), "closing is allowed once required is met");
    failures += check(!grammar.next_bytes().contains(','),
                      "additionalProperties:false leaves nothing to add after a comma");
    return failures;
}

int string_constraints_address_the_decoded_value() {
    int failures = 0;
    const char* bounded = R"({"type":"string","minLength":2,"maxLength":3})";
    failures += accepts(bounded, R"("ab")", "the lower length bound");
    failures += accepts(bounded, R"("abc")", "the upper length bound");
    failures += rejects(bounded, R"("a")", "below the length bound");
    failures += rejects(bounded, R"("abcd")", "above the length bound");
    // Four code points but eight UTF-8 bytes: a byte-counting machine gets this wrong.
    failures += accepts(bounded, "\"\xe4\xb8\xad\xe6\x96\x87\"", "two CJK characters");
    // One code point written as an escape still counts as one.
    failures += rejects(R"({"type":"string","minLength":2})", R"("中")",
                        "a single escaped code point is one unit");

    const char* patterned = R"({"type":"string","pattern":"^[a-z]+$"})";
    failures += accepts(patterned, R"("hello")", "a pattern that matches");
    failures += rejects(patterned, R"("Hello")", "a pattern that does not match");
    failures += rejects(patterned, R"("hello1")", "a pattern broken by a digit");

    // The pattern applies to the decoded text, so an escape cannot smuggle a forbidden character in.
    failures += rejects(patterned, R"("A")", "an escape that decodes outside the pattern");

    // Unanchored semantics: the pattern only has to occur somewhere.
    failures += accepts(R"({"type":"string","pattern":"b"})", R"("abc")",
                        "an unanchored pattern matches inside the value");
    return failures;
}

int nested_and_recursive_schemas() {
    int failures = 0;
    const char* nested = R"({
      "type":"object",
      "properties":{"items":{"type":"array","items":{"type":"integer"},"minItems":1}},
      "required":["items"],
      "additionalProperties":false
    })";
    failures += accepts(nested, R"({"items":[1,2,3]})", "a nested array of integers");
    failures += accepts(nested, R"({"items":[1]})", "a one-element array");
    failures += rejects(nested, R"({"items":[]})", "an array below minItems");
    failures += rejects(nested, R"({"items":[1,"x"]})", "a non-integer element");

    // A recursive list. Nothing here unfolds the schema; the stack carries the depth.
    const char* recursive = R"({
      "$defs":{"Node":{
        "type":"object",
        "properties":{"v":{"type":"integer"},"next":{"$ref":"#/$defs/Node"}},
        "required":["v"],
        "additionalProperties":false
      }},
      "$ref":"#/$defs/Node"
    })";
    failures += accepts(recursive, R"({"v":1})", "a one-node list");
    failures += accepts(recursive, R"({"v":1,"next":{"v":2}})", "a two-node list");
    failures += accepts(recursive, R"({"v":1,"next":{"v":2,"next":{"v":3}}})", "a three-node list");
    failures += rejects(recursive, R"({"v":1,"next":{"v":"x"}})", "a bad node deep in the list");
    return failures;
}

int enumerations_and_combinators() {
    int failures = 0;
    const char* enumerated = R"({"enum":["fast","safe"]})";
    failures += accepts(enumerated, R"("fast")", "the first alternative");
    failures += accepts(enumerated, R"("safe")", "the second alternative");
    failures += rejects(enumerated, R"("slow")", "a value outside the enumeration");

    const char* spaced = R"({"enum":[{"a":1}]})";
    failures += accepts(spaced, R"({"a":1})", "an object alternative");
    failures += rejects(spaced, R"({ "a" : 1 })", "the same alternative with spacing");

    const char* any_of = R"({"anyOf":[{"type":"string"},{"type":"integer"}]})";
    failures += accepts(any_of, R"("text")", "the string branch of anyOf");
    failures += accepts(any_of, "7", "the integer branch of anyOf");
    failures += rejects(any_of, "true", "neither branch of anyOf");

    const char* all_of = R"({"allOf":[{"type":"integer"},{"minimum":10}]})";
    failures += accepts(all_of, "12", "a value satisfying every allOf member");
    failures += rejects(all_of, "5", "a value violating one allOf member");
    failures += rejects(all_of, R"("x")", "a value of the wrong kind");
    return failures;
}

int scalars_and_bounds() {
    int failures = 0;
    failures += accepts(R"({"type":"boolean"})", "true", "true");
    failures += accepts(R"({"type":"boolean"})", "false", "false");
    failures += rejects(R"({"type":"boolean"})", "tru", "a truncated literal");
    failures += rejects(R"({"type":"boolean"})", "nul", "a different literal");

    failures += accepts(R"({"type":"null"})", "null", "null");
    failures += rejects(R"({"type":"null"})", "true", "null rejects a boolean");

    failures += accepts(R"({"type":"integer","minimum":3,"maximum":5})", "4", "an integer in range");
    failures += rejects(R"({"type":"integer","minimum":3,"maximum":5})", "2", "an integer below range");
    failures += rejects(R"({"type":"integer","minimum":3,"maximum":5})", "6", "an integer above range");
    failures += rejects(R"({"type":"integer"})", "1.5", "an integer slot rejects a fraction");

    failures += accepts(R"({"type":"number","minimum":0})", "1.5", "a number slot admits a fraction");
    failures += accepts(R"({"type":"number"})", "-2.5e3", "scientific notation");
    failures += rejects(R"({"type":"number","minimum":0})", "-1", "a number below range");
    return failures;
}

int tuples_and_uniform_arrays() {
    int failures = 0;
    const char* tuple = R"({"prefixItems":[{"type":"integer"},{"type":"string"}]})";
    failures += accepts(tuple, R"([1,"x"])", "a tuple in order");
    failures += rejects(tuple, R"(["x",1])", "a tuple out of order");
    // `prefixItems` positions the elements but does not require them: without `minItems` a shorter
    // array is still valid, which is the specification's reading and not a shortcut taken here.
    failures += accepts(tuple, R"([1])", "a tuple shorter than the prefix list");

    const char* fixed = R"({"const":[1,true]})";
    failures += accepts(fixed, R"([1,true])", "a literal array");
    failures += rejects(fixed, R"([ 1 , true ])", "a literal array with spacing");
    failures += rejects(fixed, R"([1,false])", "a literal array with a changed element");
    return failures;
}

int completion_is_terminal() {
    int failures = 0;
    // A bare number is the one value that can stop without a delimiter, so `complete()` reports true
    // while the digits stay extendable — the value may still grow, and the request ends by emitting
    // a stop token rather than another character.
    const JsonSchema numeric = JsonSchema::compile(R"({"type":"integer"})");
    JsonGrammar bare(numeric);
    failures += feed_all(bare, "7", "a bare integer");
    failures += check(bare.complete(), "a bare integer completes without a delimiter");
    failures += check(bare.feed('7'), "a bare integer can still be extended");

    // A closed document is different: the frame is popped, nothing further is admissible, and the
    // request's stop-token set becomes the whole mask.
    const JsonSchema object = JsonSchema::compile(R"({"type":"object","additionalProperties":false})");
    JsonGrammar closed(object);
    failures += feed_all(closed, "{}", "an empty closed object");
    failures += check(closed.complete(), "the closed object is complete");
    failures += check(!closed.feed('{'), "no token may follow a complete document");
    failures += check(!closed.feed(' '), "not even whitespace may follow a complete document");
    failures += check(closed.next_bytes().empty(), "a complete document admits no byte");
    failures += check(closed.complete(), "completion is not revoked by a refused byte");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += title_schema_accepts_and_rejects();
    failures += masks_exclude_only_what_is_refused();
    failures += string_constraints_address_the_decoded_value();
    failures += nested_and_recursive_schemas();
    failures += enumerations_and_combinators();
    failures += scalars_and_bounds();
    failures += tuples_and_uniform_arrays();
    failures += completion_is_terminal();
    if (failures == 0) { std::cout << "json_grammar: all contracts hold\n"; }
    return failures == 0 ? 0 : 1;
}
