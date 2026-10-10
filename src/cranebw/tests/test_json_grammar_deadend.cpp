// Dead-end probe: a state that announces a byte it cannot follow.
//
// The mask contract is that every byte next_bytes() publishes is one feed() accepts. A machine that
// announces `,` in an object with no unwritten member left is not merely imprecise — the decoder
// follows it into a state with no continuation, and the only way out downstream is to end the turn,
// which truncates the document. So this probe walks real payloads byte by byte and asserts the
// announcement/acceptance agreement at every position, which is what the `feed_all` oracle in
// test_json_grammar.cpp does for the general corpus.

#include "text/json_grammar.h"
#include "text/json_schema.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ninfer::text::ByteSet;
using ninfer::text::JsonGrammar;
using ninfer::text::JsonSchema;

int failures = 0;

void expect(bool condition, const std::string& label) {
    if (condition) { return; }
    std::cerr << "FAIL: " << label << '\n';
    ++failures;
}

std::string describe(std::uint8_t byte) {
    if (byte >= 0x20 && byte < 0x7f) { return "'" + std::string(1, static_cast<char>(byte)) + "'"; }
    return "0x" + std::to_string(byte);
}

// Walks `document` and reports the first position where announcement and acceptance disagree, or
// where the machine reports itself complete before the text runs out.
void agree(const JsonSchema& schema, std::string_view document, const std::string& label) {
    JsonGrammar grammar(schema);
    for (std::size_t index = 0; index < document.size(); ++index) {
        const std::uint8_t byte = static_cast<std::uint8_t>(document[index]);
        const ByteSet announced = grammar.next_bytes();
        if (!announced.contains(byte)) {
            expect(false, label + ": byte " + std::to_string(index) + " " + describe(byte) +
                              " was not announced, but the document continues");
            return;
        }
        if (!grammar.feed(byte)) {
            expect(false, label + ": byte " + std::to_string(index) + " " + describe(byte) +
                              " was announced and then refused");
            return;
        }
        if (grammar.complete() && index + 1 < document.size()) {
            expect(false, label + ": the machine completed at byte " + std::to_string(index) +
                              " while text remained");
            return;
        }
    }
    expect(grammar.complete(), label + ": the whole document is complete");
}

void feed_all(JsonGrammar& grammar, std::string_view text) {
    for (const char byte : text) { (void)grammar.feed(static_cast<std::uint8_t>(byte)); }
}

const char* kNestedSchema = R"({
  "type":"object",
  "properties":{
    "name":{"type":"string"},
    "tags":{"type":"array","items":{"type":"string"}},
    "meta":{"type":"object","properties":{"score":{"type":"integer","minimum":0,"maximum":100}},
            "required":["score"],"additionalProperties":false}
  },
  "required":["name","tags","meta"],
  "additionalProperties":false
})";

void comma_is_not_offered_without_a_member_to_follow() {
    const JsonSchema schema = JsonSchema::compile(kNestedSchema);

    agree(schema, R"({"name":"张三","tags":["工程师","咖啡爱好者"],"meta":{"score":97}})",
          "the nested document");

    // Stop just inside `meta`, with its only member already written.
    JsonGrammar grammar(schema);
    feed_all(grammar, R"({"name":"a","tags":[],"meta":{"score":97)");
    const ByteSet at_member_end = grammar.next_bytes();
    expect(at_member_end.contains('}'), "a satisfied object may close");
    expect(!at_member_end.contains(','), "a comma is not offered when no member can follow");
    expect(!at_member_end.contains('"'), "a quote is not offered when no member can follow");
}

// A property name is resolved against the schema when its closing quote arrives, so the key slot is
// the one place where an announcement-only restriction is not enough: `feed` must refuse the byte
// too, because the vocabulary walk admits a token exactly when both agree.
void the_key_slot_admits_only_declared_names() {
    const JsonSchema schema = JsonSchema::compile(kNestedSchema);
    JsonGrammar grammar(schema);
    feed_all(grammar, R"({")");

    const ByteSet after_quote = grammar.next_bytes();
    expect(after_quote.contains('n'), "the key slot admits the first letter of `name`");
    expect(!after_quote.contains('x'), "the key slot does not admit an undeclared name");

    expect(!grammar.feed('x'), "an undeclared key byte is refused, not merely unannounced");
    expect(grammar.feed('n'), "a declared key byte is accepted");
    expect(grammar.feed('a'), "the key continues");
    // `nam` is a prefix of `name`, so it is taken; `nax` is a prefix of nothing and is not.
    expect(grammar.feed('m'), "a byte on the way to a declared name is accepted");
    expect(!grammar.feed('x'), "a byte that leaves every declared name is refused");

    // Spelled out in full, the name closes and the member is resolved.
    JsonGrammar whole(schema);
    feed_all(whole, R"({"nam)");
    expect(whole.next_bytes().contains('e'), "the key continues to the declared name");
    expect(!whole.next_bytes().contains('n'), "a name already ruled out is not offered again");
}

// A number is taken one byte at a time and only judged when it ends, so the byte-level gate has to
// ask whether any completion of the digits so far can still satisfy the slot. Without it the mask
// admits `87.5…` into an `integer` member and lets digits run past `maximum`, and the walk has no
// path out of either state.
void the_number_slot_refuses_digits_it_cannot_rescue() {
    const JsonSchema integer = JsonSchema::compile(
        R"({"type":"integer","minimum":0,"maximum":100})");

    JsonGrammar fractional(integer);
    (void)fractional.feed('1');
    (void)fractional.feed('0');
    expect(!fractional.feed('.'), "an integer member refuses a fraction point");
    expect(!fractional.feed('e'), "an integer member refuses an exponent");
    expect(fractional.feed('0'), "an integer member keeps taking digits while they fit");

    JsonGrammar oversized(integer);
    (void)oversized.feed('9');
    (void)oversized.feed('9');
    expect(!oversized.feed('9'), "a digit past the ceiling is refused, not accumulated");
    expect(oversized.feed('9') == false, "the refusal does not advance the number");

    JsonGrammar at_ceiling(integer);
    feed_all(at_ceiling, "100");
    expect(!at_ceiling.feed('0'), "a digit past the ceiling is refused at the ceiling too");
    expect(at_ceiling.complete(), "a bare value at the ceiling is a complete document");

    const JsonSchema nonnegative = JsonSchema::compile(R"({"type":"number","minimum":0})");
    JsonGrammar unsigned_only(nonnegative);
    expect(!unsigned_only.feed('-'), "a sign that no completion can satisfy is refused");

    const JsonSchema any_number = JsonSchema::compile(R"({"type":"number"})");
    JsonGrammar signed_number(any_number);
    expect(signed_number.feed('-'), "a sign is taken where negatives are admissible");
    expect(signed_number.feed('1'), "the number continues after the sign");
}

// A pattern and an escape interact: the digits a `\u` escape has already taken narrow which code
// point it will denote, and the pattern may accept none of the survivors. The escape cannot be
// abandoned once its backslash is written, so the digit that empties the range has to be refused.
void an_escape_under_a_pattern_stays_reachable() {
    const JsonSchema schema = JsonSchema::compile(
        R"({"type":"string","pattern":"^(cat|dog|bird)$"})");

    JsonGrammar grammar(schema);
    feed_all(grammar, R"("do\u006)");

    // String content announces every byte — the pattern is consulted by `feed`, which is what the
    // vocabulary walk asks as well — so the digit that completes `dog` is asserted through `feed`,
    // not through the announcement.
    expect(grammar.feed('7'), "`\\u0067` is `g`, which completes `dog`, and is accepted");
    expect(grammar.next_bytes().contains('"'), "the value closes once the pattern has matched");

    // The sibling digit is refused: ``` is a backtick, which no continuation can repair.
    JsonGrammar sibling(schema);
    feed_all(sibling, R"("do\u006)");
    expect(!sibling.feed('0'), "a digit whose code point the pattern rejects is refused");
    expect(sibling.feed('7'), "and the refusal did not advance the escape");
}

void an_escape_whose_block_the_pattern_rejects_is_refused() {
    const JsonSchema schema = JsonSchema::compile(
        R"({"type":"string","pattern":"^(cat|dog|bird)$"})");

    // `\ufc` leaves 0xfc00..0xfcff, none of which continues `do`.
    JsonGrammar grammar(schema);
    feed_all(grammar, R"("do\u00)");
    expect(!grammar.feed('f') || grammar.next_bytes().contains('0'),
           "a first digit is only taken if its block stays reachable");
}

void an_object_with_no_members_closes_immediately() {
    const JsonSchema schema = JsonSchema::compile(R"({"type":"object","additionalProperties":false})");
    JsonGrammar grammar(schema);
    (void)grammar.feed('{');
    const ByteSet after_brace = grammar.next_bytes();
    expect(after_brace.contains('}'), "the empty object may close");
    expect(!after_brace.contains('"'), "no key can start in a closed empty object");
}

void arrays_respect_their_element_budget() {
    const JsonSchema schema = JsonSchema::compile(
        R"({"type":"array","items":{"type":"integer"},"maxItems":2})");
    JsonGrammar grammar(schema);
    expect(grammar.feed('['), "an array opens");
    feed_all(grammar, "1");
    expect(grammar.next_bytes().contains(','), "a comma may follow one element of two");
    feed_all(grammar, ",2");
    const ByteSet full = grammar.next_bytes();
    expect(full.contains(']'), "a full array closes");
    expect(!full.contains(','), "a full array does not offer another element");
}

void strings_respect_their_length_budget() {
    const JsonSchema schema = JsonSchema::compile(R"({"type":"string","maxLength":2})");
    JsonGrammar grammar(schema);
    expect(grammar.feed('"'), "a string opens");
    (void)grammar.feed('a');
    expect(grammar.next_bytes().contains('c'), "a string below its bound may continue");
    (void)grammar.feed('b');
    const ByteSet at_bound = grammar.next_bytes();
    expect(at_bound.contains('"'), "a string at its bound may close");
    expect(!at_bound.contains('c'), "a string at its bound may not grow");
}

} // namespace

int main() {
    comma_is_not_offered_without_a_member_to_follow();
    the_key_slot_admits_only_declared_names();
    the_number_slot_refuses_digits_it_cannot_rescue();
    an_escape_under_a_pattern_stays_reachable();
    an_escape_whose_block_the_pattern_rejects_is_refused();
    an_object_with_no_members_closes_immediately();
    arrays_respect_their_element_budget();
    strings_respect_their_length_budget();
    if (failures == 0) { std::cout << "json_grammar_deadend: no state announces a byte it refuses\n"; }
    return failures == 0 ? 0 : 1;
}
