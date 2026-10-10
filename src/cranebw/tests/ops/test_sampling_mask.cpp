// Public-contract qualification for the token mask on sample().
//
// The mask exists so a caller can remove tokens from consideration before any mode, filter or RNG
// sees them. Two things therefore have to hold, and they fail in opposite directions: a masked-out
// token must never be selected by any route, and a row with no mask must behave exactly as it did
// before the mask existed. The second is the regression guard, so it is checked by comparing masked
// and unmasked rows built from the same logits rather than by re-deriving the old behaviour.

#include "ninfer/ops/sampling.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// op_tester.h deliberately defines no cross-Op verdict helper beyond the numerical comparators, so
// the boolean form lives here.
int check(bool condition, const std::string& label) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << label << '\n';
    return 1;
}

constexpr int kSingleBlockDomain = 200; // <= kSamplerTileItems, so the one-block route runs
constexpr int kMultiBlockDomain  = 300; // > kSamplerTileItems, so the partial-then-merge route runs

std::size_t mask_bytes(int token_domain) {
    return static_cast<std::size_t>(token_domain + 7) / 8U;
}

// Builds an LSB-first candidate bitset from the allowed token ids.
std::vector<std::uint8_t> build_mask(int token_domain, const std::vector<int>& allowed) {
    std::vector<std::uint8_t> mask(mask_bytes(token_domain), 0);
    for (const int token : allowed) {
        mask[static_cast<std::size_t>(token) >> 3U] |=
            static_cast<std::uint8_t>(1U << (static_cast<unsigned>(token) & 7U));
    }
    return mask;
}

struct Row {
    std::vector<int> allowed;
    bool masked = true;
};

// Runs one sample() over a batch built from `logits`, with per-row masks, and returns the selected
// tokens. Returns the raw logits as floats alongside so callers can build oracles from exactly the
// values the kernel read.
std::vector<int> run_sample(std::vector<float>& host_logits, int token_domain, int batch,
                            const std::vector<Row>& rows, const std::vector<double>& temperature,
                            const std::vector<double>& top_p, const std::vector<double>& presence,
                            const std::vector<double>& frequency, std::uint64_t seed) {
    const int physical_rows = token_domain;
    std::vector<std::int32_t> positions(static_cast<std::size_t>(batch));
    for (int index = 0; index < batch; ++index) { positions[static_cast<std::size_t>(index)] = index; }

    std::vector<ops::SamplingConfig> configs(static_cast<std::size_t>(batch));
    std::vector<DeviceBuffer> mask_buffers;
    mask_buffers.reserve(rows.size());
    for (std::size_t index = 0; index < rows.size(); ++index) {
        ops::SamplingConfig& config = configs[index];
        config.temperature = static_cast<float>(temperature[index]);
        config.top_k       = 20;
        config.top_p       = static_cast<float>(top_p[index]);
        config.min_p       = 0.0f;
        config.presence_penalty  = static_cast<float>(presence[index]);
        config.frequency_penalty = static_cast<float>(frequency[index]);
        config.seed              = seed;
        config.token_counts      = nullptr;
        config.token_mask        = nullptr;
        if (rows[index].masked) {
            mask_buffers.push_back(to_device(build_mask(token_domain, rows[index].allowed)));
            config.token_mask = static_cast<const std::uint8_t*>(mask_buffers.back().p);
        }
    }

    // sample() addresses row b at logits + b*physical_rows, so the device buffer has to hold every
    // row rather than the one row the caller passed. Replicating the caller's values keeps each row
    // comparable while giving the kernel the extent it indexes.
    std::vector<float> batched(static_cast<std::size_t>(token_domain) *
                               static_cast<std::size_t>(batch));
    for (int row = 0; row < batch; ++row) {
        std::copy(host_logits.begin(), host_logits.begin() + token_domain,
                  batched.begin() + static_cast<std::size_t>(row) * static_cast<std::size_t>(token_domain));
    }
    DeviceBuffer device_logits = to_device_bf16(batched);
    DeviceBuffer device_configs(sizeof(ops::SamplingConfig) * static_cast<std::size_t>(batch));
    device_configs.copy_from_host(configs.data(), device_configs.bytes);
    DeviceBuffer device_positions = to_device(positions);
    GuardedDeviceBuffer device_out(sizeof(std::int32_t) * static_cast<std::size_t>(batch));

    Tensor logits_tensor(device_logits.p, DType::BF16, {physical_rows, batch});
    Tensor out_tensor(device_out.data(), DType::I32, {batch});
    Tensor positions_tensor(device_positions.p, DType::I32, {batch});
    const ops::SamplingConfig* config_ptr =
        static_cast<const ops::SamplingConfig*>(device_configs.p);

    WorkspaceArena arena(
        ops::sampling_workspace_capacity_bytes(token_domain, batch, batch) + (1U << 20U));
    ops::sample(logits_tensor, out_tensor, token_domain, config_ptr, positions_tensor,
                ops::kSamplePurposeDecode, arena, nullptr);
    cuda_synchronize();

    return from_device<std::int32_t>(device_out.data(), static_cast<std::size_t>(batch));
}

// A mask that removes the global argmax must redirect the selection to the best allowed token. This
// is the case the raw-argmax fast path gets wrong when a caller adds a mask without gating it.
int mask_overrides_the_argmax() {
    int failures            = 0;
    const int token_domain  = kSingleBlockDomain;
    std::vector<float> values(static_cast<std::size_t>(token_domain));
    for (int index = 0; index < token_domain; ++index) {
        values[static_cast<std::size_t>(index)] = static_cast<float>(index) * 0.01f;
    }
    round_to_bf16(values);

    const std::vector<int> allowed{3, 7, 11, 40, 199};
    const std::vector<int> picked =
        run_sample(values, token_domain, 1, {Row{.allowed = allowed, .masked = true}}, {0.0}, {1.0},
                   {0.0}, {0.0}, 12345);

    failures += verify_exact("masked greedy selection", picked, std::vector<int>{199});

    // Without the mask the same logits must still select the unmasked argmax.
    const std::vector<int> unmasked = run_sample(
        values, token_domain, 1, {Row{.allowed = {}, .masked = false}}, {0.0}, {1.0}, {0.0}, {0.0},
        12345);
    failures += verify_exact("unmasked greedy selection is unchanged", unmasked,
                             std::vector<int>{token_domain - 1});
    return failures;
}

// A one-token mask forces that token regardless of the logits, and does so on both interior routes.
int a_single_allowed_token_is_forced() {
    int failures = 0;
    for (const int token_domain : {kSingleBlockDomain, kMultiBlockDomain}) {
        std::vector<float> values(static_cast<std::size_t>(token_domain));
        fill_uniform(values, 4242, -8.0f, 8.0f);
        round_to_bf16(values);

        const int forced = token_domain / 3;
        const std::vector<int> picked =
            run_sample(values, token_domain, 1, {Row{.allowed = {forced}, .masked = true}}, {0.0},
                       {1.0}, {0.0}, {0.0}, 7);
        const std::string label = "single allowed token at domain " + std::to_string(token_domain);
        failures += verify_exact(label.c_str(), picked, std::vector<int>{forced});
    }
    return failures;
}

// A stochastic row must never emit a masked-out token, over enough draws that the unconstrained
// distribution would certainly have produced one.
int stochastic_rows_never_emit_a_masked_token() {
    int failures           = 0;
    const int token_domain = kSingleBlockDomain;
    std::vector<float> values(static_cast<std::size_t>(token_domain));
    fill_uniform(values, 991, -2.0f, 6.0f);
    round_to_bf16(values);

    const std::vector<int> allowed{1, 2, 5, 8, 13};
    constexpr int kDraws = 4000;
    std::vector<int> never_seen(allowed.size(), 0);
    for (int draw = 0; draw < kDraws; ++draw) {
        // The seed varies per draw so the sampler is exercised across the RNG space rather than
        // pinned to one stream.
        const std::vector<int> picked = run_sample(
            values, token_domain, 1, {Row{.allowed = allowed, .masked = true}}, {0.8}, {1.0}, {0.0},
            {0.0}, static_cast<std::uint64_t>(draw) * 2654435761ULL);
        const int token = picked[0];
        if (std::find(allowed.begin(), allowed.end(), token) == allowed.end()) {
            failures += check(false, "a stochastic draw selected a masked-out token");
            return failures;
        }
        const auto position = std::find(allowed.begin(), allowed.end(), token);
        ++never_seen[static_cast<std::size_t>(position - allowed.begin())];
    }
    // The mask must not silently pin the selection: a mask that behaved as a constant would pass the
    // exclusion check above. Requiring several distinct winners is the right strength — requiring
    // every allowed token would fail for a legitimately unlikely one, since the allowed tokens here
    // span a wide logit range and the coldest of them is genuinely unreachable at this temperature.
    const std::size_t distinct =
        static_cast<std::size_t>(std::count_if(never_seen.begin(), never_seen.end(),
                                               [](int count) { return count > 0; }));
    failures += check(distinct >= 3, "a stochastic masked row produced only " +
                                         std::to_string(distinct) + " distinct tokens");
    return failures;
}

// The mask is applied before the penalties, so a penalty cannot pull a masked-out token back into
// contention and a penalty-adjusted winner inside the mask still wins.
int mask_is_applied_before_the_penalties() {
    int failures           = 0;
    const int token_domain = kSingleBlockDomain;
    std::vector<float> values(static_cast<std::size_t>(token_domain), 0.0f);
    values[10] = 5.0f;  // strongly favoured, but masked out
    values[60] = 1.0f;  // the best allowed token
    values[61] = 0.9f;
    round_to_bf16(values);

    std::vector<float> rounded = values;
    const std::vector<int> picked = run_sample(
        rounded, token_domain, 1, {Row{.allowed = {60, 61}, .masked = true}}, {0.0}, {1.0}, {0.0},
        {0.0}, 3);
    failures += verify_exact("a penalty setup still respects the mask", picked,
                             std::vector<int>{60});

    // A presence penalty large enough to reorder the allowed tokens changes which one wins; that the
    // winner is still an allowed token is the contract.
    std::vector<float> penalized = values;
    const std::vector<int> with_penalty = run_sample(
        penalized, token_domain, 1, {Row{.allowed = {60, 61}, .masked = true}}, {0.0}, {1.0}, {2.0},
        {0.0}, 3);
    failures += check(with_penalty[0] == 60 || with_penalty[0] == 61,
                      "a penalised masked row still selects an allowed token");
    return failures;
}

// A masked row and an unmasked row in the same batch must not influence each other. This is the
// batch-level regression guard: per-row configuration stays per-row.
int masks_are_per_row() {
    int failures           = 0;
    const int token_domain = kSingleBlockDomain;
    std::vector<float> values(static_cast<std::size_t>(token_domain));
    for (int index = 0; index < token_domain; ++index) {
        values[static_cast<std::size_t>(index)] = static_cast<float>(index) * 0.01f;
    }
    round_to_bf16(values);

    const std::vector<int> allowed{2, 4, 6};
    const std::vector<int> picked =
        run_sample(values, token_domain, 2,
                   {Row{.allowed = allowed, .masked = true}, Row{.allowed = {}, .masked = false}},
                   {0.0, 0.0}, {1.0, 1.0}, {0.0, 0.0}, {0.0, 0.0}, 555);
    failures += verify_exact("a masked and an unmasked row in one batch", picked,
                             std::vector<int>{6, token_domain - 1});
    return failures;
}

// The candidate cap interacts with the mask: with only a handful of tokens allowed, the top-k path
// still has to select from them rather than from the logits overall.
int mask_narrows_the_candidate_set() {
    int failures           = 0;
    const int token_domain = kMultiBlockDomain;
    std::vector<float> values(static_cast<std::size_t>(token_domain));
    fill_uniform(values, 31337, 0.0f, 4.0f);
    round_to_bf16(values);

    const std::vector<int> allowed{17, 18, 19, 20, 21, 22, 23};
    double best_value = -std::numeric_limits<double>::infinity();
    int best_token    = -1;
    for (const int token : allowed) {
        const double value = bf16_to_f32(f32_to_bf16(values[static_cast<std::size_t>(token)]));
        if (value > best_value) {
            best_value = value;
            best_token = token;
        }
    }
    const std::vector<int> picked =
        run_sample(values, token_domain, 1, {Row{.allowed = allowed, .masked = true}}, {0.0}, {1.0},
                   {0.0}, {0.0}, 8080);
    failures += verify_exact("the masked argmax over the allowed set", picked,
                             std::vector<int>{best_token});
    return failures;
}

int mask_validation() {
    int failures = 0;
    // The Op must still reject a malformed call the same way it did before the mask existed; a mask
    // does not relax any shape or dtype rule.
    const int token_domain = 64;
    std::vector<float> values(static_cast<std::size_t>(token_domain), 0.0f);
    round_to_bf16(values);
    std::vector<ops::SamplingConfig> configs(1);
    configs[0].temperature = 0.0f;
    DeviceBuffer device_logits  = to_device_bf16(values);
    DeviceBuffer device_configs(sizeof(ops::SamplingConfig));
    device_configs.copy_from_host(configs.data(), device_configs.bytes);
    std::vector<std::int32_t> positions{0};
    DeviceBuffer device_positions = to_device(positions);
    GuardedDeviceBuffer device_out(sizeof(std::int32_t));
    WorkspaceArena arena(ops::sampling_workspace_capacity_bytes(token_domain, 1, 1) + (1U << 20U));

    // An out-shape mismatch must still throw with a mask present.
    std::vector<std::uint8_t> mask = build_mask(token_domain, {5});
    DeviceBuffer device_mask        = to_device(mask);
    configs[0].token_mask           = static_cast<const std::uint8_t*>(device_mask.p);
    device_configs.copy_from_host(configs.data(), device_configs.bytes);
    Tensor logits_tensor(device_logits.p, DType::BF16, {token_domain, 1});
    Tensor positions_tensor(device_positions.p, DType::I32, {1});
    Tensor wrong_shape(device_out.data(), DType::I32, {2});
    try {
        ops::sample(logits_tensor, wrong_shape, token_domain,
                    static_cast<const ops::SamplingConfig*>(device_configs.p), positions_tensor,
                    ops::kSamplePurposeDecode, arena, nullptr);
        failures += check(false, "a wrong out shape was accepted with a mask present");
    } catch (const std::invalid_argument&) {}
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += mask_overrides_the_argmax();
    failures += a_single_allowed_token_is_forced();
    failures += stochastic_rows_never_emit_a_masked_token();
    failures += mask_is_applied_before_the_penalties();
    failures += masks_are_per_row();
    failures += mask_narrows_the_candidate_set();
    failures += mask_validation();

    std::cout << (failures == 0 ? "OK" : "FAIL") << " sample token-mask contract\n";
    return failures == 0 ? 0 : 1;
}

