#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <chrono>
#include <array>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;
using namespace pybind11::literals;

namespace heat_native {

std::uint64_t fnv1a64(const py::bytes& payload) {
    const auto bytes = static_cast<std::string>(payload);
    std::uint64_t hash = 14695981039346656037ULL;
    for (const auto value : bytes) {
        hash ^= static_cast<std::uint8_t>(value);
        hash *= 1099511628211ULL;
    }
    return hash;
}

constexpr std::uint32_t kProtocolVersion = 1;
constexpr const char* kSchemaHash = "d3-a5-v3-r104-a516-p6-r200-c424-t90";
constexpr std::size_t kPlayers = 6;
constexpr std::size_t kZones = 6;
constexpr std::size_t kCardsPerZone = 424;
constexpr std::size_t kImportedCardsPerZone = 24;
constexpr std::size_t kTrackSpaces = 90;
constexpr std::size_t kCorners = 8;
constexpr std::size_t kSpinRecords = 200;
constexpr std::size_t kPlayerFields = 18;
constexpr std::size_t kControlFields = 9;
constexpr std::size_t kObservationDim = 104;
constexpr std::size_t kActionDim = 516;

template <typename T>
T narrow_integer(std::int64_t value, const char* label) {
    static_assert(std::is_integral_v<T>);
    if constexpr (std::is_unsigned_v<T>) {
        if (value < 0 || static_cast<std::uint64_t>(value) > std::numeric_limits<T>::max()) {
            throw py::value_error(std::string(label) + " is outside its native range");
        }
    } else if (value < std::numeric_limits<T>::min() || value > std::numeric_limits<T>::max()) {
        throw py::value_error(std::string(label) + " is outside its native range");
    }
    return static_cast<T>(value);
}

template <typename T>
py::array require_array(
    const py::dict& payload,
    const char* name,
    std::initializer_list<py::ssize_t> shape
) {
    if (!payload.contains(name)) {
        throw py::key_error(std::string("state payload requires ") + name);
    }
    auto array = py::cast<py::array>(payload[name]);
    if (!array.dtype().is(py::dtype::of<T>())) {
        throw py::type_error(std::string(name) + " has the wrong dtype");
    }
    if ((array.flags() & py::array::c_style) == 0) {
        throw py::value_error(std::string(name) + " must be C-contiguous");
    }
    if (array.ndim() != static_cast<py::ssize_t>(shape.size())) {
        throw py::value_error(std::string(name) + " has the wrong rank");
    }
    py::ssize_t axis = 0;
    for (const auto extent : shape) {
        if (array.shape(axis++) != extent) {
            throw py::value_error(std::string(name) + " has the wrong shape");
        }
    }
    return array;
}

struct CardToken {
    std::uint16_t identity = 0;
    std::int8_t value = 0;
    std::uint8_t type = 0;
};
static_assert(sizeof(CardToken) == 4);

struct CardZone {
    std::array<CardToken, kCardsPerZone> cards{};
    std::uint16_t length = 0;
};

struct SpinRecord {
    std::int16_t round = 0;
    std::int16_t corner_start = 0;
};

struct CompactPlayer {
    bool present = false;
    bool active = false;
    std::int16_t player_id = 0;
    std::int8_t gear = 0;
    std::int16_t position = 0;
    std::int16_t lap = 0;
    bool spun_out = false;
    bool finished = false;
    std::int16_t finish_order = 0;
    std::uint16_t spin_length = 0;
    bool boost_used = false;
    std::int16_t speed_cards = 0;
    std::int16_t speed_boost = 0;
    std::int16_t speed_adrenaline = 0;
    std::int16_t slipstream_moved = 0;
    bool cluttered = false;
    std::int16_t turn_start_position = 0;
    std::int16_t turn_start_lap = 0;
    std::array<SpinRecord, kSpinRecords> spins{};
    std::array<CardZone, kZones> zones{};
};

struct CompactTrack {
    std::uint8_t length = 0;
    std::array<std::int16_t, kTrackSpaces> indices{};
    std::array<std::uint8_t, kTrackSpaces> lanes{};
    std::uint8_t corner_count = 0;
    std::array<std::array<std::int16_t, 3>, kCorners> corners{};
    std::uint8_t start_count = 0;
    std::array<std::int16_t, kPlayers> starts{};
    std::uint8_t laps = 0;
};

class PythonMt19937 {
public:
    PythonMt19937() = default;

    explicit PythonMt19937(const py::tuple& state) {
        import_python_state(state);
    }

    void import_python_state(const py::tuple& state) {
        if (state.size() != 3) {
            throw py::value_error("Python RNG state must have three fields");
        }
        version_ = py::cast<std::uint8_t>(state[0]);
        if (version_ != 3) {
            throw py::value_error("only CPython RNG state version 3 is supported");
        }
        const auto words = py::cast<py::tuple>(state[1]);
        if (words.size() != 625) {
            throw py::value_error("Python RNG state must contain 624 words and an index");
        }
        for (std::size_t index = 0; index < 624; ++index) {
            words_[index] = py::cast<std::uint32_t>(words[index]);
        }
        index_ = py::cast<std::uint16_t>(words[624]);
        if (index_ > 624) {
            throw py::value_error("Python RNG index exceeds 624");
        }
        gauss_present_ = !state[2].is_none();
        gauss_ = gauss_present_ ? py::cast<double>(state[2]) : 0.0;
    }

    py::tuple export_python_state() const {
        py::tuple words(625);
        for (std::size_t index = 0; index < 624; ++index) {
            words[index] = words_[index];
        }
        words[624] = index_;
        return py::make_tuple(
            version_,
            words,
            gauss_present_ ? py::cast(gauss_) : py::none()
        );
    }

    std::uint32_t next_u32() {
        if (index_ >= 624) {
            twist();
        }
        std::uint32_t value = words_[index_++];
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c5680U;
        value ^= (value << 15) & 0xefc60000U;
        value ^= value >> 18;
        return value;
    }

    double random() {
        const auto high = next_u32() >> 5;
        const auto low = next_u32() >> 6;
        return (static_cast<double>(high) * 67108864.0 + low)
            * (1.0 / 9007199254740992.0);
    }

    py::int_ getrandbits(std::uint32_t bits) {
        if (bits == 0) {
            return py::int_(0);
        }
        py::object result = py::int_(0);
        const auto words_needed = (bits - 1) / 32 + 1;
        for (std::uint32_t word_index = 0; word_index < words_needed; ++word_index) {
            auto word = next_u32();
            const auto remaining = bits - word_index * 32;
            if (remaining < 32) {
                word >>= 32 - remaining;
            }
            py::object term = py::int_(word).attr("__lshift__")(word_index * 32);
            result = result.attr("__or__")(term);
        }
        return py::reinterpret_borrow<py::int_>(result);
    }

    std::uint64_t randbelow(std::uint64_t bound) {
        if (bound == 0) {
            throw py::value_error("randbelow bound must be positive");
        }
        std::uint32_t bits = 0;
        for (auto value = bound; value != 0; value >>= 1) {
            ++bits;
        }
        while (true) {
            std::uint64_t value;
            if (bits <= 32) {
                value = next_u32() >> (32 - bits);
            } else {
                const auto low = static_cast<std::uint64_t>(next_u32());
                auto high = static_cast<std::uint64_t>(next_u32());
                if (bits < 64) {
                    high >>= 64 - bits;
                }
                value = low | (high << 32);
            }
            if (value < bound) {
                return value;
            }
        }
    }

    py::list shuffled(std::uint32_t length) {
        py::list result;
        std::vector<std::uint32_t> values(length);
        for (std::uint32_t index = 0; index < length; ++index) {
            values[index] = index;
        }
        for (std::uint32_t remaining = length; remaining > 1; --remaining) {
            const auto selected = static_cast<std::size_t>(randbelow(remaining));
            std::swap(values[remaining - 1], values[selected]);
        }
        for (const auto value : values) {
            result.append(value);
        }
        return result;
    }

    void import_numeric(
        std::uint8_t version,
        const std::uint32_t* words,
        bool gauss_present,
        double gauss
    ) {
        if (version != 3 || words[624] > 624) {
            throw py::value_error("invalid CPython MT19937 numeric state");
        }
        version_ = version;
        std::copy(words, words + 624, words_.begin());
        index_ = static_cast<std::uint16_t>(words[624]);
        gauss_present_ = gauss_present;
        gauss_ = gauss;
    }

    void export_numeric(std::uint32_t* words, bool& gauss_present, double& gauss) const {
        std::copy(words_.begin(), words_.end(), words);
        words[624] = index_;
        gauss_present = gauss_present_;
        gauss = gauss_;
    }

private:
    void twist() {
        constexpr std::uint32_t matrix = 0x9908b0dfU;
        constexpr std::uint32_t upper = 0x80000000U;
        constexpr std::uint32_t lower = 0x7fffffffU;
        std::size_t index = 0;
        for (; index < 624 - 397; ++index) {
            const auto mixed = (words_[index] & upper) | (words_[index + 1] & lower);
            words_[index] = words_[index + 397] ^ (mixed >> 1)
                ^ ((mixed & 1U) ? matrix : 0U);
        }
        for (; index < 623; ++index) {
            const auto mixed = (words_[index] & upper) | (words_[index + 1] & lower);
            words_[index] = words_[index - (624 - 397)] ^ (mixed >> 1)
                ^ ((mixed & 1U) ? matrix : 0U);
        }
        const auto mixed = (words_[623] & upper) | (words_[0] & lower);
        words_[623] = words_[396] ^ (mixed >> 1) ^ ((mixed & 1U) ? matrix : 0U);
        index_ = 0;
    }

    std::uint8_t version_ = 3;
    std::array<std::uint32_t, 624> words_{};
    std::uint16_t index_ = 624;
    bool gauss_present_ = false;
    double gauss_ = 0.0;
};

struct CompactState {
    std::uint64_t game_id = 0;
    bool game_active = false;
    std::uint16_t round = 0;
    std::uint8_t phase = 0;
    std::uint8_t turn_order_length = 0;
    std::array<std::int16_t, kPlayers> turn_order{};
    std::uint8_t starting_player_count = 0;
    std::uint16_t stress_counter = 0;
    std::uint8_t pending_kind = 0;
    std::int16_t pending_player = -1;
    std::uint8_t turn_cursor = 0;
    std::uint64_t decision_epoch = 0;
    CompactTrack track;
    std::array<CompactPlayer, kPlayers> players{};
    PythonMt19937 rng;
};

class NativeState {
public:
    explicit NativeState(const py::dict& payload) {
        import_payload(payload);
    }

    NativeState(const NativeState&) = default;

    std::unique_ptr<NativeState> clone() const {
        return std::make_unique<NativeState>(*this);
    }

    std::uint64_t game_id() const noexcept {
        return state_.game_id;
    }

    std::uint16_t round_num() const noexcept {
        return state_.round;
    }

    py::tuple reward_state(std::int16_t player_id) const {
        const auto& player = state_.players[player_slot(player_id)];
        return py::make_tuple(player.position, player.lap, player.spun_out);
    }

    py::dict export_payload() const {
        py::dict payload;
        py::array_t<std::int64_t> control(kControlFields);
        auto* control_data = control.mutable_data();
        control_data[0] = static_cast<std::int64_t>(state_.game_id);
        control_data[1] = state_.game_active;
        control_data[2] = state_.round;
        control_data[3] = state_.phase;
        control_data[4] = state_.turn_order_length;
        control_data[5] = state_.starting_player_count;
        control_data[6] = state_.stress_counter;
        control_data[7] = 3;

        std::array<std::uint32_t, 625> rng_words{};
        bool gauss_present = false;
        double gauss = 0.0;
        state_.rng.export_numeric(rng_words.data(), gauss_present, gauss);
        control_data[8] = gauss_present;
        payload["control"] = std::move(control);

        py::array_t<std::int64_t> turn_order(kPlayers);
        auto* turn_data = turn_order.mutable_data();
        for (std::size_t index = 0; index < kPlayers; ++index) {
            turn_data[index] = state_.turn_order[index];
        }
        payload["turn_order"] = std::move(turn_order);

        py::array_t<std::int64_t> track_control(4);
        auto* track_control_data = track_control.mutable_data();
        track_control_data[0] = state_.track.length;
        track_control_data[1] = state_.track.corner_count;
        track_control_data[2] = state_.track.start_count;
        track_control_data[3] = state_.track.laps;
        payload["track_control"] = std::move(track_control);

        py::array_t<std::int64_t> track_indices(kTrackSpaces);
        py::array_t<std::int64_t> track_lanes(kTrackSpaces);
        for (std::size_t index = 0; index < kTrackSpaces; ++index) {
            track_indices.mutable_data()[index] = state_.track.indices[index];
            track_lanes.mutable_data()[index] = state_.track.lanes[index];
        }
        payload["track_indices"] = std::move(track_indices);
        payload["track_lanes"] = std::move(track_lanes);

        py::array_t<std::int64_t> corners({kCorners, std::size_t{3}});
        for (std::size_t corner = 0; corner < kCorners; ++corner) {
            for (std::size_t field = 0; field < 3; ++field) {
                corners.mutable_at(corner, field) = state_.track.corners[corner][field];
            }
        }
        payload["track_corners"] = std::move(corners);

        py::array_t<std::int64_t> starts(kPlayers);
        for (std::size_t index = 0; index < kPlayers; ++index) {
            starts.mutable_data()[index] = state_.track.starts[index];
        }
        payload["track_starts"] = std::move(starts);

        py::array_t<std::int64_t> players({kPlayers, kPlayerFields});
        py::array_t<std::int64_t> spins({kPlayers, kSpinRecords, std::size_t{2}});
        py::array_t<std::int64_t> card_ids({kZones, kPlayers, kCardsPerZone});
        py::array_t<std::int64_t> card_types({kZones, kPlayers, kCardsPerZone});
        py::array_t<std::int64_t> card_values({kZones, kPlayers, kCardsPerZone});
        py::array_t<std::int64_t> zone_lengths({kZones, kPlayers});
        for (std::size_t player_index = 0; player_index < kPlayers; ++player_index) {
            const auto& player = state_.players[player_index];
            const std::array<std::int64_t, kPlayerFields> fields = {
                player.present,
                player.active,
                player.player_id,
                player.gear,
                player.position,
                player.lap,
                player.spun_out,
                player.finished,
                player.finish_order,
                player.spin_length,
                player.boost_used,
                player.speed_cards,
                player.speed_boost,
                player.speed_adrenaline,
                player.slipstream_moved,
                player.cluttered,
                player.turn_start_position,
                player.turn_start_lap,
            };
            for (std::size_t field = 0; field < kPlayerFields; ++field) {
                players.mutable_at(player_index, field) = fields[field];
            }
            for (std::size_t spin = 0; spin < kSpinRecords; ++spin) {
                spins.mutable_at(player_index, spin, 0) = player.spins[spin].round;
                spins.mutable_at(player_index, spin, 1) = player.spins[spin].corner_start;
            }
            for (std::size_t zone = 0; zone < kZones; ++zone) {
                zone_lengths.mutable_at(zone, player_index) = player.zones[zone].length;
                for (std::size_t card = 0; card < kCardsPerZone; ++card) {
                    const auto& token = player.zones[zone].cards[card];
                    card_ids.mutable_at(zone, player_index, card) = token.identity;
                    card_types.mutable_at(zone, player_index, card) = token.type;
                    card_values.mutable_at(zone, player_index, card) = token.value;
                }
            }
        }
        payload["players"] = std::move(players);
        payload["spin_log"] = std::move(spins);
        payload["card_ids"] = std::move(card_ids);
        payload["card_types"] = std::move(card_types);
        payload["card_values"] = std::move(card_values);
        payload["zone_lengths"] = std::move(zone_lengths);

        py::array_t<std::uint32_t> exported_rng(625);
        std::copy(rng_words.begin(), rng_words.end(), exported_rng.mutable_data());
        payload["rng_words"] = std::move(exported_rng);
        py::array_t<double> rng_gauss(1);
        rng_gauss.mutable_data()[0] = gauss;
        payload["rng_gauss"] = std::move(rng_gauss);
        return payload;
    }

    py::dict receipt() const {
        py::dict result;
        result["state_schema_hash"] = kSchemaHash;
        result["compact_state_bytes"] = sizeof(CompactState);
        result["card_token_bytes"] = sizeof(CardToken);
        result["players_capacity"] = kPlayers;
        result["round_capacity"] = kSpinRecords;
        result["cards_per_zone_capacity"] = kCardsPerZone;
        result["track_space_capacity"] = kTrackSpaces;
        return result;
    }

    py::dict apply_cards_to_react(
        const py::array& selected_ids,
        const py::array& selected_lengths
    );

    py::dict start_round();

    py::dict apply_gears(
        const py::array& selected_gears,
        const py::array& heat_costs
    );

    py::dict apply_react(
        std::int16_t player_id,
        std::uint8_t cooldown_count,
        bool use_boost,
        bool use_adrenaline_speed,
        bool use_adrenaline_cooldown
    );

    py::dict apply_slipstream(std::int16_t player_id, bool take);

    py::dict apply_discard(
        std::int16_t player_id,
        const py::array& card_ids,
        std::uint8_t count
    );
    py::dict apply_gear_actions(const py::array& flat_actions);
    py::dict apply_card_actions(const py::array& flat_actions);
    py::dict apply_flat_action(std::int16_t player_id, std::uint16_t flat_action);

    py::array_t<float> observation(std::int16_t player_id, std::uint8_t kind) const;
    py::array_t<bool> legal_mask(std::int16_t player_id, std::uint8_t kind) const;
    py::dict bootstrap_row(std::int16_t player_id) const;
    double reward(
        std::int16_t player_id,
        std::int16_t previous_position,
        std::int16_t previous_lap,
        bool previous_spun_out,
        bool done,
        bool terminated,
        bool solo_mode,
        double shaping_weight,
        double spinout_weight
    ) const;
    double terminal_margin(std::int16_t player_id) const;
    std::uint64_t canonical_digest() const;
    py::array_t<std::uint64_t> boundary_receipt() const;

private:
    void import_payload(const py::dict& payload) {
        const auto control = require_array<std::int64_t>(
            payload, "control", {kControlFields}
        );
        const auto* control_data = static_cast<const std::int64_t*>(control.data());
        if (control_data[0] < 0) {
            throw py::value_error("game_id must be non-negative");
        }
        state_.game_id = static_cast<std::uint64_t>(control_data[0]);
        state_.game_active = control_data[1] != 0;
        state_.round = narrow_integer<std::uint16_t>(control_data[2], "round");
        state_.phase = narrow_integer<std::uint8_t>(control_data[3], "phase");
        state_.turn_order_length = narrow_integer<std::uint8_t>(
            control_data[4], "turn order length"
        );
        state_.starting_player_count = narrow_integer<std::uint8_t>(
            control_data[5], "starting player count"
        );
        state_.stress_counter = narrow_integer<std::uint16_t>(
            control_data[6], "stress counter"
        );
        if (state_.turn_order_length > kPlayers) {
            throw py::value_error("turn order exceeds fixed capacity");
        }

        const auto turn_order = require_array<std::int64_t>(
            payload, "turn_order", {kPlayers}
        );
        const auto* turn_data = static_cast<const std::int64_t*>(turn_order.data());
        for (std::size_t index = 0; index < kPlayers; ++index) {
            state_.turn_order[index] = narrow_integer<std::int16_t>(
                turn_data[index], "turn order value"
            );
        }

        const auto track_control = require_array<std::int64_t>(
            payload, "track_control", {4}
        );
        const auto* track_values = static_cast<const std::int64_t*>(track_control.data());
        state_.track.length = narrow_integer<std::uint8_t>(track_values[0], "track length");
        state_.track.corner_count = narrow_integer<std::uint8_t>(
            track_values[1], "corner count"
        );
        state_.track.start_count = narrow_integer<std::uint8_t>(
            track_values[2], "start count"
        );
        state_.track.laps = narrow_integer<std::uint8_t>(track_values[3], "track laps");
        if (state_.track.length > kTrackSpaces || state_.track.corner_count > kCorners
            || state_.track.start_count > kPlayers) {
            throw py::value_error("track descriptor exceeds fixed capacity");
        }

        const auto indices = require_array<std::int64_t>(
            payload, "track_indices", {kTrackSpaces}
        );
        const auto lanes = require_array<std::int64_t>(
            payload, "track_lanes", {kTrackSpaces}
        );
        const auto* index_data = static_cast<const std::int64_t*>(indices.data());
        const auto* lane_data = static_cast<const std::int64_t*>(lanes.data());
        for (std::size_t index = 0; index < kTrackSpaces; ++index) {
            state_.track.indices[index] = narrow_integer<std::int16_t>(
                index_data[index], "track space index"
            );
            state_.track.lanes[index] = narrow_integer<std::uint8_t>(
                lane_data[index], "track lane count"
            );
        }
        const auto corners = require_array<std::int64_t>(
            payload, "track_corners", {kCorners, 3}
        );
        const auto* corner_data = static_cast<const std::int64_t*>(corners.data());
        for (std::size_t corner = 0; corner < kCorners; ++corner) {
            for (std::size_t field = 0; field < 3; ++field) {
                state_.track.corners[corner][field] = narrow_integer<std::int16_t>(
                    corner_data[corner * 3 + field], "track corner value"
                );
            }
        }
        const auto starts = require_array<std::int64_t>(
            payload, "track_starts", {kPlayers}
        );
        const auto* start_data = static_cast<const std::int64_t*>(starts.data());
        for (std::size_t index = 0; index < kPlayers; ++index) {
            state_.track.starts[index] = narrow_integer<std::int16_t>(
                start_data[index], "track start"
            );
        }

        import_players(payload);
        const auto rng_words = require_array<std::uint32_t>(
            payload, "rng_words", {625}
        );
        const auto rng_gauss = require_array<double>(payload, "rng_gauss", {1});
        state_.rng.import_numeric(
            narrow_integer<std::uint8_t>(control_data[7], "RNG version"),
            static_cast<const std::uint32_t*>(rng_words.data()),
            control_data[8] != 0,
            *static_cast<const double*>(rng_gauss.data())
        );
    }

    void import_players(const py::dict& payload) {
        const auto players = require_array<std::int64_t>(
            payload, "players", {kPlayers, kPlayerFields}
        );
        const auto spins = require_array<std::int64_t>(
            payload, "spin_log", {kPlayers, kSpinRecords, 2}
        );
        const auto card_ids = require_array<std::int64_t>(
            payload, "card_ids", {kZones, kPlayers, kImportedCardsPerZone}
        );
        const auto card_types = require_array<std::int64_t>(
            payload, "card_types", {kZones, kPlayers, kImportedCardsPerZone}
        );
        const auto card_values = require_array<std::int64_t>(
            payload, "card_values", {kZones, kPlayers, kImportedCardsPerZone}
        );
        const auto zone_lengths = require_array<std::int64_t>(
            payload, "zone_lengths", {kZones, kPlayers}
        );
        const auto* player_data = static_cast<const std::int64_t*>(players.data());
        const auto* spin_data = static_cast<const std::int64_t*>(spins.data());
        const auto* id_data = static_cast<const std::int64_t*>(card_ids.data());
        const auto* type_data = static_cast<const std::int64_t*>(card_types.data());
        const auto* value_data = static_cast<const std::int64_t*>(card_values.data());
        const auto* length_data = static_cast<const std::int64_t*>(zone_lengths.data());
        for (std::size_t index = 0; index < kPlayers; ++index) {
            const auto* fields = player_data + index * kPlayerFields;
            auto& player = state_.players[index];
            player.present = fields[0] != 0;
            player.active = fields[1] != 0;
            player.player_id = narrow_integer<std::int16_t>(fields[2], "player id");
            player.gear = narrow_integer<std::int8_t>(fields[3], "gear");
            player.position = narrow_integer<std::int16_t>(fields[4], "position");
            player.lap = narrow_integer<std::int16_t>(fields[5], "lap");
            player.spun_out = fields[6] != 0;
            player.finished = fields[7] != 0;
            player.finish_order = narrow_integer<std::int16_t>(fields[8], "finish order");
            player.spin_length = narrow_integer<std::uint16_t>(fields[9], "spin length");
            if (player.spin_length > kSpinRecords) {
                throw py::value_error("spin log exceeds fixed capacity");
            }
            player.boost_used = fields[10] != 0;
            player.speed_cards = narrow_integer<std::int16_t>(fields[11], "card speed");
            player.speed_boost = narrow_integer<std::int16_t>(fields[12], "boost speed");
            player.speed_adrenaline = narrow_integer<std::int16_t>(fields[13], "adrenaline speed");
            player.slipstream_moved = narrow_integer<std::int16_t>(fields[14], "slipstream movement");
            player.cluttered = fields[15] != 0;
            player.turn_start_position = narrow_integer<std::int16_t>(fields[16], "turn start position");
            player.turn_start_lap = narrow_integer<std::int16_t>(fields[17], "turn start lap");
            for (std::size_t spin = 0; spin < kSpinRecords; ++spin) {
                const auto base = (index * kSpinRecords + spin) * 2;
                player.spins[spin].round = narrow_integer<std::int16_t>(spin_data[base], "spin round");
                player.spins[spin].corner_start = narrow_integer<std::int16_t>(spin_data[base + 1], "spin corner");
            }
            for (std::size_t zone = 0; zone < kZones; ++zone) {
                auto& target_zone = player.zones[zone];
                target_zone.length = narrow_integer<std::uint16_t>(
                    length_data[zone * kPlayers + index], "card zone length"
                );
                if (target_zone.length > kCardsPerZone) {
                    throw py::value_error("card zone exceeds fixed capacity");
                }
                for (std::size_t card = 0; card < kImportedCardsPerZone; ++card) {
                    const auto base = (zone * kPlayers + index)
                        * kImportedCardsPerZone + card;
                    target_zone.cards[card].identity = narrow_integer<std::uint16_t>(id_data[base], "card identity");
                    target_zone.cards[card].type = narrow_integer<std::uint8_t>(type_data[base], "card type");
                    target_zone.cards[card].value = narrow_integer<std::int8_t>(value_data[base], "card value");
                }
            }
        }
    }

    static void append(CardZone& zone, CardToken card, const char* label);
    static CardToken remove_at(CardZone& zone, std::size_t index);
    static std::size_t find_identity(const CardZone& zone, std::uint16_t identity);
    CardToken draw(CompactPlayer& player);
    void replenish(CompactPlayer& player);
    void pay_heat(CompactPlayer& player, std::uint8_t amount);
    std::int16_t resolve_flip(CompactPlayer& player);
    std::int16_t resolve_blocked(const CompactPlayer& player, std::int32_t target) const;
    void credit_movement(CompactPlayer& player, std::int16_t amount);
    bool slipstream_eligible(const CompactPlayer& player) const;
    void check_corner(CompactPlayer& player);
    std::size_t player_slot(std::int16_t player_id) const;
    py::dict advance_to_react_after(std::size_t completed_turn_index);
    py::dict decision_result(
        const char* kind,
        const CompactPlayer& player,
        std::uint8_t turn_index
    );
    py::dict group_result(const char* kind) const;
    void expect_pending(std::uint8_t kind, std::int16_t player_id = -1) const;
    bool adrenaline_eligible(const CompactPlayer& player) const;
    static double clip01(double value);
    static double clip_signed(double value);
    static std::uint8_t action_token(const CardToken& card);
    static std::array<std::uint8_t, 8> card_requirements(std::size_t codec_index);
    void hash_bytes(std::uint64_t& hash, const void* data, std::size_t size) const;

    CompactState state_{};
};

void NativeState::append(CardZone& zone, CardToken card, const char* label) {
    if (zone.length >= kCardsPerZone) {
        throw py::value_error(std::string(label) + " exceeds fixed card capacity");
    }
    zone.cards[zone.length++] = card;
}

CardToken NativeState::remove_at(CardZone& zone, std::size_t index) {
    if (index >= zone.length) {
        throw py::index_error("card index is outside active zone");
    }
    const auto card = zone.cards[index];
    for (std::size_t next = index + 1; next < zone.length; ++next) {
        zone.cards[next - 1] = zone.cards[next];
    }
    zone.cards[--zone.length] = {};
    return card;
}

std::size_t NativeState::find_identity(const CardZone& zone, std::uint16_t identity) {
    for (std::size_t index = 0; index < zone.length; ++index) {
        if (zone.cards[index].identity == identity) {
            return index;
        }
    }
    throw py::value_error("selected card identity is not in the expected zone");
}

CardToken NativeState::draw(CompactPlayer& player) {
    auto& draw_zone = player.zones[1];
    auto& discard = player.zones[2];
    if (draw_zone.length == 0 && discard.length != 0) {
        draw_zone = discard;
        discard = {};
        for (std::size_t remaining = draw_zone.length; remaining > 1; --remaining) {
            const auto selected = static_cast<std::size_t>(state_.rng.randbelow(remaining));
            std::swap(draw_zone.cards[remaining - 1], draw_zone.cards[selected]);
        }
    }
    if (draw_zone.length == 0) {
        return {};
    }
    return remove_at(draw_zone, draw_zone.length - 1);
}

void NativeState::replenish(CompactPlayer& player) {
    state_.phase = 8;
    auto& played = player.zones[5];
    auto& discard = player.zones[2];
    while (played.length != 0) {
        append(discard, remove_at(played, 0), "discard pile");
    }
    auto& hand = player.zones[0];
    while (hand.length < 7) {
        const auto card = draw(player);
        if (card.identity == 0) {
            break;
        }
        append(hand, card, "hand");
    }
    player.boost_used = false;
    player.speed_cards = 0;
    player.speed_boost = 0;
    player.speed_adrenaline = 0;
    player.slipstream_moved = 0;
    player.cluttered = false;
    player.turn_start_position = 0;
}

void NativeState::pay_heat(CompactPlayer& player, std::uint8_t amount) {
    auto& heat = player.zones[3];
    auto& discard = player.zones[2];
    if (amount > heat.length) {
        throw py::value_error("player cannot pay requested heat");
    }
    for (std::uint8_t paid = 0; paid < amount; ++paid) {
        append(discard, remove_at(heat, 0), "discard pile");
    }
}

std::int16_t NativeState::resolve_flip(CompactPlayer& player) {
    while (true) {
        const auto card = draw(player);
        if (card.identity == 0) {
            return 0;
        }
        if (card.type == 1) {
            append(player.zones[5], card, "played cards");
            return card.value;
        }
        append(player.zones[2], card, "discard pile");
    }
}

std::int16_t NativeState::resolve_blocked(
    const CompactPlayer& player,
    std::int32_t target
) const {
    const auto length = static_cast<std::int32_t>(state_.track.length);
    auto position = static_cast<std::int16_t>((target % length + length) % length);
    for (std::int32_t offset = 0; offset < length; ++offset) {
        const auto candidate = static_cast<std::int16_t>(
            (position - offset + length) % length
        );
        std::uint8_t occupied = 0;
        for (const auto& other : state_.players) {
            if (other.present && !other.finished && other.player_id != player.player_id
                && other.position == candidate) {
                ++occupied;
            }
        }
        if (occupied < state_.track.lanes[candidate]) {
            return candidate;
        }
    }
    return position;
}

void NativeState::credit_movement(CompactPlayer& player, std::int16_t amount) {
    const auto raw = static_cast<std::int32_t>(player.position) + amount;
    const auto crossings = raw / state_.track.length;
    const auto target = raw % state_.track.length;
    player.position = resolve_blocked(player, target);
    player.lap = static_cast<std::int16_t>(player.lap + crossings);
    if (player.lap > state_.track.laps) {
        player.finished = true;
        player.active = false;
        std::int16_t finished = 0;
        for (const auto& other : state_.players) {
            if (other.present && other.finished) {
                ++finished;
            }
        }
        player.finish_order = finished;
        state_.game_active = false;
        for (const auto& other : state_.players) {
            state_.game_active = state_.game_active || (other.present && other.active);
        }
    }
}

bool NativeState::slipstream_eligible(const CompactPlayer& player) const {
    if (player.finished) {
        return false;
    }
    if (player.lap >= state_.track.laps
        && player.position + 2 >= state_.track.length) {
        return false;
    }
    for (const auto& other : state_.players) {
        if (!other.present || other.finished || other.player_id == player.player_id) {
            continue;
        }
        const auto distance = (
            other.position - player.position + state_.track.length
        ) % state_.track.length;
        if (distance == 1 || distance == 2) {
            return true;
        }
    }
    return false;
}

void NativeState::check_corner(CompactPlayer& player) {
    state_.phase = 6;
    if (player.finished) {
        return;
    }
    const auto spaces_moved = (
        (player.lap - player.turn_start_lap) * state_.track.length
        + player.position - player.turn_start_position
    );
    if (spaces_moved <= 0) {
        return;
    }
    const auto speed = player.speed_cards + player.speed_boost + player.speed_adrenaline;
    std::int16_t heat_cost = 0;
    std::int16_t first_corner_start = -1;
    for (std::size_t corner_index = 0; corner_index < state_.track.corner_count; ++corner_index) {
        const auto& corner = state_.track.corners[corner_index];
        bool crossed = spaces_moved >= state_.track.length;
        if (!crossed) {
            for (std::int16_t step = 1; step <= spaces_moved; ++step) {
                const auto position = (
                    player.turn_start_position + step
                ) % state_.track.length;
                if (position >= corner[0] && position <= corner[1]) {
                    crossed = true;
                    break;
                }
            }
        }
        if (crossed) {
            if (first_corner_start < 0) {
                first_corner_start = corner[0];
            }
            heat_cost = static_cast<std::int16_t>(
                heat_cost + std::max<std::int16_t>(0, speed - corner[2])
            );
        }
    }
    if (heat_cost == 0) {
        return;
    }
    auto& heat = player.zones[3];
    if (heat_cost <= heat.length) {
        pay_heat(player, static_cast<std::uint8_t>(heat_cost));
        return;
    }
    const auto remaining_heat = heat.length;
    pay_heat(player, remaining_heat);
    player.position = std::max<std::int16_t>(0, first_corner_start - 1);
    const auto stress_count = player.gear <= 2 ? 1 : 2;
    for (std::int16_t index = 0; index < stress_count; ++index) {
        if (state_.stress_counter == 0x7fffU) {
            throw py::value_error("stress token range exhausted");
        }
        ++state_.stress_counter;
        append(
            player.zones[0],
            CardToken{
                static_cast<std::uint16_t>(0x8000U | state_.stress_counter),
                0,
                3,
            },
            "hand"
        );
    }
    player.gear = 1;
    player.spun_out = true;
    if (player.spin_length >= kSpinRecords) {
        throw py::value_error("spin log exceeds fixed capacity");
    }
    player.spins[player.spin_length++] = {static_cast<std::int16_t>(state_.round), first_corner_start};
}

std::size_t NativeState::player_slot(std::int16_t player_id) const {
    for (std::size_t index = 0; index < kPlayers; ++index) {
        if (state_.players[index].present && state_.players[index].player_id == player_id) {
            return index;
        }
    }
    throw py::key_error("unknown native player identity");
}

py::dict NativeState::decision_result(
    const char* kind,
    const CompactPlayer& player,
    std::uint8_t turn_index
) {
    const std::string kind_name(kind);
    if (kind_name == "react") {
        state_.pending_kind = 3;
    } else if (kind_name == "slipstream") {
        state_.pending_kind = 4;
    } else if (kind_name == "discard") {
        state_.pending_kind = 5;
    } else {
        throw py::value_error("unknown native decision kind");
    }
    state_.pending_player = player.player_id;
    state_.turn_cursor = turn_index;
    ++state_.decision_epoch;
    py::dict result;
    result["kind"] = kind;
    result["game_id"] = state_.game_id;
    result["player_id"] = player.player_id;
    result["turn_index"] = turn_index;
    result["phase"] = state_.phase;
    result["decision_epoch"] = state_.decision_epoch;
    return result;
}

py::dict NativeState::group_result(const char* kind) const {
    py::dict result;
    result["kind"] = kind;
    result["game_id"] = state_.game_id;
    result["decision_epoch"] = state_.decision_epoch;
    py::list player_ids;
    for (const auto& player : state_.players) {
        if (!player.present || player.finished) {
            continue;
        }
        if (std::string(kind) == "gear" && player.spun_out) {
            continue;
        }
        player_ids.append(player.player_id);
    }
    result["player_ids"] = std::move(player_ids);
    return result;
}

void NativeState::expect_pending(std::uint8_t kind, std::int16_t player_id) const {
    if (state_.pending_kind != kind) {
        throw py::value_error("submission does not match the pending native decision");
    }
    if (player_id >= 0 && state_.pending_player != player_id) {
        throw py::value_error("submission player does not match the pending decision");
    }
}

py::dict NativeState::start_round() {
    if (state_.pending_kind != 0 && state_.pending_kind != 6) {
        throw py::value_error("cannot start a round while a decision is pending");
    }
    std::vector<std::size_t> active;
    for (std::size_t slot = 0; slot < kPlayers; ++slot) {
        if (state_.players[slot].present && !state_.players[slot].finished) {
            active.push_back(slot);
        }
    }
    if (active.empty()) {
        state_.pending_kind = 7;
        py::dict complete;
        complete["kind"] = "game_complete";
        complete["game_id"] = state_.game_id;
        return complete;
    }
    std::sort(
        active.begin(),
        active.end(),
        [this](std::size_t left, std::size_t right) {
            const auto& lhs = state_.players[left];
            const auto& rhs = state_.players[right];
            if (lhs.position != rhs.position) {
                return lhs.position > rhs.position;
            }
            if (lhs.lap != rhs.lap) {
                return lhs.lap > rhs.lap;
            }
            return lhs.player_id < rhs.player_id;
        }
    );
    state_.turn_order = {};
    state_.turn_order_length = narrow_integer<std::uint8_t>(
        static_cast<std::int64_t>(active.size()), "active turn order"
    );
    for (std::size_t index = 0; index < active.size(); ++index) {
        state_.turn_order[index] = state_.players[active[index]].player_id;
    }
    state_.pending_kind = 1;
    state_.pending_player = -1;
    state_.turn_cursor = 0;
    ++state_.decision_epoch;
    return group_result("gear");
}

py::dict NativeState::apply_gears(
    const py::array& selected_gears,
    const py::array& heat_costs
) {
    expect_pending(1);
    const py::dict payload = py::dict(
        "selected_gears"_a=selected_gears,
        "heat_costs"_a=heat_costs
    );
    const auto gears = require_array<std::int64_t>(
        payload, "selected_gears", {kPlayers}
    );
    const auto costs = require_array<std::int64_t>(payload, "heat_costs", {kPlayers});
    const auto* gear_data = static_cast<const std::int64_t*>(gears.data());
    const auto* cost_data = static_cast<const std::int64_t*>(costs.data());
    state_.phase = 0;
    for (std::size_t slot = 0; slot < kPlayers; ++slot) {
        auto& player = state_.players[slot];
        if (!player.present || player.finished) {
            continue;
        }
        if (player.spun_out) {
            player.gear = 1;
            continue;
        }
        const auto new_gear = narrow_integer<std::int8_t>(
            gear_data[slot], "selected gear"
        );
        const auto heat_cost = narrow_integer<std::uint8_t>(
            cost_data[slot], "gear heat cost"
        );
        const auto delta = std::abs(
            static_cast<std::int16_t>(new_gear) - player.gear
        );
        const auto expected_cost = delta == 2 ? 1 : 0;
        if (new_gear < 1 || new_gear > 4 || delta > 2
            || heat_cost != expected_cost || heat_cost > player.zones[3].length) {
            throw py::value_error("illegal native gear submission");
        }
        player.gear = new_gear;
        if (heat_cost != 0) {
            pay_heat(player, heat_cost);
        }
    }
    state_.pending_kind = 2;
    state_.pending_player = -1;
    ++state_.decision_epoch;
    return group_result("cards");
}

py::dict NativeState::advance_to_react_after(std::size_t completed_turn_index) {
    for (std::size_t turn_index = completed_turn_index + 1;
         turn_index < state_.turn_order_length;
         ++turn_index) {
        auto& player = state_.players[player_slot(state_.turn_order[turn_index])];
        if (player.finished) {
            continue;
        }
        if (player.cluttered) {
            player.gear = 1;
            replenish(player);
            continue;
        }
        state_.phase = 2;
        player.turn_start_position = player.position;
        player.turn_start_lap = player.lap;
        std::int16_t speed = 0;
        const auto initial_played = player.zones[5].length;
        for (std::size_t card = 0; card < initial_played; ++card) {
            const auto token = player.zones[5].cards[card];
            if (token.type == 3) {
                speed = static_cast<std::int16_t>(speed + resolve_flip(player));
            } else {
                speed = static_cast<std::int16_t>(speed + token.value);
            }
        }
        player.speed_cards = speed;
        credit_movement(player, speed);
        if (player.finished) {
            replenish(player);
            continue;
        }
        state_.phase = 3;
        return decision_result("react", player, static_cast<std::uint8_t>(turn_index));
    }
    py::dict complete;
    for (auto& player : state_.players) {
        if (player.present && player.active) {
            player.spun_out = false;
        }
    }
    ++state_.round;
    state_.pending_kind = 6;
    state_.pending_player = -1;
    complete["kind"] = "round_complete";
    complete["game_id"] = state_.game_id;
    return complete;
}

py::dict NativeState::apply_cards_to_react(
    const py::array& selected_ids,
    const py::array& selected_lengths
) {
    if (state_.pending_kind != 0) {
        expect_pending(2);
    }
    const py::dict payload = py::dict(
        "ids"_a=selected_ids,
        "lengths"_a=selected_lengths
    );
    const auto ids = require_array<std::int64_t>(payload, "ids", {kPlayers, 4});
    const auto lengths = require_array<std::int64_t>(payload, "lengths", {kPlayers});
    const auto* id_data = static_cast<const std::int64_t*>(ids.data());
    const auto* length_data = static_cast<const std::int64_t*>(lengths.data());
    state_.phase = 1;
    for (std::size_t player_index = 0; player_index < kPlayers; ++player_index) {
        auto& player = state_.players[player_index];
        if (!player.present || player.finished) {
            continue;
        }
        const auto count = narrow_integer<std::uint8_t>(
            length_data[player_index], "selected card count"
        );
        if (count > 4) {
            throw py::value_error("selected cards exceed gear action capacity");
        }
        std::size_t playable = 0;
        for (std::size_t card = 0; card < player.zones[0].length; ++card) {
            playable += player.zones[0].cards[card].type != 2;
        }
        player.cluttered = playable < static_cast<std::size_t>(player.gear);
        player.zones[5] = {};
        for (std::size_t card = 0; card < count; ++card) {
            const auto identity = narrow_integer<std::uint16_t>(
                id_data[player_index * 4 + card], "selected card identity"
            );
            append(
                player.zones[5],
                remove_at(player.zones[0], find_identity(player.zones[0], identity)),
                "played cards"
            );
        }
    }
    return advance_to_react_after(static_cast<std::size_t>(-1));
}

py::dict NativeState::apply_react(
    std::int16_t player_id,
    std::uint8_t cooldown_count,
    bool use_boost,
    bool use_adrenaline_speed,
    bool use_adrenaline_cooldown
) {
    if (state_.pending_kind == 0 && state_.phase == 3) {
        // Test-only state imports may begin exactly at the scalar ADRENALINE
        // boundary without a serialized live program counter.
        state_.pending_kind = 3;
        state_.pending_player = player_id;
    }
    expect_pending(3, player_id);
    const auto slot = player_slot(player_id);
    auto& player = state_.players[slot];
    state_.phase = 4;
    const auto gear_cooldown = player.gear == 1 ? 3 : (player.gear == 2 ? 1 : 0);
    const auto actual_cooldown = std::min<std::uint8_t>(
        cooldown_count,
        static_cast<std::uint8_t>(gear_cooldown + (use_adrenaline_cooldown ? 1 : 0))
    );
    for (std::uint8_t cooled = 0; cooled < actual_cooldown; ++cooled) {
        auto& hand = player.zones[0];
        std::size_t heat_index = hand.length;
        for (std::size_t card = 0; card < hand.length; ++card) {
            if (hand.cards[card].type == 2) {
                heat_index = card;
                break;
            }
        }
        if (heat_index == hand.length) {
            break;
        }
        append(player.zones[3], remove_at(hand, heat_index), "heat pool");
    }
    std::int16_t boost = 0;
    if (use_boost) {
        if (player.boost_used || player.zones[3].length == 0) {
            throw py::value_error("illegal native boost action");
        }
        pay_heat(player, 1);
        boost = resolve_flip(player);
        player.boost_used = true;
    }
    player.speed_boost = boost;
    player.speed_adrenaline = use_adrenaline_speed ? 1 : 0;
    const auto extra_movement = static_cast<std::int16_t>(
        boost + player.speed_adrenaline
    );
    if (extra_movement > 0) {
        credit_movement(player, extra_movement);
    }
    const auto turn_index = std::find(
        state_.turn_order.begin(),
        state_.turn_order.begin() + state_.turn_order_length,
        player_id
    ) - state_.turn_order.begin();
    if (player.finished) {
        replenish(player);
        return advance_to_react_after(turn_index);
    }
    if (slipstream_eligible(player)) {
        return decision_result("slipstream", player, static_cast<std::uint8_t>(turn_index));
    }
    check_corner(player);
    bool discardable = false;
    for (std::size_t card = 0; card < player.zones[0].length; ++card) {
        const auto type = player.zones[0].cards[card].type;
        discardable = discardable || type == 1 || type == 4;
    }
    if (discardable) {
        return decision_result("discard", player, static_cast<std::uint8_t>(turn_index));
    }
    replenish(player);
    return advance_to_react_after(turn_index);
}

py::dict NativeState::apply_slipstream(std::int16_t player_id, bool take) {
    expect_pending(4, player_id);
    const auto slot = player_slot(player_id);
    auto& player = state_.players[slot];
    state_.phase = 5;
    player.slipstream_moved = 0;
    if (take && slipstream_eligible(player)) {
        credit_movement(player, 2);
        player.slipstream_moved = 2;
    }
    check_corner(player);
    const auto turn_index = std::find(
        state_.turn_order.begin(),
        state_.turn_order.begin() + state_.turn_order_length,
        player_id
    ) - state_.turn_order.begin();
    for (std::size_t card = 0; card < player.zones[0].length; ++card) {
        const auto type = player.zones[0].cards[card].type;
        if (type == 1 || type == 4) {
            return decision_result("discard", player, static_cast<std::uint8_t>(turn_index));
        }
    }
    replenish(player);
    return advance_to_react_after(turn_index);
}

py::dict NativeState::apply_discard(
    std::int16_t player_id,
    const py::array& card_ids,
    std::uint8_t count
) {
    expect_pending(5, player_id);
    const py::dict payload = py::dict("card_ids"_a=card_ids);
    const auto ids = require_array<std::int64_t>(payload, "card_ids", {7});
    if (count > 7) {
        throw py::value_error("discard count exceeds fixed decision capacity");
    }
    auto& player = state_.players[player_slot(player_id)];
    state_.phase = 7;
    const auto* id_data = static_cast<const std::int64_t*>(ids.data());
    for (std::size_t card = 0; card < count; ++card) {
        const auto identity = narrow_integer<std::uint16_t>(
            id_data[card], "discard card identity"
        );
        const auto hand_index = find_identity(player.zones[0], identity);
        const auto type = player.zones[0].cards[hand_index].type;
        if (type != 1 && type != 4) {
            throw py::value_error("only speed and upgrade cards may be discarded");
        }
        append(
            player.zones[2],
            remove_at(player.zones[0], hand_index),
            "discard pile"
        );
    }
    replenish(player);
    const auto turn_index = std::find(
        state_.turn_order.begin(),
        state_.turn_order.begin() + state_.turn_order_length,
        player_id
    ) - state_.turn_order.begin();
    return advance_to_react_after(turn_index);
}

double NativeState::clip01(double value) {
    return std::clamp(value, 0.0, 1.0);
}

double NativeState::clip_signed(double value) {
    return std::clamp(value, -1.0, 1.0);
}

bool NativeState::adrenaline_eligible(const CompactPlayer& player) const {
    if (player.finished) {
        return false;
    }
    std::array<const CompactPlayer*, kPlayers> active{};
    std::size_t count = 0;
    for (const auto& other : state_.players) {
        if (other.present && !other.finished) {
            active[count++] = &other;
        }
    }
    if (count <= 1) {
        return false;
    }
    std::sort(active.begin(), active.begin() + count, [](const auto* left, const auto* right) {
        if (left->lap != right->lap) {
            return left->lap < right->lap;
        }
        if (left->position != right->position) {
            return left->position < right->position;
        }
        return left->player_id < right->player_id;
    });
    const auto recipients = state_.starting_player_count >= 5 ? 2U : 1U;
    for (std::size_t index = 0; index < std::min<std::size_t>(recipients, count); ++index) {
        if (active[index]->player_id == player.player_id) {
            return true;
        }
    }
    return false;
}

std::uint8_t NativeState::action_token(const CardToken& card) {
    if (card.type == 2) {
        return 0;  // H
    }
    if (card.type == 1 && card.value >= 1 && card.value <= 4) {
        return static_cast<std::uint8_t>(card.value);  // S1..S4
    }
    if (card.type == 3) {
        return 5;  // ST
    }
    if (card.type == 4 && card.value == 0) {
        return 6;  // U0
    }
    if (card.type == 4 && card.value == 5) {
        return 7;  // U5
    }
    return 0xffU;
}

std::array<std::uint8_t, 8> NativeState::card_requirements(
    std::size_t codec_index
) {
    if (codec_index >= 494) {
        throw py::value_error("card action is outside the codec range");
    }
    std::array<std::uint8_t, 8> required{};
    std::size_t current = 0;
    std::array<std::uint8_t, 8> candidate{};
    for (std::uint8_t size = 1; size <= 4; ++size) {
        std::function<bool(std::uint8_t, std::uint8_t)> visit;
        visit = [&](std::uint8_t depth, std::uint8_t first) {
            if (depth == size) {
                if (current++ == codec_index) {
                    required = candidate;
                    return true;
                }
                return false;
            }
            for (std::uint8_t token = first; token < 8; ++token) {
                ++candidate[token];
                if (visit(static_cast<std::uint8_t>(depth + 1), token)) {
                    return true;
                }
                --candidate[token];
            }
            return false;
        };
        if (visit(0, 0)) {
            return required;
        }
    }
    throw std::logic_error("card codec enumeration ended early");
}

py::dict NativeState::apply_gear_actions(const py::array& flat_actions) {
    const py::dict payload = py::dict("actions"_a=flat_actions);
    const auto actions = require_array<std::int64_t>(payload, "actions", {kPlayers});
    const auto* action_data = static_cast<const std::int64_t*>(actions.data());
    py::array_t<std::int64_t> gears(kPlayers);
    py::array_t<std::int64_t> costs(kPlayers);
    std::fill(gears.mutable_data(), gears.mutable_data() + kPlayers, -1);
    std::fill(costs.mutable_data(), costs.mutable_data() + kPlayers, -1);
    for (std::size_t slot = 0; slot < kPlayers; ++slot) {
        const auto& player = state_.players[slot];
        if (!player.present || player.finished || player.spun_out) {
            continue;
        }
        const auto action = action_data[slot];
        if (action < 0 || action >= 4) {
            throw py::value_error("gear flat action is outside its codec range");
        }
        const auto mask = legal_mask(player.player_id, 1);
        if (!mask.data()[action]) {
            throw py::value_error("gear flat action is not legal");
        }
        const auto gear = static_cast<std::int16_t>(action + 1);
        gears.mutable_data()[slot] = gear;
        costs.mutable_data()[slot] = std::abs(gear - player.gear) == 2 ? 1 : 0;
    }
    return apply_gears(gears, costs);
}

py::dict NativeState::apply_card_actions(const py::array& flat_actions) {
    const py::dict payload = py::dict("actions"_a=flat_actions);
    const auto actions = require_array<std::int64_t>(payload, "actions", {kPlayers});
    const auto* action_data = static_cast<const std::int64_t*>(actions.data());
    py::array_t<std::int64_t> identities({kPlayers, std::size_t{4}});
    py::array_t<std::int64_t> lengths(kPlayers);
    std::fill(identities.mutable_data(), identities.mutable_data() + kPlayers * 4, 0);
    std::fill(lengths.mutable_data(), lengths.mutable_data() + kPlayers, 0);
    for (std::size_t slot = 0; slot < kPlayers; ++slot) {
        const auto& player = state_.players[slot];
        if (!player.present || player.finished) {
            continue;
        }
        const auto action = action_data[slot];
        if (action < 4 || action >= 498) {
            throw py::value_error("cards flat action is outside its codec range");
        }
        const auto mask = legal_mask(player.player_id, 2);
        if (!mask.data()[action]) {
            throw py::value_error("cards flat action is not legal");
        }
        auto required = card_requirements(static_cast<std::size_t>(action - 4));
        std::size_t selected = 0;
        std::size_t playable = 0;
        for (std::size_t index = 0; index < player.zones[0].length; ++index) {
            playable += player.zones[0].cards[index].type != 2;
        }
        auto select = [&](bool heat_cards) {
            for (std::size_t index = 0; index < player.zones[0].length; ++index) {
                const auto& card = player.zones[0].cards[index];
                if ((card.type == 2) != heat_cards) {
                    continue;
                }
                const auto token = action_token(card);
                if (token != 0xffU && required[token] != 0) {
                    if (selected >= 4) {
                        throw py::value_error("decoded card action exceeds gear capacity");
                    }
                    identities.mutable_at(slot, selected++) = card.identity;
                    --required[token];
                }
            }
        };
        if (playable < static_cast<std::size_t>(player.gear)) {
            // The scalar forced-play contract concatenates every playable card
            // first and only then the Heat fillers, rather than preserving the
            // mixed hand order across those two categories.
            select(false);
            select(true);
        } else {
            select(false);
        }
        if (std::any_of(required.begin(), required.end(), [](auto count) { return count != 0; })) {
            throw py::value_error("card flat action cannot be realized from the hand");
        }
        lengths.mutable_data()[slot] = static_cast<std::int64_t>(selected);
    }
    return apply_cards_to_react(identities, lengths);
}

py::dict NativeState::apply_flat_action(
    std::int16_t player_id,
    std::uint16_t flat_action
) {
    if (flat_action >= kActionDim) {
        throw py::value_error("flat action is outside the codec range");
    }
    const auto mask = legal_mask(player_id, state_.pending_kind);
    if (!mask.data()[flat_action]) {
        throw py::value_error("flat action is not legal at the pending decision");
    }
    if (state_.pending_kind == 3) {
        if (flat_action < 498 || flat_action >= 506) {
            throw py::value_error("react flat action is outside its codec range");
        }
        constexpr std::array<std::array<std::uint8_t, 4>, 8> table{{
            {{0, 0, 0, 0}}, {{1, 0, 0, 0}}, {{2, 0, 0, 0}}, {{3, 0, 0, 0}},
            {{0, 1, 0, 0}}, {{0, 0, 1, 0}}, {{3, 0, 0, 1}}, {{0, 1, 1, 0}},
        }};
        const auto& action = table[flat_action - 498];
        return apply_react(player_id, action[0], action[1], action[2], action[3]);
    }
    if (state_.pending_kind == 4) {
        if (flat_action < 506 || flat_action >= 508) {
            throw py::value_error("slipstream flat action is outside its codec range");
        }
        return apply_slipstream(player_id, flat_action == 506);
    }
    if (state_.pending_kind == 5) {
        if (flat_action < 508 || flat_action >= 516) {
            throw py::value_error("discard flat action is outside its codec range");
        }
        const auto count = static_cast<std::uint8_t>(flat_action - 508);
        const auto& player = state_.players[player_slot(player_id)];
        std::array<CardToken, 7> discardable{};
        std::size_t length = 0;
        for (std::size_t index = 0; index < player.zones[0].length; ++index) {
            const auto& card = player.zones[0].cards[index];
            if (card.type == 1 || card.type == 4) {
                discardable[length++] = card;
            }
        }
        if (count > length) {
            throw py::value_error("discard flat action exceeds available cards");
        }
        std::sort(discardable.begin(), discardable.begin() + length, [](const auto& left, const auto& right) {
            if (left.value != right.value) {
                return left.value < right.value;
            }
            return left.identity < right.identity;
        });
        py::array_t<std::int64_t> identities(7);
        std::fill(identities.mutable_data(), identities.mutable_data() + 7, 0);
        for (std::size_t index = 0; index < count; ++index) {
            identities.mutable_data()[index] = discardable[index].identity;
        }
        return apply_discard(player_id, identities, count);
    }
    throw py::value_error("flat action does not match the pending decision");
}

py::array_t<float> NativeState::observation(
    std::int16_t player_id,
    std::uint8_t kind
) const {
    if (kind > 5) {
        throw py::value_error("observation decision kind must be 0..5");
    }
    const auto& player = state_.players[player_slot(player_id)];
    py::array_t<float> result(kObservationDim);
    auto* output = result.mutable_data();
    std::fill(output, output + kObservationDim, 0.0F);
    std::size_t cursor = 0;
    auto write = [&](double value) {
        output[cursor++] = static_cast<float>(clip_signed(value));
    };

    std::array<std::uint8_t, 8> hand_counts{};
    for (std::size_t index = 0; index < player.zones[0].length; ++index) {
        const auto& card = player.zones[0].cards[index];
        std::uint8_t slot = 0xffU;
        if (card.type == 1 && card.value >= 1 && card.value <= 4) {
            slot = static_cast<std::uint8_t>(card.value - 1);
        } else if (card.type == 2) {
            slot = 4;
        } else if (card.type == 3) {
            slot = 5;
        } else if (card.type == 4) {
            slot = card.value == 0 ? 6 : 7;
        }
        if (slot != 0xffU) {
            ++hand_counts[slot];
        }
    }
    for (const auto count : hand_counts) {
        write(clip01(static_cast<double>(count) / 7.0));
    }

    for (std::int8_t gear = 1; gear <= 4; ++gear) {
        write(player.gear == gear ? 1.0 : 0.0);
    }
    const auto length = state_.track.length == 0 ? 1 : state_.track.length;
    const auto laps = state_.track.laps == 0 ? 1 : state_.track.laps;
    write(clip01(static_cast<double>(player.position) / length));
    write(clip01(static_cast<double>(player.lap) / laps));
    write(clip01(static_cast<double>(player.zones[3].length) / 6.0));
    write(player.finished ? 1.0 : 0.0);

    std::array<std::uint16_t, 4> deck_counts{};
    for (const auto zone_index : {std::size_t{1}, std::size_t{2}}) {
        const auto& zone = player.zones[zone_index];
        for (std::size_t index = 0; index < zone.length; ++index) {
            const auto type = zone.cards[index].type;
            if (type >= 1 && type <= 4) {
                ++deck_counts[type - 1];
            }
        }
    }
    const auto deck_total = static_cast<std::uint16_t>(
        player.zones[1].length + player.zones[2].length
    );
    for (const auto count : deck_counts) {
        write(deck_total == 0 ? 0.0 : clip01(static_cast<double>(count) / deck_total));
    }
    write(deck_total == 0 ? 0.0 : clip01(static_cast<double>(player.zones[1].length) / deck_total));
    write(deck_total == 0 ? 0.0 : clip01(static_cast<double>(player.zones[2].length) / deck_total));

    std::array<std::size_t, kCorners> corner_order{};
    for (std::size_t index = 0; index < state_.track.corner_count; ++index) {
        corner_order[index] = index;
    }
    auto forward_distance = [&](std::int16_t start) {
        auto distance = static_cast<std::int16_t>((start - player.position) % length);
        if (distance < 0) {
            distance = static_cast<std::int16_t>(distance + length);
        }
        return distance == 0 ? static_cast<std::int16_t>(length) : distance;
    };
    std::stable_sort(
        corner_order.begin(),
        corner_order.begin() + state_.track.corner_count,
        [&](std::size_t left, std::size_t right) {
            return forward_distance(state_.track.corners[left][0])
                < forward_distance(state_.track.corners[right][0]);
        }
    );
    std::int16_t max_limit = 1;
    std::int16_t max_corner_length = 1;
    std::uint8_t max_lanes = 1;
    for (std::size_t index = 0; index < state_.track.corner_count; ++index) {
        const auto& corner = state_.track.corners[index];
        max_limit = std::max(max_limit, corner[2]);
        max_corner_length = std::max<std::int16_t>(
            max_corner_length, static_cast<std::int16_t>(corner[1] - corner[0] + 1)
        );
    }
    for (std::size_t index = 0; index < state_.track.length; ++index) {
        max_lanes = std::max(max_lanes, state_.track.lanes[index]);
    }
    for (std::size_t index = 0; index < kCorners; ++index) {
        if (index >= state_.track.corner_count) {
            write(0.0); write(0.0); write(0.0); write(0.0);
            continue;
        }
        const auto& corner = state_.track.corners[corner_order[index]];
        const auto entry_lanes = corner[0] >= 0 && corner[0] < state_.track.length
            ? state_.track.lanes[corner[0]] : 1;
        write(clip01(static_cast<double>(forward_distance(corner[0])) / length));
        write(clip01(static_cast<double>(corner[2]) / max_limit));
        write(clip01(static_cast<double>(corner[1] - corner[0] + 1) / max_corner_length));
        write(clip01(static_cast<double>(entry_lanes) / max_lanes));
    }
    write(clip01(static_cast<double>(laps - player.lap) / laps));
    const auto total_length = static_cast<std::int32_t>(length) * laps;
    const auto absolute_position = static_cast<std::int32_t>(player.lap) * length + player.position;
    write(clip01(static_cast<double>(total_length - absolute_position) / total_length));
    write(clip01(static_cast<double>(player.zones[3].length) / 6.0));
    write(clip01(static_cast<double>(player.position) / length));

    write(adrenaline_eligible(player) ? 1.0 : 0.0);
    std::size_t ahead = 0;
    std::size_t present_count = 0;
    for (const auto& other : state_.players) {
        if (!other.present) {
            continue;
        }
        ++present_count;
        if (other.player_id != player.player_id
            && (other.lap > player.lap
                || (other.lap == player.lap && other.position > player.position))) {
            ++ahead;
        }
    }
    write(present_count <= 1 ? 0.0 : clip01(static_cast<double>(ahead) / (present_count - 1)));

    std::array<const CompactPlayer*, kPlayers - 1> opponents{};
    std::size_t opponent_count = 0;
    for (const auto& other : state_.players) {
        if (other.present && other.player_id != player.player_id) {
            opponents[opponent_count++] = &other;
        }
    }
    std::sort(opponents.begin(), opponents.begin() + opponent_count, [](const auto* left, const auto* right) {
        return left->player_id < right->player_id;
    });
    const auto half = static_cast<double>(length) / 2.0;
    for (std::size_t index = 0; index < kPlayers - 1; ++index) {
        if (index >= opponent_count) {
            write(0.0); write(0.0); write(0.0); write(0.0); write(0.0);
            continue;
        }
        const auto& opponent = *opponents[index];
        auto raw = static_cast<std::int16_t>((opponent.position - player.position) % length);
        if (raw < 0) {
            raw = static_cast<std::int16_t>(raw + length);
        }
        auto signed_distance = static_cast<double>(raw);
        if (signed_distance > half) {
            signed_distance -= length;
        }
        write(1.0);
        write(half > 0.0 ? clip_signed(signed_distance / half) : 0.0);
        write(clip01(static_cast<double>(opponent.gear) / 4.0));
        write(clip_signed(static_cast<double>(opponent.lap - player.lap) / laps));
        write(opponent.finished ? 1.0 : 0.0);
    }

    const auto phase_start = cursor;
    if (kind != 0) {
        output[phase_start + kind - 1] = 1.0F;
        if (kind == 3) {
            const auto max_cooldown = player.gear == 1 ? 3 : (player.gear == 2 ? 1 : 0);
            output[phase_start + 5] = player.zones[3].length > 0 && !player.boost_used ? 1.0F : 0.0F;
            output[phase_start + 6] = adrenaline_eligible(player) ? 1.0F : 0.0F;
            output[phase_start + 7] = static_cast<float>(clip01(max_cooldown / 3.0));
        } else if (kind == 4) {
            output[phase_start + 8] = 1.0F;
        }
    }
    output[phase_start + 9] = static_cast<float>(clip01(state_.round / 50.0));
    cursor += 19;
    if (cursor != kObservationDim) {
        throw std::logic_error("native observation layout drifted");
    }
    return result;
}

py::array_t<bool> NativeState::legal_mask(
    std::int16_t player_id,
    std::uint8_t kind
) const {
    if (kind < 1 || kind > 5) {
        throw py::value_error("legal-mask decision kind must be 1..5");
    }
    const auto& player = state_.players[player_slot(player_id)];
    py::array_t<bool> result(kActionDim);
    auto* mask = result.mutable_data();
    std::fill(mask, mask + kActionDim, false);
    if (kind == 1) {
        if (player.spun_out) {
            throw py::value_error("spun-out players do not yield gear decisions");
        }
        for (std::int16_t target = 1; target <= 4; ++target) {
            const auto distance = std::abs(target - player.gear);
            mask[target - 1] = distance <= 1 || (distance == 2 && player.zones[3].length > 0);
        }
    } else if (kind == 2) {
        std::array<std::uint16_t, 8> available{};
        std::uint16_t playable = 0;
        std::uint16_t unknown_playable = 0;
        for (std::size_t index = 0; index < player.zones[0].length; ++index) {
            const auto& card = player.zones[0].cards[index];
            const auto token = action_token(card);
            if (card.type != 2) {
                ++playable;
            }
            if (token != 0xffU) {
                ++available[token];
            } else if (card.type != 2) {
                ++unknown_playable;
            }
        }
        std::array<std::uint8_t, 8> required{};
        std::size_t card_index = 0;
        for (std::uint8_t size = 1; size <= 4; ++size) {
            std::function<void(std::uint8_t, std::uint8_t)> visit;
            visit = [&](std::uint8_t depth, std::uint8_t first) {
                if (depth == size) {
                    bool legal = false;
                    if (playable >= player.gear) {
                        legal = size == player.gear && required[0] == 0;
                        for (std::size_t token = 0; token < required.size(); ++token) {
                            legal = legal && required[token] <= available[token];
                        }
                    } else if (unknown_playable == 0) {
                        auto forced = available;
                        forced[0] = std::min<std::uint16_t>(
                            available[0], static_cast<std::uint16_t>(player.gear - playable)
                        );
                        legal = true;
                        for (std::size_t token = 0; token < required.size(); ++token) {
                            legal = legal && required[token] == forced[token];
                        }
                    }
                    mask[4 + card_index++] = legal;
                    return;
                }
                for (std::uint8_t token = first; token < 8; ++token) {
                    ++required[token];
                    visit(static_cast<std::uint8_t>(depth + 1), token);
                    --required[token];
                }
            };
            visit(0, 0);
        }
        if (card_index != 494) {
            throw std::logic_error("native card codec enumeration drifted");
        }
    } else if (kind == 3) {
        constexpr std::array<std::array<std::uint8_t, 4>, 8> table{{
            {{0, 0, 0, 0}}, {{1, 0, 0, 0}}, {{2, 0, 0, 0}}, {{3, 0, 0, 0}},
            {{0, 1, 0, 0}}, {{0, 0, 1, 0}}, {{3, 0, 0, 1}}, {{0, 1, 1, 0}},
        }};
        const auto max_cooldown = player.gear == 1 ? 3 : (player.gear == 2 ? 1 : 0);
        const auto can_boost = player.zones[3].length > 0 && !player.boost_used;
        const auto has_adrenaline = adrenaline_eligible(player);
        for (std::size_t index = 0; index < table.size(); ++index) {
            const auto& action = table[index];
            mask[498 + index] = action[0] <= max_cooldown + action[3]
                && (!action[1] || can_boost)
                && (!(action[2] || action[3]) || has_adrenaline);
        }
    } else if (kind == 4) {
        mask[506] = true;
        mask[507] = true;
    } else {
        std::size_t discardable = 0;
        for (std::size_t index = 0; index < player.zones[0].length; ++index) {
            const auto type = player.zones[0].cards[index].type;
            discardable += type == 1 || type == 4;
        }
        mask[508] = true;
        for (std::size_t count = 1; count < 8 && count <= discardable; ++count) {
            mask[508 + count] = true;
        }
    }
    return result;
}

py::dict NativeState::bootstrap_row(std::int16_t player_id) const {
    py::dict result;
    result["observation"] = observation(player_id, 0);
    py::array_t<bool> mask(kActionDim);
    std::fill(mask.mutable_data(), mask.mutable_data() + kActionDim, false);
    result["legal_mask"] = std::move(mask);
    result["value_only"] = true;
    result["player_id"] = player_id;
    result["game_id"] = state_.game_id;
    return result;
}

double NativeState::reward(
    std::int16_t player_id,
    std::int16_t previous_position,
    std::int16_t previous_lap,
    bool previous_spun_out,
    bool done,
    bool terminated,
    bool solo_mode,
    double shaping_weight,
    double spinout_weight
) const {
    const auto& player = state_.players[player_slot(player_id)];
    double value = 0.0;
    if (done) {
        if (solo_mode) {
            if (terminated) {
                value += 5.0;
            }
        } else if (state_.starting_player_count > 1 && player.finished) {
            value += 1.0 - 2.0 * (player.finish_order - 1.0)
                / (state_.starting_player_count - 1.0);
        }
    }
    if (shaping_weight != 0.0) {
        const auto length = state_.track.length == 0 ? 1 : state_.track.length;
        const auto previous_absolute = previous_lap * length + previous_position;
        const auto current_absolute = player.lap * length + player.position;
        auto progress = static_cast<double>(current_absolute - previous_absolute) / length;
        if (solo_mode) {
            progress = std::max(0.0, progress);
        }
        value += shaping_weight * progress;
    }
    if (spinout_weight != 0.0 && player.spun_out && !previous_spun_out) {
        value -= std::min(0.05, spinout_weight);
    }
    return value;
}

double NativeState::terminal_margin(std::int16_t player_id) const {
    if (state_.starting_player_count <= 1) {
        return 0.0;
    }
    const auto& player = state_.players[player_slot(player_id)];
    const auto length = state_.track.length == 0 ? 1 : state_.track.length;
    const auto laps = state_.track.laps == 0 ? 1 : state_.track.laps;
    const auto total_length = length * laps;
    auto remaining = [&](const CompactPlayer& item) {
        if (item.finished) {
            return 0.0;
        }
        return std::max(
            0.0,
            static_cast<double>(total_length - (item.lap * length + item.position))
        );
    };
    auto best_opponent = std::numeric_limits<double>::infinity();
    for (const auto& other : state_.players) {
        if (other.present && other.player_id != player_id) {
            best_opponent = std::min(best_opponent, remaining(other));
        }
    }
    return clip_signed((best_opponent - remaining(player)) / length);
}

void NativeState::hash_bytes(
    std::uint64_t& hash,
    const void* data,
    std::size_t size
) const {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    for (std::size_t index = 0; index < size; ++index) {
        hash ^= bytes[index];
        hash *= 1099511628211ULL;
    }
}

std::uint64_t NativeState::canonical_digest() const {
    std::uint64_t hash = 14695981039346656037ULL;
    auto add = [&](const auto& value) { hash_bytes(hash, &value, sizeof(value)); };
    add(state_.game_id); add(state_.game_active); add(state_.round); add(state_.phase);
    add(state_.turn_order_length); add(state_.starting_player_count); add(state_.stress_counter);
    for (std::size_t index = 0; index < state_.turn_order_length; ++index) add(state_.turn_order[index]);
    add(state_.track.length); add(state_.track.corner_count); add(state_.track.start_count); add(state_.track.laps);
    for (std::size_t index = 0; index < state_.track.length; ++index) {
        add(state_.track.indices[index]); add(state_.track.lanes[index]);
    }
    for (std::size_t index = 0; index < state_.track.corner_count; ++index) {
        for (const auto field : state_.track.corners[index]) add(field);
    }
    for (std::size_t index = 0; index < state_.track.start_count; ++index) add(state_.track.starts[index]);
    for (const auto& player : state_.players) {
        if (!player.present) continue;
        add(player.player_id); add(player.active); add(player.gear); add(player.position); add(player.lap);
        add(player.spun_out); add(player.finished); add(player.finish_order); add(player.boost_used);
        add(player.speed_cards); add(player.speed_boost); add(player.speed_adrenaline);
        add(player.slipstream_moved); add(player.cluttered); add(player.turn_start_position); add(player.turn_start_lap);
        add(player.spin_length);
        for (std::size_t index = 0; index < player.spin_length; ++index) {
            add(player.spins[index].round); add(player.spins[index].corner_start);
        }
        for (const auto& zone : player.zones) {
            add(zone.length);
            for (std::size_t index = 0; index < zone.length; ++index) {
                add(zone.cards[index].identity); add(zone.cards[index].type); add(zone.cards[index].value);
            }
        }
    }
    std::array<std::uint32_t, 625> words{};
    bool gauss_present = false;
    double gauss = 0.0;
    state_.rng.export_numeric(words.data(), gauss_present, gauss);
    hash_bytes(hash, words.data(), words.size() * sizeof(words[0]));
    add(gauss_present); add(gauss);
    return hash;
}

py::array_t<std::uint64_t> NativeState::boundary_receipt() const {
    py::array_t<std::uint64_t> result(8);
    auto* values = result.mutable_data();
    values[0] = 1;
    values[1] = state_.game_id;
    values[2] = state_.decision_epoch;
    values[3] = state_.round;
    values[4] = state_.phase;
    values[5] = state_.pending_kind;
    values[6] = static_cast<std::uint64_t>(static_cast<std::int64_t>(state_.pending_player));
    values[7] = canonical_digest();
    return result;
}

struct BufferView {
    float* data;
    py::ssize_t rows;
    py::ssize_t columns;
};

BufferView require_float32_matrix(
    const py::array& array,
    const char* name,
    bool writable,
    py::ssize_t expected_columns = -1
) {
    if (!array.dtype().is(py::dtype::of<float>())) {
        throw py::type_error(std::string(name) + " must have dtype float32");
    }
    if (array.ndim() != 2) {
        throw py::value_error(std::string(name) + " must be a rank-2 matrix");
    }
    if ((array.flags() & py::array::c_style) == 0) {
        throw py::value_error(std::string(name) + " must be C-contiguous");
    }
    if (writable && !array.writeable()) {
        throw py::value_error(std::string(name) + " must be writable");
    }
    if (expected_columns >= 0 && array.shape(1) != expected_columns) {
        throw py::value_error(
            std::string(name) + " has an incompatible column count"
        );
    }
    return {
        static_cast<float*>(const_cast<void*>(array.data())),
        array.shape(0),
        array.shape(1),
    };
}

template <typename T>
const T* require_vector(
    const py::array& array,
    const char* name,
    py::ssize_t expected_rows
) {
    if (!array.dtype().is(py::dtype::of<T>())) {
        throw py::type_error(std::string(name) + " has the wrong dtype");
    }
    if (array.ndim() != 1 || array.shape(0) != expected_rows) {
        throw py::value_error(std::string(name) + " has the wrong shape");
    }
    if ((array.flags() & py::array::c_style) == 0) {
        throw py::value_error(std::string(name) + " must be C-contiguous");
    }
    return static_cast<const T*>(array.data());
}

enum class PoolSlotState : std::uint8_t {
    Runnable,
    Running,
    Complete,
    Finalized,
    Faulted,
};

struct PoolJob {
    explicit PoolJob(const NativeState& source) : state(source) {}

    NativeState state;
    PoolSlotState status = PoolSlotState::Runnable;
    std::uint32_t work_units = 0;
    bool inject_fault = false;
    std::uint64_t digest = 0;
    std::uint64_t work_receipt = 0;
};

class NativePool {
public:
    NativePool(const py::dict& config, const py::array& bound_rows)
        : row_capacity_(read_positive(config, "row_capacity", 0)),
          worker_count_(read_positive(config, "worker_count", 1)),
          slot_capacity_(read_positive(
              config, "slot_capacity", std::max<std::size_t>(worker_count_ * 2, 1)
          )),
          queue_capacity_(read_positive(config, "queue_capacity", slot_capacity_)),
          closed_(false),
          generation_(1) {
        const auto rows = require_float32_matrix(
            bound_rows, "bound_rows", true, kObservationDim
        );
        if (rows.rows != static_cast<py::ssize_t>(row_capacity_)) {
            throw py::value_error(
                "bound_rows row count must equal config row_capacity"
            );
        }
        if (queue_capacity_ > slot_capacity_) {
            throw py::value_error("queue_capacity cannot exceed slot_capacity");
        }
        workers_.reserve(worker_count_);
        for (std::size_t index = 0; index < worker_count_; ++index) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    ~NativePool() {
        shutdown_noexcept();
    }

    NativePool(const NativePool&) = delete;
    NativePool& operator=(const NativePool&) = delete;

    py::dict receipt() const {
        py::dict result;
        result["protocol_version"] = kProtocolVersion;
        result["state_schema_hash"] = kSchemaHash;
        result["codec_version"] = 3;
        result["rng_version"] = 1;
        result["receipt_version"] = 1;
        result["row_capacity"] = row_capacity_;
        result["observation_dim"] = kObservationDim;
        result["action_dim"] = kActionDim;
        result["worker_count"] = worker_count_;
        result["slot_capacity"] = slot_capacity_;
        result["queue_capacity"] = queue_capacity_;
        result["build_type"] = HEAT_NATIVE_DEBUG ? "debug" : "release";
        result["source_identity"] = HEAT_NATIVE_SOURCE_ID;
        result["compiler"] = compiler_identity();
        result["cpp_standard"] = 20;
        return result;
    }

    std::uint64_t roundtrip(const py::array& source, const py::array& destination) {
        ensure_open();
        const auto input = require_float32_matrix(source, "source", false);
        const auto output = require_float32_matrix(destination, "destination", true);
        if (input.rows != output.rows || input.columns != output.columns) {
            throw py::value_error("source and destination shapes must match");
        }
        if (source.data() == destination.data()) {
            throw py::value_error("source and destination must not alias");
        }
        const auto bytes = static_cast<std::size_t>(input.rows * input.columns)
            * sizeof(float);
        std::memcpy(output.data, input.data, bytes);
        return generation_.fetch_add(1);
    }

    std::uint64_t gil_release_smoke(std::uint32_t milliseconds) {
        ensure_open();
        py::gil_scoped_release release;
        std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
        return generation_.fetch_add(1);
    }

    void admit(const py::list& manifests) {
        ensure_open();
        std::vector<std::shared_ptr<PoolJob>> additions;
        additions.reserve(manifests.size());
        for (const auto& item : manifests) {
            const auto manifest = py::cast<py::dict>(item);
            if (!manifest.contains("state")) {
                throw py::key_error("manifest requires state");
            }
            const auto& state = py::cast<const NativeState&>(manifest["state"]);
            auto job = std::make_shared<PoolJob>(state);
            if (manifest.contains("work_units")) {
                job->work_units = py::cast<std::uint32_t>(manifest["work_units"]);
            }
            if (manifest.contains("inject_fault")) {
                job->inject_fault = py::cast<bool>(manifest["inject_fault"]);
            }
            additions.push_back(std::move(job));
        }

        std::lock_guard lock(mutex_);
        throw_if_faulted_locked();
        if (jobs_.size() + additions.size() > slot_capacity_) {
            throw py::value_error("native pool slot capacity would be exceeded");
        }
        if (runnable_.size() + additions.size() > queue_capacity_) {
            throw py::value_error("native pool runnable queue is full");
        }
        std::vector<std::uint64_t> addition_ids;
        addition_ids.reserve(additions.size());
        for (const auto& job : additions) {
            const auto game_id = job->state.game_id();
            if (jobs_.contains(game_id)) {
                throw py::value_error("manifest contains a duplicate live game_id");
            }
            if (std::find(addition_ids.begin(), addition_ids.end(), game_id)
                != addition_ids.end()) {
                throw py::value_error("manifest contains a duplicate game_id");
            }
            addition_ids.push_back(game_id);
        }
        for (const auto& job : additions) {
            const auto game_id = job->state.game_id();
            jobs_.emplace(game_id, job);
            runnable_.push_back(game_id);
            ++admitted_count_;
        }
        runnable_high_water_ = std::max(runnable_high_water_, runnable_.size());
        work_cv_.notify_all();
    }

    py::list wait_completed(std::size_t max_games, std::uint32_t timeout_ms) {
        ensure_open();
        if (max_games == 0) {
            throw py::value_error("max_games must be positive");
        }
        std::vector<std::shared_ptr<PoolJob>> results;
        {
            py::gil_scoped_release release;
            std::unique_lock lock(mutex_);
            completed_cv_.wait_for(
                lock,
                std::chrono::milliseconds(timeout_ms),
                [&] { return !completed_.empty() || faulted_ || closed_.load(); }
            );
            if (faulted_) {
                const auto message = fault_message_;
                lock.unlock();
                throw std::runtime_error(message);
            }
            while (!completed_.empty() && results.size() < max_games) {
                const auto game_id = completed_.front();
                completed_.pop_front();
                const auto found = jobs_.find(game_id);
                if (found == jobs_.end()) {
                    continue;
                }
                found->second->status = PoolSlotState::Finalized;
                results.push_back(found->second);
                jobs_.erase(found);
                ++finalized_count_;
            }
        }
        std::sort(results.begin(), results.end(), [](const auto& left, const auto& right) {
            return left->state.game_id() < right->state.game_id();
        });
        py::list output;
        for (const auto& job : results) {
            output.append(py::dict(
                "game_id"_a = job->state.game_id(),
                "digest"_a = job->digest,
                "work_receipt"_a = job->work_receipt
            ));
        }
        return output;
    }

    py::dict stats() const {
        std::lock_guard lock(mutex_);
        py::dict result;
        result["worker_count"] = worker_count_;
        result["slot_capacity"] = slot_capacity_;
        result["queue_capacity"] = queue_capacity_;
        result["live_slots"] = jobs_.size();
        result["runnable"] = runnable_.size();
        result["running"] = running_count_;
        result["completed_pending"] = completed_.size();
        result["runnable_high_water"] = runnable_high_water_;
        result["admitted"] = admitted_count_;
        result["completed"] = completed_count_;
        result["finalized"] = finalized_count_;
        result["worker_waits"] = worker_wait_count_;
        result["faulted"] = faulted_;
        return result;
    }

    void close(std::uint32_t timeout_ms) {
        if (closed_.exchange(true)) {
            return;
        }
        const auto started = std::chrono::steady_clock::now();
        {
            std::lock_guard lock(mutex_);
            stop_requested_.store(true);
        }
        work_cv_.notify_all();
        {
            py::gil_scoped_release release;
            for (auto& worker : workers_) {
                if (worker.joinable()) {
                    worker.join();
                }
            }
        }
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - started
        ).count();
        if (elapsed > timeout_ms) {
            throw std::runtime_error("NativePool shutdown exceeded timeout_ms");
        }
    }

    bool closed() const noexcept {
        return closed_.load();
    }

private:
    static std::size_t read_positive(
        const py::dict& config,
        const char* name,
        std::size_t fallback
    ) {
        if (!config.contains(name)) {
            if (fallback == 0) {
                throw py::key_error(std::string("config requires ") + name);
            }
            return fallback;
        }
        const auto value = py::cast<std::int64_t>(config[name]);
        if (value < 1) {
            throw py::value_error(std::string(name) + " must be positive");
        }
        return static_cast<std::size_t>(value);
    }

    static std::string compiler_identity() {
#if defined(_MSC_VER)
        return "msvc-" + std::to_string(_MSC_VER);
#elif defined(__clang__)
        return "clang-" + std::string(__clang_version__);
#elif defined(__GNUC__)
        return "gcc-" + std::string(__VERSION__);
#else
        return "unknown";
#endif
    }

    static std::uint64_t mix_work(std::uint64_t value) noexcept {
        value += 0x9e3779b97f4a7c15ULL;
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31);
    }

    void worker_loop() noexcept {
        while (true) {
            std::shared_ptr<PoolJob> job;
            {
                std::unique_lock lock(mutex_);
                ++worker_wait_count_;
                work_cv_.wait(lock, [&] {
                    return stop_requested_.load() || !runnable_.empty();
                });
                if (stop_requested_.load()) {
                    return;
                }
                const auto game_id = runnable_.front();
                runnable_.pop_front();
                const auto found = jobs_.find(game_id);
                if (found == jobs_.end()) {
                    continue;
                }
                job = found->second;
                job->status = PoolSlotState::Running;
                ++running_count_;
            }
            try {
                if (job->inject_fault) {
                    throw std::runtime_error(
                        "injected native worker fault for game "
                        + std::to_string(job->state.game_id())
                    );
                }
                job->digest = job->state.canonical_digest();
                auto receipt = job->digest;
                for (std::uint32_t index = 0; index < job->work_units; ++index) {
                    if ((index & 1023U) == 0 && stop_requested_.load()) {
                        std::lock_guard lock(mutex_);
                        --running_count_;
                        return;
                    }
                    receipt = mix_work(receipt ^ index);
                }
                job->work_receipt = receipt;
                {
                    std::lock_guard lock(mutex_);
                    --running_count_;
                    job->status = PoolSlotState::Complete;
                    completed_.push_back(job->state.game_id());
                    ++completed_count_;
                }
                completed_cv_.notify_all();
            } catch (const std::exception& error) {
                {
                    std::lock_guard lock(mutex_);
                    --running_count_;
                    job->status = PoolSlotState::Faulted;
                    faulted_ = true;
                    fault_message_ = error.what();
                    stop_requested_.store(true);
                }
                completed_cv_.notify_all();
                work_cv_.notify_all();
                return;
            }
        }
    }

    void throw_if_faulted_locked() const {
        if (faulted_) {
            throw std::runtime_error(fault_message_);
        }
    }

    void ensure_open() const {
        if (closed_.load()) {
            throw std::runtime_error("NativePool is closed");
        }
    }

    void shutdown_noexcept() noexcept {
        if (!closed_.exchange(true)) {
            {
                std::lock_guard lock(mutex_);
                stop_requested_.store(true);
            }
            work_cv_.notify_all();
        }
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    std::size_t row_capacity_;
    std::size_t worker_count_;
    std::size_t slot_capacity_;
    std::size_t queue_capacity_;
    std::atomic<bool> closed_;
    std::atomic<std::uint64_t> generation_;
    mutable std::mutex mutex_;
    std::condition_variable work_cv_;
    std::condition_variable completed_cv_;
    std::vector<std::thread> workers_;
    std::unordered_map<std::uint64_t, std::shared_ptr<PoolJob>> jobs_;
    std::deque<std::uint64_t> runnable_;
    std::deque<std::uint64_t> completed_;
    std::atomic<bool> stop_requested_{false};
    bool faulted_ = false;
    std::string fault_message_;
    std::size_t running_count_ = 0;
    std::size_t runnable_high_water_ = 0;
    std::uint64_t admitted_count_ = 0;
    std::uint64_t completed_count_ = 0;
    std::uint64_t finalized_count_ = 0;
    std::uint64_t worker_wait_count_ = 0;
};

struct TrajectoryRow {
    std::array<float, kObservationDim> observation{};
    std::array<std::uint64_t, 9> legal_mask{};
    std::uint64_t game_id = 0;
    std::uint32_t decision_sequence = 0;
    std::int16_t seat_id = 0;
    std::uint16_t policy_id = 0;
    std::uint16_t action = 0;
    float logp = 0.0F;
    float value = 0.0F;
    float reward = 0.0F;
    bool done = false;
    bool record_for_ppo = true;
};

struct StreamKey {
    std::uint64_t game_id;
    std::int16_t seat_id;

    bool operator<(const StreamKey& other) const noexcept {
        return game_id < other.game_id
            || (game_id == other.game_id && seat_id < other.seat_id);
    }
};

class NativeTrajectoryArena {
public:
    NativeTrajectoryArena(std::size_t block_rows, std::size_t block_count)
        : block_rows_(block_rows), block_count_(block_count) {
        if (block_rows_ == 0 || block_count_ == 0) {
            throw py::value_error("trajectory block sizes must be positive");
        }
        blocks_.reserve(block_count_);
        for (std::size_t index = 0; index < block_count_; ++index) {
            blocks_.push_back(std::make_unique<TrajectoryRow[]>(block_rows_));
        }
        rows_.reserve(block_rows_ * block_count_);
    }

    void append(
        const py::array& observations,
        const py::array& legal_masks,
        const py::array& game_ids,
        const py::array& seat_ids,
        const py::array& decision_sequences,
        const py::array& policy_ids,
        const py::array& actions,
        const py::array& logps,
        const py::array& values,
        const py::array& rewards,
        const py::array& dones,
        const py::array& record_for_ppo
    ) {
        const auto obs = require_float32_matrix(
            observations, "observations", false, kObservationDim
        );
        const auto row_count = obs.rows;
        if (!legal_masks.dtype().is(py::dtype::of<bool>())) {
            throw py::type_error("legal_masks has the wrong dtype");
        }
        if (legal_masks.ndim() != 2 || legal_masks.shape(0) != row_count
            || legal_masks.shape(1) != static_cast<py::ssize_t>(kActionDim)) {
            throw py::value_error("legal_masks has the wrong shape");
        }
        if ((legal_masks.flags() & py::array::c_style) == 0) {
            throw py::value_error("legal_masks must be C-contiguous");
        }
        const auto* masks = static_cast<const bool*>(legal_masks.data());
        const auto* ids = require_vector<std::uint64_t>(game_ids, "game_ids", row_count);
        const auto* seats = require_vector<std::int16_t>(seat_ids, "seat_ids", row_count);
        const auto* sequences = require_vector<std::uint32_t>(
            decision_sequences, "decision_sequences", row_count
        );
        const auto* policies = require_vector<std::uint16_t>(
            policy_ids, "policy_ids", row_count
        );
        const auto* selected = require_vector<std::int64_t>(actions, "actions", row_count);
        const auto* old_logps = require_vector<float>(logps, "logps", row_count);
        const auto* old_values = require_vector<float>(values, "values", row_count);
        const auto* row_rewards = require_vector<float>(rewards, "rewards", row_count);
        const auto* row_dones = require_vector<bool>(dones, "dones", row_count);
        const auto* row_record = require_vector<bool>(
            record_for_ppo, "record_for_ppo", row_count
        );
        if (rows_.size() + static_cast<std::size_t>(row_count) > capacity()) {
            throw py::value_error("trajectory arena capacity would be exceeded");
        }
        for (py::ssize_t index = 0; index < row_count; ++index) {
            if (selected[index] < 0 || selected[index] >= static_cast<std::int64_t>(kActionDim)) {
                throw py::value_error("trajectory action is outside the frozen codec");
            }
            if (!masks[index * kActionDim + selected[index]]) {
                throw py::value_error("trajectory action is not legal in its stored mask");
            }
        }

        for (py::ssize_t index = 0; index < row_count; ++index) {
            auto* row = allocate_row();
            std::copy(
                obs.data + index * kObservationDim,
                obs.data + (index + 1) * kObservationDim,
                row->observation.begin()
            );
            for (std::size_t action = 0; action < kActionDim; ++action) {
                if (masks[index * kActionDim + action]) {
                    row->legal_mask[action / 64] |= 1ULL << (action % 64);
                }
            }
            row->game_id = ids[index];
            row->seat_id = seats[index];
            row->decision_sequence = sequences[index];
            row->policy_id = policies[index];
            row->action = static_cast<std::uint16_t>(selected[index]);
            row->logp = old_logps[index];
            row->value = old_values[index];
            row->reward = row_rewards[index];
            row->done = row_dones[index];
            row->record_for_ppo = row_record[index];
            rows_.push_back(row);
            const StreamKey key{row->game_id, row->seat_id};
            const auto found = last_rows_.find(key);
            if (found == last_rows_.end()
                || row->decision_sequence > found->second->decision_sequence) {
                last_rows_[key] = row;
            }
        }
    }

    void fold_bootstrap(
        std::uint64_t game_id,
        std::int16_t seat_id,
        float bootstrap_value,
        double gamma
    ) {
        auto* row = last_row(game_id, seat_id);
        if (row->done) {
            throw py::value_error("trajectory stream is already closed");
        }
        row->reward = static_cast<float>(
            static_cast<double>(row->reward) + gamma * bootstrap_value
        );
        row->done = true;
        ++bootstrap_folds_;
    }

    void close_stream(
        std::uint64_t game_id,
        std::int16_t seat_id,
        float terminal_reward,
        float terminal_margin,
        double margin_coefficient
    ) {
        auto* row = last_row(game_id, seat_id);
        if (row->done) {
            throw py::value_error("trajectory stream is already closed");
        }
        row->reward = static_cast<float>(
            static_cast<double>(row->reward)
            + terminal_reward
            + margin_coefficient * terminal_margin
        );
        row->done = true;
        ++closed_streams_;
    }

    py::dict finalize(double gamma, double gae_lambda) const {
        std::vector<const TrajectoryRow*> ordered;
        ordered.reserve(rows_.size());
        for (const auto* row : rows_) {
            if (row->record_for_ppo) {
                ordered.push_back(row);
            }
        }
        std::sort(ordered.begin(), ordered.end(), [](const auto* left, const auto* right) {
            if (left->game_id != right->game_id) return left->game_id < right->game_id;
            if (left->seat_id != right->seat_id) return left->seat_id < right->seat_id;
            return left->decision_sequence < right->decision_sequence;
        });
        for (std::size_t index = 1; index < ordered.size(); ++index) {
            const auto* previous = ordered[index - 1];
            const auto* current = ordered[index];
            if (previous->game_id == current->game_id
                && previous->seat_id == current->seat_id
                && previous->decision_sequence == current->decision_sequence) {
                throw py::value_error("duplicate trajectory logical identity");
            }
        }

        const auto count = static_cast<py::ssize_t>(ordered.size());
        py::array_t<float> observations({count, static_cast<py::ssize_t>(kObservationDim)});
        py::array_t<bool> masks({count, static_cast<py::ssize_t>(kActionDim)});
        py::array_t<std::int64_t> actions(count);
        py::array_t<float> logps(count);
        py::array_t<float> values(count);
        py::array_t<float> rewards(count);
        py::array_t<float> dones(count);
        py::array_t<float> advantages(count);
        py::array_t<float> returns(count);
        py::array_t<std::uint64_t> game_ids(count);
        py::array_t<std::int16_t> seat_ids(count);
        py::array_t<std::uint32_t> sequences(count);
        py::array_t<std::uint16_t> policy_ids(count);
        if (count > 0) {
            std::fill(
                masks.mutable_data(), masks.mutable_data() + count * kActionDim, false
            );
        }
        for (py::ssize_t index = 0; index < count; ++index) {
            const auto* row = ordered[index];
            std::copy(
                row->observation.begin(),
                row->observation.end(),
                observations.mutable_data() + index * kObservationDim
            );
            for (std::size_t action = 0; action < kActionDim; ++action) {
                masks.mutable_data()[index * kActionDim + action]
                    = (row->legal_mask[action / 64] & (1ULL << (action % 64))) != 0;
            }
            actions.mutable_data()[index] = row->action;
            logps.mutable_data()[index] = row->logp;
            values.mutable_data()[index] = row->value;
            rewards.mutable_data()[index] = row->reward;
            dones.mutable_data()[index] = row->done ? 1.0F : 0.0F;
            game_ids.mutable_data()[index] = row->game_id;
            seat_ids.mutable_data()[index] = row->seat_id;
            sequences.mutable_data()[index] = row->decision_sequence;
            policy_ids.mutable_data()[index] = row->policy_id;
        }

        std::size_t stream_start = 0;
        while (stream_start < ordered.size()) {
            auto stream_end = stream_start + 1;
            while (stream_end < ordered.size()
                && ordered[stream_end]->game_id == ordered[stream_start]->game_id
                && ordered[stream_end]->seat_id == ordered[stream_start]->seat_id) {
                ++stream_end;
            }
            if (!ordered[stream_end - 1]->done) {
                throw py::value_error("trajectory stream is not terminal or bootstrapped");
            }
            const auto gamma_f = static_cast<float>(gamma);
            const auto gae_lambda_f = static_cast<float>(gae_lambda);
            float last_gae = 0.0F;
            for (auto offset = stream_end; offset-- > stream_start;) {
                const auto* row = ordered[offset];
                const auto nonterminal = row->done ? 0.0F : 1.0F;
                const auto next_value = offset + 1 == stream_end
                    ? 0.0F
                    : ordered[offset + 1]->value;
                const auto delta = row->reward
                    + gamma_f * next_value * nonterminal
                    - row->value;
                last_gae = delta
                    + gamma_f * gae_lambda_f * nonterminal * last_gae;
                advantages.mutable_data()[offset] = last_gae;
                returns.mutable_data()[offset] = last_gae + row->value;
            }
            stream_start = stream_end;
        }
        py::dict result;
        result["obs"] = std::move(observations);
        result["masks"] = std::move(masks);
        result["actions"] = std::move(actions);
        result["logps"] = std::move(logps);
        result["values"] = std::move(values);
        result["rewards"] = std::move(rewards);
        result["dones"] = std::move(dones);
        result["advantages"] = std::move(advantages);
        result["returns"] = std::move(returns);
        result["game_ids"] = std::move(game_ids);
        result["seat_ids"] = std::move(seat_ids);
        result["decision_sequences"] = std::move(sequences);
        result["policy_ids"] = std::move(policy_ids);
        return result;
    }

    py::dict stats() const {
        py::dict result;
        result["rows"] = rows_.size();
        result["capacity"] = capacity();
        result["blocks_used"] = rows_.empty() ? 0 : (rows_.size() - 1) / block_rows_ + 1;
        result["blocks_total"] = block_count_;
        result["bootstrap_folds"] = bootstrap_folds_;
        result["closed_streams"] = closed_streams_;
        return result;
    }

    void reset() noexcept {
        rows_.clear();
        last_rows_.clear();
        used_rows_ = 0;
        bootstrap_folds_ = 0;
        closed_streams_ = 0;
    }

private:
    std::size_t capacity() const noexcept {
        return block_rows_ * block_count_;
    }

    TrajectoryRow* allocate_row() {
        const auto block = used_rows_ / block_rows_;
        const auto offset = used_rows_ % block_rows_;
        ++used_rows_;
        auto* row = &blocks_[block][offset];
        *row = TrajectoryRow{};
        return row;
    }

    TrajectoryRow* last_row(std::uint64_t game_id, std::int16_t seat_id) {
        const auto found = last_rows_.find(StreamKey{game_id, seat_id});
        if (found == last_rows_.end()) {
            throw py::key_error("trajectory stream does not exist");
        }
        return found->second;
    }

    std::size_t block_rows_;
    std::size_t block_count_;
    std::size_t used_rows_ = 0;
    std::vector<std::unique_ptr<TrajectoryRow[]>> blocks_;
    std::vector<TrajectoryRow*> rows_;
    std::map<StreamKey, TrajectoryRow*> last_rows_;
    std::uint64_t bootstrap_folds_ = 0;
    std::uint64_t closed_streams_ = 0;
};

}  // namespace heat_native

PYBIND11_MODULE(_core, module) {
    module.doc() = "HEAT Direction D3 native substrate";
    module.def("_fnv1a64", &heat_native::fnv1a64);
    py::class_<heat_native::PythonMt19937>(module, "_PythonMt19937")
        .def(py::init<const py::tuple&>())
        .def("getstate", &heat_native::PythonMt19937::export_python_state)
        .def("random", &heat_native::PythonMt19937::random)
        .def("getrandbits", &heat_native::PythonMt19937::getrandbits)
        .def("randbelow", &heat_native::PythonMt19937::randbelow)
        .def("shuffled", &heat_native::PythonMt19937::shuffled);
    py::class_<heat_native::NativeState>(module, "_NativeState")
        .def(py::init<const py::dict&>())
        .def("clone", &heat_native::NativeState::clone)
        .def_property_readonly("game_id", &heat_native::NativeState::game_id)
        .def_property_readonly("round_num", &heat_native::NativeState::round_num)
        .def("reward_state", &heat_native::NativeState::reward_state)
        .def("export_payload", &heat_native::NativeState::export_payload)
        .def("receipt", &heat_native::NativeState::receipt)
        .def("start_round", &heat_native::NativeState::start_round)
        .def("apply_gears", &heat_native::NativeState::apply_gears)
        .def("apply_cards_to_react", &heat_native::NativeState::apply_cards_to_react)
        .def("apply_react", &heat_native::NativeState::apply_react)
        .def("apply_slipstream", &heat_native::NativeState::apply_slipstream)
        .def("apply_discard", &heat_native::NativeState::apply_discard)
        .def("apply_gear_actions", &heat_native::NativeState::apply_gear_actions)
        .def("apply_card_actions", &heat_native::NativeState::apply_card_actions)
        .def("apply_flat_action", &heat_native::NativeState::apply_flat_action)
        .def("observation", &heat_native::NativeState::observation)
        .def("legal_mask", &heat_native::NativeState::legal_mask)
        .def("bootstrap_row", &heat_native::NativeState::bootstrap_row)
        .def("reward", &heat_native::NativeState::reward)
        .def("terminal_margin", &heat_native::NativeState::terminal_margin)
        .def("canonical_digest", &heat_native::NativeState::canonical_digest)
        .def("boundary_receipt", &heat_native::NativeState::boundary_receipt);
    py::class_<heat_native::NativePool>(module, "NativePool")
        .def(py::init<const py::dict&, const py::array&>())
        .def("receipt", &heat_native::NativePool::receipt)
        .def("roundtrip", &heat_native::NativePool::roundtrip)
        .def("gil_release_smoke", &heat_native::NativePool::gil_release_smoke)
        .def("admit", &heat_native::NativePool::admit)
        .def(
            "wait_completed",
            &heat_native::NativePool::wait_completed,
            py::arg("max_games"),
            py::arg("timeout_ms")
        )
        .def("stats", &heat_native::NativePool::stats)
        .def("close", &heat_native::NativePool::close, py::arg("timeout_ms") = 1000)
        .def_property_readonly("closed", &heat_native::NativePool::closed);
    py::class_<heat_native::NativeTrajectoryArena>(module, "NativeTrajectoryArena")
        .def(py::init<std::size_t, std::size_t>())
        .def("append", &heat_native::NativeTrajectoryArena::append)
        .def("fold_bootstrap", &heat_native::NativeTrajectoryArena::fold_bootstrap)
        .def("close_stream", &heat_native::NativeTrajectoryArena::close_stream)
        .def("finalize", &heat_native::NativeTrajectoryArena::finalize)
        .def("stats", &heat_native::NativeTrajectoryArena::stats)
        .def("reset", &heat_native::NativeTrajectoryArena::reset);
}
