#pragma once

#include "rkva-native-state.h"

#include <cstdint>
#include <string>
#include <vector>

// Host-side owner of the authoritative native KV history for one remote layer.
// Both the KVarN4 record producer (descriptor layouts 1..3) and the layout-4
// native Q4_0 row producer publish one op16 payload per append; the ordered
// socket/async machinery is shared and never inspects the concrete payload
// representation. Rewind support stays representation-specific: layout 4 has
// no exact F16 tail and therefore reports can_rewind() == false.
class rkva_native_owner {
public:
    virtual ~rkva_native_owner() = default;

    virtual uint32_t rows() const = 0;

    // Optional device-side precompute; the CPU Q4_0 producer has none.
    virtual bool prepare(const float * keys, const float * values, const int32_t * positions,
                         uint32_t count, const float * device_keys = nullptr,
                         const float * device_values = nullptr) = 0;

    // Emits one complete op16 payload (rkva_kvarn_range_upload header + planes).
    virtual bool append(const float * keys, const float * values, const int32_t * positions,
                        uint32_t count, bool reset_snapshot, std::vector<uint8_t> & payload) = 0;

    virtual bool can_rewind(uint32_t keep) const = 0;
    virtual bool rewind_tail(uint32_t keep, std::vector<uint8_t> & output) const = 0;
    virtual bool rewind(uint32_t keep) = 0;

    // Native KVarN4 FULL persistence. Other native row layouts keep the
    // existing refusal: their standard Q4/Q8 snapshot path is separate.
    virtual bool state_export(uint64_t generation, rkva_native_state::View & view) const {
        (void) generation;
        (void) view;
        return false;
    }
    virtual bool state_import(rkva_native_state::Storage & state, std::string * error) {
        (void) state;
        (void) error;
        return false;
    }
};
