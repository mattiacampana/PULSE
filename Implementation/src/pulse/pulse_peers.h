#ifndef SENSWEAR_PULSE_PEERS_H_
#define SENSWEAR_PULSE_PEERS_H_

#include "pulse_math.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef PULSE_MAX_PEERS
#define PULSE_MAX_PEERS 16U
#endif

#define PULSE_UCB_TIE_EPSILON 1.0e-12

typedef struct {
	uint64_t peer_id;
	float utility;
	uint32_t visits;
	uint8_t occupied;
} pulse_peer_entry_t;

typedef struct {
	pulse_peer_entry_t entries[PULSE_MAX_PEERS];
	uint32_t total_selection_opportunities;
	uint32_t tie_break_rng_state;
	float initial_utility;
	float ucb_exploration;
} pulse_peer_table_t;

typedef enum {
	PULSE_PEER_SELECTION_NO_CANDIDATES = 0,
	PULSE_PEER_SELECTION_NO_CONTACT,
	PULSE_PEER_SELECTION_CONTACT
} pulse_peer_selection_action_t;

typedef struct {
	pulse_peer_selection_action_t action;
	uint64_t peer_id;
	size_t candidate_index;
	float ucb_index;
} pulse_peer_selection_t;

size_t pulse_peer_table_storage_size(void);
size_t pulse_peer_entry_storage_size(void);

pulse_status_t pulse_peer_table_init(pulse_peer_table_t* table,
									 float initial_utility,
									 float ucb_exploration,
									 uint32_t tie_break_seed);

const pulse_peer_entry_t* pulse_peer_find(const pulse_peer_table_t* table, uint64_t peer_id);

size_t pulse_peer_count(const pulse_peer_table_t* table);

/*
 * Selects against the implicit no-contact index 0. The candidate IDs must be unique.
 * A non-empty call increments total_selection_opportunities exactly once, including when
 * no-contact wins. The seeded PRNG is consumed only when multiple positive best indices tie.
 */
pulse_status_t pulse_peer_select(pulse_peer_table_t* table,
								 const uint64_t* candidate_ids,
								 size_t candidate_count,
								 pulse_peer_selection_t* selection);

/* Adds a peer lazily and updates signed EMA utility after a completed exchange only. */
pulse_status_t pulse_peer_observe(pulse_peer_table_t* table,
								  uint64_t peer_id,
								  float agreement,
								  float utility_momentum);

#ifdef __cplusplus
}
#endif

#endif /* SENSWEAR_PULSE_PEERS_H_ */
