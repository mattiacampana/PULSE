#include "pulse_peers.h"

#include <math.h>
#include <string.h>

#define PULSE_DEFAULT_NONZERO_SEED 0x6D2B79F5U

static pulse_peer_entry_t* find_mutable(pulse_peer_table_t* table, uint64_t peer_id) {
	size_t index;

	for (index = 0U; index < PULSE_MAX_PEERS; ++index) {
		if (table->entries[index].occupied != 0U && table->entries[index].peer_id == peer_id) {
			return &table->entries[index];
		}
	}

	return NULL;
}

static pulse_peer_entry_t* find_free(pulse_peer_table_t* table) {
	size_t index;

	for (index = 0U; index < PULSE_MAX_PEERS; ++index) {
		if (table->entries[index].occupied == 0U) {
			return &table->entries[index];
		}
	}

	return NULL;
}

static uint32_t next_random(pulse_peer_table_t* table) {
	uint32_t value = table->tie_break_rng_state;

	value ^= value << 13;
	value ^= value >> 17;
	value ^= value << 5;
	table->tie_break_rng_state = value;
	return value;
}

static uint32_t bounded_random(pulse_peer_table_t* table, uint32_t bound) {
	uint32_t threshold = (uint32_t) (-bound) % bound;
	uint32_t value;

	do {
		value = next_random(table);
	} while (value < threshold);

	return value % bound;
}

static double peer_index(const pulse_peer_table_t* table, uint64_t peer_id, double log_term) {
	const pulse_peer_entry_t* entry = pulse_peer_find(table, peer_id);
	double utility = entry != NULL ? (double) entry->utility : (double) table->initial_utility;
	double visits = entry != NULL ? (double) entry->visits : 0.0;
	double bonus = (double) table->ucb_exploration * sqrt(log_term / (1.0 + visits));

	return utility + bonus;
}

size_t pulse_peer_table_storage_size(void) {
	return sizeof(pulse_peer_table_t);
}

size_t pulse_peer_entry_storage_size(void) {
	return sizeof(pulse_peer_entry_t);
}

pulse_status_t pulse_peer_table_init(pulse_peer_table_t* table,
									 float initial_utility,
									 float ucb_exploration,
									 uint32_t tie_break_seed) {
	if (table == NULL || !isfinite(initial_utility) || initial_utility < -1.0f ||
		initial_utility > 1.0f || !isfinite(ucb_exploration) || ucb_exploration < 0.0f) {
		return PULSE_STATUS_BAD_ARGUMENT;
	}

	memset(table, 0, sizeof(*table));
	table->initial_utility = initial_utility;
	table->ucb_exploration = ucb_exploration;
	table->tie_break_rng_state = tie_break_seed != 0U ? tie_break_seed : PULSE_DEFAULT_NONZERO_SEED;
	return PULSE_STATUS_OK;
}

const pulse_peer_entry_t* pulse_peer_find(const pulse_peer_table_t* table, uint64_t peer_id) {
	size_t index;

	if (table == NULL) {
		return NULL;
	}

	for (index = 0U; index < PULSE_MAX_PEERS; ++index) {
		if (table->entries[index].occupied != 0U && table->entries[index].peer_id == peer_id) {
			return &table->entries[index];
		}
	}

	return NULL;
}

size_t pulse_peer_count(const pulse_peer_table_t* table) {
	size_t count = 0U;
	size_t index;

	if (table == NULL) {
		return 0U;
	}

	for (index = 0U; index < PULSE_MAX_PEERS; ++index) {
		if (table->entries[index].occupied != 0U) {
			++count;
		}
	}

	return count;
}

pulse_status_t pulse_peer_select(pulse_peer_table_t* table,
								 const uint64_t* candidate_ids,
								 size_t candidate_count,
								 pulse_peer_selection_t* selection) {
	double log_term;
	double best_index;
	size_t best_tie_count = 0U;
	size_t selected_tie_rank = 0U;
	size_t selected_candidate = 0U;
	size_t first;
	size_t second;

	if (table == NULL || selection == NULL || (candidate_count > 0U && candidate_ids == NULL) ||
		candidate_count > PULSE_MAX_PEERS) {
		return PULSE_STATUS_BAD_ARGUMENT;
	}

	selection->action = PULSE_PEER_SELECTION_NO_CANDIDATES;
	selection->peer_id = 0U;
	selection->candidate_index = 0U;
	selection->ucb_index = 0.0f;

	if (candidate_count == 0U) {
		return PULSE_STATUS_OK;
	}

	for (first = 0U; first < candidate_count; ++first) {
		for (second = first + 1U; second < candidate_count; ++second) {
			if (candidate_ids[first] == candidate_ids[second]) {
				return PULSE_STATUS_BAD_ARGUMENT;
			}
		}
	}

	log_term = log(2.0 + (double) table->total_selection_opportunities);
	best_index = peer_index(table, candidate_ids[0], log_term);

	for (first = 1U; first < candidate_count; ++first) {
		double index = peer_index(table, candidate_ids[first], log_term);

		if (index > best_index) {
			best_index = index;
		}
	}

	if (table->total_selection_opportunities < UINT32_MAX) {
		++table->total_selection_opportunities;
	}

	/* A tie at or below the no-contact index is deliberately resolved conservatively. */
	if (best_index <= 0.0) {
		selection->action = PULSE_PEER_SELECTION_NO_CONTACT;
		selection->ucb_index = (float) best_index;
		return PULSE_STATUS_OK;
	}

	for (first = 0U; first < candidate_count; ++first) {
		double index = peer_index(table, candidate_ids[first], log_term);

		if (fabs(index - best_index) <= PULSE_UCB_TIE_EPSILON) {
			++best_tie_count;
		}
	}

	if (best_tie_count > 1U) {
		selected_tie_rank = (size_t) bounded_random(table, (uint32_t) best_tie_count);
	}

	best_tie_count = 0U;
	for (first = 0U; first < candidate_count; ++first) {
		double index = peer_index(table, candidate_ids[first], log_term);

		if (fabs(index - best_index) <= PULSE_UCB_TIE_EPSILON) {
			if (best_tie_count == selected_tie_rank) {
				selected_candidate = first;
				break;
			}
			++best_tie_count;
		}
	}

	selection->action = PULSE_PEER_SELECTION_CONTACT;
	selection->peer_id = candidate_ids[selected_candidate];
	selection->candidate_index = selected_candidate;
	selection->ucb_index = (float) best_index;
	return PULSE_STATUS_OK;
}

pulse_status_t pulse_peer_observe(pulse_peer_table_t* table,
								  uint64_t peer_id,
								  float agreement,
								  float utility_momentum) {
	pulse_peer_entry_t* entry;
	float previous;
	float updated;

	if (table == NULL || !isfinite(agreement) || agreement < -1.0f || agreement > 1.0f ||
		!isfinite(utility_momentum) || utility_momentum <= 0.0f || utility_momentum > 1.0f) {
		return PULSE_STATUS_BAD_ARGUMENT;
	}

	entry = find_mutable(table, peer_id);
	if (entry == NULL) {
		entry = find_free(table);
		if (entry == NULL) {
			return PULSE_STATUS_CAPACITY;
		}

		entry->peer_id = peer_id;
		entry->utility = table->initial_utility;
		entry->visits = 0U;
		entry->occupied = 1U;
	}

	previous = entry->utility;
	updated = (1.0f - utility_momentum) * previous + utility_momentum * agreement;
	if (!isfinite(updated)) {
		return PULSE_STATUS_NUMERIC_ERROR;
	}

	entry->utility = updated;
	if (entry->visits < UINT32_MAX) {
		++entry->visits;
	}

	return PULSE_STATUS_OK;
}
