/*
 * Copyright (c) 2026
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef SENSWEAR_PULSE_CAPTURE_LOG_H_
#define SENSWEAR_PULSE_CAPTURE_LOG_H_

#include <stdbool.h>
#include <stdint.h>

#include "pulse_metrics.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PULSE_CAPTURE_PROOF_MAGIC UINT32_C(0x504e474d)
#define PULSE_CAPTURE_PROOF_SCHEMA_VERSION 2U
#define PULSE_CAPTURE_PROOF_ROLE_UNRESOLVED UINT8_MAX

#define PULSE_CAPTURE_PROOF_FLAG_SUCCESS UINT8_C(0x01)
#define PULSE_CAPTURE_PROOF_FLAG_REMOTE_SESSION_STARTED UINT8_C(0x02)
#define PULSE_CAPTURE_PROOF_FLAG_WARMUP UINT8_C(0x04)
#define PULSE_CAPTURE_PROOF_FLAG_RECORD_READY UINT8_C(0x08)
#define PULSE_CAPTURE_PROOF_FLAG_PASSIVE_EXPECTED UINT8_C(0x10)

/*
 * Deliberately compact: with the board's 16-byte flash write alignment, each
 * proof occupies 48 bytes of CRC-protected NVS data plus one 16-byte ATE.
 * Thus all four 105-boot path blocks fit in the 36 KiB storage partition while
 * retaining the sector NVS reserves for garbage collection.
 */
struct pulse_capture_proof {
	uint32_t magic;
	uint32_t event_duration_us;
	uint32_t initial_head_hash;
	uint32_t result_head_hash;
	uint32_t firmware_revision_hash;
	uint32_t artifact_hash;
	uint16_t exchange_id;
	int16_t event_errno;
	int16_t infrastructure_errno;
	int16_t result_release_errno;
	uint8_t sequence;
	uint8_t trial_id;
	uint8_t schema_version;
	uint8_t record_size;
	uint8_t event_path;
	uint8_t role;
	uint8_t failure_reason;
	uint8_t flags;
} __attribute__((packed, aligned(4)));

static inline bool pulse_capture_proof_flag_get(const struct pulse_capture_proof* proof,
											 uint8_t flag) {
	return proof != NULL && (proof->flags & flag) != 0U;
}

static inline void pulse_capture_proof_flag_set(struct pulse_capture_proof* proof,
											 uint8_t flag,
											 bool value) {
	if (proof == NULL) {
		return;
	}
	if (value) {
		proof->flags |= flag;
	} else {
		proof->flags &= (uint8_t) ~flag;
	}
}

int pulse_capture_log_init(void);
int pulse_capture_log_next_sequence(enum pulse_event_path path, uint32_t* sequence);
int pulse_capture_log_append(const struct pulse_capture_proof* proof);
void pulse_capture_log_dump(const char* board_id);

#ifdef __cplusplus
}
#endif

#endif /* SENSWEAR_PULSE_CAPTURE_LOG_H_ */
