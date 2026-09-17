/*
 * Copyright (c) 2026
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef SENSWEAR_PULSE_METRICS_H_
#define SENSWEAR_PULSE_METRICS_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <zephyr/kernel.h>
#include <zephyr/timing/timing.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Keep these names stable: tools/pulse uses them as the public result schema. */
enum pulse_metric_role {
	pulse_metric_role_Local = 0,
	pulse_metric_role_Initiator,
	pulse_metric_role_Responder,
};

enum pulse_event_path {
	pulse_event_path_Local = 0,
	pulse_event_path_NoContact,
	pulse_event_path_AcceptedConnected,
	pulse_event_path_AcceptedDiscovery,
	pulse_event_path_ScanOnly,
	pulse_event_path_Failed,
};

enum pulse_connection_start {
	pulse_connection_start_NotApplicable = 0,
	pulse_connection_start_Connected,
	pulse_connection_start_Disconnected,
};

enum pulse_failure_reason {
	pulse_failure_None = 0,
	pulse_failure_Timeout,
	pulse_failure_Disconnected,
	pulse_failure_Protocol,
	pulse_failure_Crc,
	pulse_failure_ModelMismatch,
	pulse_failure_Resource,
	pulse_failure_Cancelled,
	pulse_failure_Internal,
};

enum pulse_stage {
	pulse_stage_LocalStep1Encoder = 0,
	pulse_stage_LocalStep1HeadBackward,
	pulse_stage_LocalStep1Update,
	pulse_stage_LocalStep2Encoder,
	pulse_stage_LocalStep2HeadBackward,
	pulse_stage_LocalStep2Update,
	pulse_stage_PeerSelection,
	pulse_stage_HeadSerialization,
	pulse_stage_HeadTransfer,
	pulse_stage_HeadDeserialization,
	pulse_stage_ResponderEncoder,
	pulse_stage_ResponderHeadBackward,
	pulse_stage_GradientSerialization,
	pulse_stage_GradientTransfer,
	pulse_stage_GradientDeserialization,
	pulse_stage_InitiatorEncoder,
	pulse_stage_InitiatorHeadBackward,
	pulse_stage_Agreement,
	pulse_stage_Normalization,
	pulse_stage_Mixing,
	pulse_stage_UtilityUpdate,
	pulse_stage_BleScan,
	pulse_stage_BleConnect,
	pulse_stage_EndToEnd,
	pulse_stage_Count,
};

/* The event gate is GPIO0. GPIO1..3 carry this three-bit code, LSB first. */
enum pulse_marker_stage {
	pulse_marker_stage_Event = 0,
	pulse_marker_stage_Selection = 1,
	pulse_marker_stage_Encoder = 2,
	pulse_marker_stage_HeadBackward = 3,
	pulse_marker_stage_Serialization = 4,
	pulse_marker_stage_Radio = 5,
	pulse_marker_stage_Mixing = 6,
	pulse_marker_stage_Control = 7,
};

struct pulse_stage_timing {
	timing_t start;
	timing_t end;
	uint64_t wall_start;
	uint64_t wall_end;
};

struct pulse_metric_record {
	uint32_t event_index;
	uint32_t trial_id;
	uint32_t exchange_id;
	bool warmup;
	enum pulse_metric_role role;
	enum pulse_event_path event_path;
	enum pulse_connection_start connection_start;
	enum pulse_failure_reason failure_reason;
	bool success;
	bool timeout;
	bool remote_session_started;
	uint8_t input_batch_id;
	uint8_t reachable_peers;
	uint32_t head_version_before;
	uint32_t head_version_after;
	uint32_t selection_opportunities_before;
	uint32_t selection_opportunities_after;
	uint32_t peer_visits_before;
	uint32_t peer_visits_after;
	float ucb_index;
	float utility_before;
	float utility_after;
	float local_gradient_norm;
	float remote_gradient_norm;
	float gradient_dot;
	float agreement;
	float normalization_scale;
	float mixing_alpha;
	uint32_t head_payload_bytes;
	uint32_t gradient_payload_bytes;
	uint32_t application_tx_bytes;
	uint32_t application_rx_bytes;
	uint32_t protocol_tx_bytes;
	uint32_t protocol_rx_bytes;
	uint32_t att_tx_bytes;
	uint32_t att_rx_bytes;
	uint16_t tx_chunks;
	uint16_t rx_chunks;
	int8_t rssi_dbm;
	uint16_t att_mtu;
	uint16_t data_length;
	uint16_t connection_interval_units;
	uint8_t tx_phy;
	uint8_t rx_phy;
	uint32_t initial_head_hash;
	uint32_t result_head_hash;
	uint32_t stack_peak_bytes;
	uint32_t system_heap_peak_bytes;
	uint64_t capture_envelope_duration_us;
	timing_t event_start;
	timing_t event_end;
	uint64_t event_wall_start;
	uint64_t event_wall_end;
	struct pulse_stage_timing stages[pulse_stage_Count];
};

struct pulse_run_metadata {
	const char* run_id;
	const char* pair_id;
	const char* role;
	const char* build_variant;
	const char* build_guard;
	const char* firmware_revision;
	const char* hardware_revision;
	const char* model_id;
	const char* encoder_hash;
	const char* fixture_hash;
	const char* artifact_kind;
	const char* artifact_source;
	const char* artifact_sha256;
	const char* input_mode;
	const char* enabled_sensors;
	const char* tie_break_policy;
	const char* trial_state_policy;
	const char* security_state;
	const char* local_identity_address;
	const char* local_identity_address_type;
	const char* peer_identity_address;
	const char* peer_identity_address_type;
	uint32_t model_version;
	uint32_t cpu_clock_hz;
	uint32_t parameter_count;
	uint32_t head_bytes;
	uint32_t compute_workspace_bytes;
	uint32_t gradient_buffer_bytes;
	uint32_t received_head_bytes;
	uint32_t utility_table_bytes;
	uint16_t peer_entry_bytes;
	uint8_t class_count;
	uint8_t batch_size;
	uint8_t local_steps;
	uint8_t bytes_per_parameter;
	float learning_rate;
	float alpha_max;
	float utility_momentum;
	float initial_utility;
	float ucb_exploration;
	float norm_epsilon;
	uint16_t peer_limit;
	uint16_t warmup_repetitions;
	uint16_t measured_repetitions;
	uint32_t timeout_ms;
	int8_t tx_power_dbm;
	int8_t initial_rssi_dbm;
	uint16_t initial_att_mtu;
	uint16_t initial_data_length;
	uint16_t initial_connection_interval_units;
	uint8_t initial_tx_phy;
	uint8_t initial_rx_phy;
};

int pulse_metrics_init(void);
bool pulse_metrics_marker_available(void);
uint64_t pulse_metrics_counter_frequency_hz(void);

void pulse_metrics_record_reset(struct pulse_metric_record* record);
void pulse_metrics_event_begin(struct pulse_metric_record* record,
							   enum pulse_event_path path,
							   enum pulse_metric_role role,
							   enum pulse_connection_start connection_start,
							   uint32_t trial_id,
							   uint32_t exchange_id);
void pulse_metrics_event_end(struct pulse_metric_record* record,
							 bool success,
							 enum pulse_failure_reason failure_reason);
void pulse_metrics_stage_begin(struct pulse_metric_record* record,
							   enum pulse_stage stage,
							   enum pulse_marker_stage marker);
void pulse_metrics_stage_end(struct pulse_metric_record* record, enum pulse_stage stage);

int pulse_metrics_store(const struct pulse_metric_record* record);
void pulse_metrics_dump_and_reset(const char* board_id);
void pulse_metrics_dump_metadata(const struct pulse_run_metadata* metadata, const char* board_id);

uint32_t pulse_metrics_thread_stack_peak(struct k_thread* thread, size_t stack_size);
void pulse_metrics_system_heap_peak_reset(void);
uint32_t pulse_metrics_system_heap_peak_bytes(void);
uint32_t pulse_metrics_hash32(const void* data, size_t length);

const char* pulse_metrics_role_name(enum pulse_metric_role role);
const char* pulse_metrics_path_name(enum pulse_event_path path);
const char* pulse_metrics_stage_name(enum pulse_stage stage);

#ifdef __cplusplus
}
#endif

#endif /* SENSWEAR_PULSE_METRICS_H_ */
