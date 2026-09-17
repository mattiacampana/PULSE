/*
 * Copyright (c) 2026
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pulse_metrics.h"

#include <errno.h>
#include <string.h>

#include <zephyr/sys/printk.h>
#include <zephyr/sys/sys_heap.h>
#include <zephyr/sys/util.h>
#include <zephyr/version.h>

#if defined(CONFIG_SENSWEAR_PULSE_GPIO_MARKERS)
#include <zephyr/drivers/gpio.h>
#include "daughter_if.h"
#endif

#if defined(CONFIG_SYS_HEAP_RUNTIME_STATS) && CONFIG_HEAP_MEM_POOL_SIZE > 0
extern struct k_heap _system_heap;
#endif

#if defined(CONFIG_SENSWEAR_PULSE_UART_PACED_CAPTURE)
#define PULSE_CAPTURE_QUIET_GUARD_MS CONFIG_SENSWEAR_PULSE_CAPTURE_QUIET_GUARD_MS
#else
#define PULSE_CAPTURE_QUIET_GUARD_MS 0U
#endif

BUILD_ASSERT(pulse_marker_stage_Control < 8, "marker stage must fit in three bits");

static const char* const role_names[] = {
	[pulse_metric_role_Local] = "local",
	[pulse_metric_role_Initiator] = "initiator",
	[pulse_metric_role_Responder] = "responder",
};

static const char* const path_names[] = {
	[pulse_event_path_Local] = "local",
	[pulse_event_path_NoContact] = "no_contact",
	[pulse_event_path_AcceptedConnected] = "accepted_connected",
	[pulse_event_path_AcceptedDiscovery] = "accepted_discovery",
	[pulse_event_path_ScanOnly] = "scan_only",
	[pulse_event_path_Failed] = "failed",
};

static const char* const failure_names[] = {
	[pulse_failure_None] = "none",
	[pulse_failure_Timeout] = "timeout",
	[pulse_failure_Disconnected] = "disconnected",
	[pulse_failure_Protocol] = "protocol",
	[pulse_failure_Crc] = "crc",
	[pulse_failure_ModelMismatch] = "model_mismatch",
	[pulse_failure_Resource] = "resource",
	[pulse_failure_Cancelled] = "cancelled",
	[pulse_failure_Internal] = "internal",
};

static const char* const connection_names[] = {
	[pulse_connection_start_NotApplicable] = "na",
	[pulse_connection_start_Connected] = "connected",
	[pulse_connection_start_Disconnected] = "disconnected",
};

static const char* const stage_names[] = {
	[pulse_stage_LocalStep1Encoder] = "local_step_1_encoder",
	[pulse_stage_LocalStep1HeadBackward] = "local_step_1_head_forward_backward",
	[pulse_stage_LocalStep1Update] = "local_step_1_update",
	[pulse_stage_LocalStep2Encoder] = "local_step_2_encoder",
	[pulse_stage_LocalStep2HeadBackward] = "local_step_2_head_forward_backward",
	[pulse_stage_LocalStep2Update] = "local_step_2_update",
	[pulse_stage_PeerSelection] = "peer_selection",
	[pulse_stage_HeadSerialization] = "head_serialization",
	[pulse_stage_HeadTransfer] = "head_transfer",
	[pulse_stage_HeadDeserialization] = "head_deserialization",
	[pulse_stage_ResponderEncoder] = "responder_encoder",
	[pulse_stage_ResponderHeadBackward] = "responder_head_forward_backward",
	[pulse_stage_GradientSerialization] = "gradient_serialization",
	[pulse_stage_GradientTransfer] = "gradient_transfer",
	[pulse_stage_GradientDeserialization] = "gradient_deserialization",
	[pulse_stage_InitiatorEncoder] = "initiator_encoder",
	[pulse_stage_InitiatorHeadBackward] = "initiator_head_forward_backward",
	[pulse_stage_Agreement] = "agreement",
	[pulse_stage_Normalization] = "normalization",
	[pulse_stage_Mixing] = "mixing",
	[pulse_stage_UtilityUpdate] = "utility_update",
	[pulse_stage_BleScan] = "ble_scan",
	[pulse_stage_BleConnect] = "ble_connect",
	[pulse_stage_EndToEnd] = "end_to_end",
};

static struct pulse_metric_record metric_records[CONFIG_SENSWEAR_PULSE_METRIC_RECORDS]
	__attribute__((section(".pulse.metrics")));
static size_t metric_record_count;
static K_MUTEX_DEFINE(metric_lock);

static bool marker_ready;
#if defined(CONFIG_SENSWEAR_PULSE_GPIO_MARKERS)
static const struct gpio_dt_spec* marker_pins[daughter_if_gpio_count];
static const struct device* marker_port;
static gpio_port_pins_t marker_mask;
#endif

static const char* safe_string(const char* value) {
	return value != NULL ? value : "unknown";
}

static const char* failure_name(enum pulse_failure_reason failure) {
	if ((unsigned int) failure >= ARRAY_SIZE(failure_names) || failure_names[failure] == NULL) {
		return "invalid";
	}
	return failure_names[failure];
}

static const char* connection_name(enum pulse_connection_start connection) {
	if ((unsigned int) connection >= ARRAY_SIZE(connection_names) ||
		connection_names[connection] == NULL) {
		return "invalid";
	}
	return connection_names[connection];
}

static enum pulse_marker_stage marker_for_stage(enum pulse_stage stage) {
	switch (stage) {
	case pulse_stage_PeerSelection:
		return pulse_marker_stage_Selection;
	case pulse_stage_LocalStep1Encoder:
	case pulse_stage_LocalStep2Encoder:
	case pulse_stage_ResponderEncoder:
	case pulse_stage_InitiatorEncoder:
		return pulse_marker_stage_Encoder;
	case pulse_stage_LocalStep1HeadBackward:
	case pulse_stage_LocalStep2HeadBackward:
	case pulse_stage_ResponderHeadBackward:
	case pulse_stage_InitiatorHeadBackward:
		return pulse_marker_stage_HeadBackward;
	case pulse_stage_HeadSerialization:
	case pulse_stage_HeadDeserialization:
	case pulse_stage_GradientSerialization:
	case pulse_stage_GradientDeserialization:
		return pulse_marker_stage_Serialization;
	case pulse_stage_HeadTransfer:
	case pulse_stage_GradientTransfer:
		return pulse_marker_stage_Radio;
	case pulse_stage_LocalStep1Update:
	case pulse_stage_LocalStep2Update:
	case pulse_stage_Agreement:
	case pulse_stage_Normalization:
	case pulse_stage_Mixing:
	case pulse_stage_UtilityUpdate:
		return pulse_marker_stage_Mixing;
	case pulse_stage_BleScan:
	case pulse_stage_BleConnect:
		return pulse_marker_stage_Control;
	case pulse_stage_EndToEnd:
	case pulse_stage_Count:
	default:
		return pulse_marker_stage_Event;
	}
}

const char* pulse_metrics_role_name(enum pulse_metric_role role) {
	if ((unsigned int) role >= ARRAY_SIZE(role_names) || role_names[role] == NULL) {
		return "invalid";
	}
	return role_names[role];
}

const char* pulse_metrics_path_name(enum pulse_event_path path) {
	if ((unsigned int) path >= ARRAY_SIZE(path_names) || path_names[path] == NULL) {
		return "invalid";
	}
	return path_names[path];
}

const char* pulse_metrics_stage_name(enum pulse_stage stage) {
	if ((unsigned int) stage >= ARRAY_SIZE(stage_names) || stage_names[stage] == NULL) {
		return "invalid";
	}
	return stage_names[stage];
}

static void marker_write(bool event_active, enum pulse_marker_stage stage) {
#if defined(CONFIG_SENSWEAR_PULSE_GPIO_MARKERS)
	if (!marker_ready) {
		return;
	}

	gpio_port_value_t value = event_active ? BIT(marker_pins[daughter_if_GPIO0]->pin) : 0U;

	if (event_active) {
		if (((uint8_t) stage & BIT(0)) != 0U) {
			value |= BIT(marker_pins[daughter_if_GPIO1]->pin);
		}
		if (((uint8_t) stage & BIT(1)) != 0U) {
			value |= BIT(marker_pins[daughter_if_GPIO2]->pin);
		}
		if (((uint8_t) stage & BIT(2)) != 0U) {
			value |= BIT(marker_pins[daughter_if_GPIO3]->pin);
		}
	}

	(void) gpio_port_set_masked_raw(marker_port, marker_mask, value);
#else
	ARG_UNUSED(event_active);
	ARG_UNUSED(stage);
#endif
}

int pulse_metrics_init(void) {
	timing_init();
	timing_start();

#if defined(CONFIG_SENSWEAR_PULSE_GPIO_MARKERS)
	int ret = 0;
	unsigned int claimed = 0U;

	for (unsigned int i = 0U; i < daughter_if_gpio_count; ++i) {
		marker_pins[i] = daughter_if_gpio_claim((enum daughter_if_gpio) i);
		if (marker_pins[i] == NULL) {
			ret = -EBUSY;
			break;
		}
		claimed++;

		if (marker_port == NULL) {
			marker_port = marker_pins[i]->port;
		} else if (marker_port != marker_pins[i]->port) {
			ret = -ENOTSUP;
			break;
		}

		ret = gpio_pin_configure_dt(marker_pins[i], GPIO_OUTPUT_INACTIVE);
		if (ret != 0) {
			break;
		}
		marker_mask |= BIT(marker_pins[i]->pin);
	}

	if (ret != 0) {
		for (unsigned int i = 0U; i < claimed; ++i) {
			(void) daughter_if_gpio_release((enum daughter_if_gpio) i);
			marker_pins[i] = NULL;
		}
		marker_port = NULL;
		marker_mask = 0U;
		marker_ready = false;
		return ret;
	}

	marker_ready = true;
	marker_write(false, pulse_marker_stage_Event);
#else
	marker_ready = false;
#endif
	return 0;
}

bool pulse_metrics_marker_available(void) {
	return marker_ready;
}

uint64_t pulse_metrics_counter_frequency_hz(void) {
	return timing_freq_get();
}

void pulse_metrics_record_reset(struct pulse_metric_record* record) {
	if (record != NULL) {
		memset(record, 0, sizeof(*record));
		/* Zero is a valid RSSI. Use the BLE unavailable sentinel until either
		 * an advertisement or a connected-link sample supplies a value. */
		record->rssi_dbm = -127;
	}
}

void pulse_metrics_event_begin(struct pulse_metric_record* record,
							   enum pulse_event_path path,
							   enum pulse_metric_role role,
							   enum pulse_connection_start connection_start,
							   uint32_t trial_id,
							   uint32_t exchange_id) {
	if (record == NULL) {
		return;
	}

	pulse_metrics_record_reset(record);
	record->trial_id = trial_id;
	record->exchange_id = exchange_id;
	record->event_path = path;
	record->role = role;
	record->connection_start = connection_start;
	record->failure_reason = pulse_failure_None;
	marker_write(true, pulse_marker_stage_Event);
	record->event_wall_start = k_cycle_get_64();
	record->event_start = timing_counter_get();
	record->stages[pulse_stage_EndToEnd].start = record->event_start;
	record->stages[pulse_stage_EndToEnd].wall_start = record->event_wall_start;
}

void pulse_metrics_event_end(struct pulse_metric_record* record,
							 bool success,
							 enum pulse_failure_reason failure_reason) {
	if (record == NULL) {
		return;
	}

	record->event_end = timing_counter_get();
	record->event_wall_end = k_cycle_get_64();
	record->stages[pulse_stage_EndToEnd].end = record->event_end;
	record->stages[pulse_stage_EndToEnd].wall_end = record->event_wall_end;
	record->success = success;
	record->failure_reason = failure_reason;
	record->timeout = failure_reason == pulse_failure_Timeout;
	marker_write(false, pulse_marker_stage_Event);
}

void pulse_metrics_stage_begin(struct pulse_metric_record* record,
							   enum pulse_stage stage,
							   enum pulse_marker_stage marker) {
	if (record == NULL || stage >= pulse_stage_Count) {
		return;
	}
	marker_write(true, marker);
	record->stages[stage].wall_start = k_cycle_get_64();
	record->stages[stage].start = timing_counter_get();
}

void pulse_metrics_stage_end(struct pulse_metric_record* record, enum pulse_stage stage) {
	if (record == NULL || stage >= pulse_stage_Count) {
		return;
	}
	record->stages[stage].end = timing_counter_get();
	record->stages[stage].wall_end = k_cycle_get_64();
	marker_write(true, pulse_marker_stage_Event);
}

int pulse_metrics_store(const struct pulse_metric_record* record) {
	if (record == NULL) {
		return -EINVAL;
	}

	k_mutex_lock(&metric_lock, K_FOREVER);
	if (metric_record_count >= ARRAY_SIZE(metric_records)) {
		k_mutex_unlock(&metric_lock);
		return -ENOSPC;
	}
	metric_records[metric_record_count++] = *record;
	k_mutex_unlock(&metric_lock);
	return 0;
}

static uint64_t interval_cycles(timing_t start, timing_t end) {
	if (start == 0U || end == 0U) {
		return 0U;
	}
	return timing_cycles_get(&start, &end);
}

static uint64_t cycles_to_us(uint64_t cycles) {
	return timing_cycles_to_ns(cycles) / 1000U;
}

static uint64_t wall_interval_us(uint64_t start, uint64_t end) {
	if (start == 0U || end == 0U || end < start) {
		return 0U;
	}
	return k_cyc_to_us_floor64(end - start);
}

void pulse_metrics_dump_and_reset(const char* board_id) {
	const char* id = safe_string(board_id);

	k_mutex_lock(&metric_lock, K_FOREVER);
	for (size_t i = 0U; i < metric_record_count; ++i) {
		const struct pulse_metric_record* record = &metric_records[i];
		uint64_t event_cycles = interval_cycles(record->event_start, record->event_end);
		uint64_t event_wall_us = wall_interval_us(record->event_wall_start, record->event_wall_end);

		printk("PULSE_EVENT,board_id=%s,event_index=%u,warmup=%u,trial_id=%u,"
			   "exchange_id=%u,role=%s,"
			   "event_path=%s,connection_state_start=%s,success=%u,failure_reason=%s,"
			   "timeout=%u,remote_session_started=%u,input_batch_id=%u,"
			   "event_start_counter=%llu,event_end_counter=%llu,"
			   "cycles=%llu,cycle_duration_us=%llu,wall_start_counter=%llu,"
			   "wall_end_counter=%llu,wall_duration_us=%llu,duration_us=%llu,"
			   "capture_envelope_duration_us=%llu,"
			   "reachable_peer_count=%u,"
			   "selection_opportunity_before=%u,selection_opportunity_after=%u,"
			   "peer_utility_before=%.9g,peer_utility_after=%.9g,peer_visits_before=%u,"
			   "peer_visits_after=%u,ucb_index=%.9g,local_gradient_norm=%.9g,"
			   "remote_gradient_norm=%.9g,gradient_dot=%.9g,agreement=%.9g,"
			   "normalization_scale=%.9g,mixing_alpha=%.9g,head_payload_bytes=%u,"
			   "gradient_payload_bytes=%u,application_tx_bytes=%u,application_rx_bytes=%u,"
			   "protocol_header_tx_bytes=%u,"
			   "protocol_header_rx_bytes=%u,att_tx_bytes=%u,att_rx_bytes=%u,"
			   "tx_chunks=%u,rx_chunks=%u,link_layer_tx_bytes=-1,link_layer_rx_bytes=-1,"
			   "pulse_framing_tx_bytes=%u,pulse_framing_rx_bytes=%u,"
			   "att_value_tx_bytes=%u,att_value_rx_bytes=%u,"
			   "att_value_tx_fragments=%u,att_value_rx_fragments=%u,"
			   "retransmissions=-1,rssi_dbm=%d,att_mtu=%u,data_length=%u,tx_phy=%u,"
			   "rx_phy=%u,connection_interval_units=%u,initial_head_hash=%08x,"
			   "result_head_hash=%08x,head_version_before=%u,head_version_after=%u,"
			   "thread_stack_peak_bytes=%u,system_heap_peak_bytes=%u\n",
			   id,
			   record->event_index,
			   record->warmup ? 1U : 0U,
			   record->trial_id,
			   record->exchange_id,
			   pulse_metrics_role_name(record->role),
			   pulse_metrics_path_name(record->event_path),
			   connection_name(record->connection_start),
			   record->success ? 1U : 0U,
			   failure_name(record->failure_reason),
			   record->timeout ? 1U : 0U,
			   record->remote_session_started ? 1U : 0U,
			   record->input_batch_id,
			   (unsigned long long) record->event_start,
			   (unsigned long long) record->event_end,
			   (unsigned long long) event_cycles,
			   (unsigned long long) cycles_to_us(event_cycles),
			   (unsigned long long) record->event_wall_start,
			   (unsigned long long) record->event_wall_end,
			   (unsigned long long) event_wall_us,
			   (unsigned long long) event_wall_us,
			   (unsigned long long) record->capture_envelope_duration_us,
			   record->reachable_peers,
			   record->selection_opportunities_before,
			   record->selection_opportunities_after,
			   (double) record->utility_before,
			   (double) record->utility_after,
			   record->peer_visits_before,
			   record->peer_visits_after,
			   (double) record->ucb_index,
			   (double) record->local_gradient_norm,
			   (double) record->remote_gradient_norm,
			   (double) record->gradient_dot,
			   (double) record->agreement,
			   (double) record->normalization_scale,
			   (double) record->mixing_alpha,
			   record->head_payload_bytes,
			   record->gradient_payload_bytes,
			   record->application_tx_bytes,
			   record->application_rx_bytes,
			   record->protocol_tx_bytes,
			   record->protocol_rx_bytes,
			   record->att_tx_bytes,
			   record->att_rx_bytes,
			   record->tx_chunks,
			   record->rx_chunks,
			   record->protocol_tx_bytes,
			   record->protocol_rx_bytes,
			   record->att_tx_bytes,
			   record->att_rx_bytes,
			   record->tx_chunks,
			   record->rx_chunks,
			   record->rssi_dbm,
			   record->att_mtu,
			   record->data_length,
			   record->tx_phy,
			   record->rx_phy,
			   record->connection_interval_units,
			   record->initial_head_hash,
			   record->result_head_hash,
			   record->head_version_before,
			   record->head_version_after,
			   record->stack_peak_bytes,
			   record->system_heap_peak_bytes);

		for (enum pulse_stage stage = 0; stage < pulse_stage_Count; ++stage) {
			uint64_t cycles =
				interval_cycles(record->stages[stage].start, record->stages[stage].end);
			uint64_t wall_us =
				wall_interval_us(record->stages[stage].wall_start, record->stages[stage].wall_end);

			if (cycles == 0U && wall_us == 0U) {
				continue;
			}
			printk("PULSE_STAGE,board_id=%s,event_index=%u,warmup=%u,trial_id=%u,"
				   "exchange_id=%u,role=%s,"
				   "event_path=%s,connection_state_start=%s,success=%u,"
				   "stage_code=%u,stage=%s,start_counter=%llu,end_counter=%llu,"
				   "cycles=%llu,cycle_duration_us=%llu,wall_start_counter=%llu,"
				   "wall_end_counter=%llu,wall_duration_us=%llu,duration_us=%llu\n",
				   id,
				   record->event_index,
				   record->warmup ? 1U : 0U,
				   record->trial_id,
				   record->exchange_id,
				   pulse_metrics_role_name(record->role),
				   pulse_metrics_path_name(record->event_path),
				   connection_name(record->connection_start),
				   record->success ? 1U : 0U,
				   (unsigned int) marker_for_stage(stage),
				   pulse_metrics_stage_name(stage),
				   (unsigned long long) record->stages[stage].start,
				   (unsigned long long) record->stages[stage].end,
				   (unsigned long long) cycles,
				   (unsigned long long) cycles_to_us(cycles),
				   (unsigned long long) record->stages[stage].wall_start,
				   (unsigned long long) record->stages[stage].wall_end,
				   (unsigned long long) wall_us,
				   (unsigned long long) wall_us);
		}
	}
	metric_record_count = 0U;
	k_mutex_unlock(&metric_lock);
}

void pulse_metrics_dump_metadata(const struct pulse_run_metadata* metadata, const char* board_id) {
	if (metadata == NULL) {
		return;
	}

	printk("PULSE_META,run_id=%s,pair_id=%s,board_id=%s,role=%s,build_variant=%s,build_guard=%s,"
		   "firmware_revision=%s,"
		   "hardware_revision=%s,ncs_version=%s,zephyr_version=%s,compiler=%s,"
		   "optimization=%s,model_id=%s,model_version=%u,encoder_hash=%s,"
		   "fixture_hash=%s,artifact_kind=%s,artifact_source=%s,artifact_sha256=%s,"
		   "task=HAR,class_count=%u,parameter_count=%u,activation=relu,"
		   "loss_reduction=mean,tensor_dtype=float32,serialized_bytes_per_parameter=%u,"
		   "head_serialized_bytes=%u,mutable_head_bytes=%u,compute_workspace_bytes=%u,"
		   "gradient_buffer_bytes_each=%u,gradient_buffer_count=2,received_head_bytes=%u,"
		   "utility_table_bytes=%u,peer_entry_bytes=%u,cpu_clock_hz=%u,"
		   "timing_counter_hz=%llu,wall_counter_hz=%u,batch_size=%u,"
		   "local_steps=%u,learning_rate=%.9g,alpha_max=%.9g,utility_mu=%.9g,"
		   "initial_utility=%.9g,ucb_beta=%.9g,epsilon=%.9g,peer_limit=%u,"
		   "tie_break_policy=%s,trial_state_policy=%s,input_mode=%s,enabled_sensors=%s,"
		   "warmup_repetitions=%u,"
		   "measured_repetitions=%u,timeout_ms=%u,discovery_settle_ms=%u,tx_power_dbm=%d,"
		   "initial_rssi_dbm=%d,"
		   "initial_att_mtu=%u,initial_data_length=%u,initial_tx_phy=%u,initial_rx_phy=%u,"
		   "initial_connection_interval_units=%u,security_state=%s,"
		   "local_identity_address=%s,local_identity_address_type=%s,"
		   "peer_identity_address=%s,peer_identity_address_type=%s,"
		   "execution_context=dedicated_pulse_event_workqueue_plus_constant_time_bt_callbacks,"
		   "cycle_timing_semantics=dwt_active_core_counter_includes_preemption_and_interrupts,"
		   "duration_us_semantics=zephyr_64bit_api_grtc_syscounter_includes_sleep,"
		   "stack_watermark_scope=event_workqueue_cumulative_since_start_excludes_bt_rx_and_uart,"
		   "serialization_timing_semantics=full_tensor_fp32le_plus_crc,"
		   "transfer_timing_semantics=async_att_wall_plus_incremental_wire_crc,"
		   "deserialization_timing_semantics=post_transfer_fp32le_decode,"
		   "responder_gradient_endpoint=crc_validated_initiator_receipt_ack_at_responder,"
		   "att_value_counter_semantics=endpoint_submitted_or_accepted_not_over_air,"
		   "communication_accounting=firmware_att_value_boundary,"
		   "result_release_semantics=post_event_export_handshake_excluded_from_protocol_counters,"
		   "link_layer_accounting=not_measured_no_ble_sniffer,"
		   "capture_pacing=%s,capture_quiet_guard_ms=%u,capture_min_event_period_ms=%u,"
		   "capture_envelope_semantics=%s,"
		   "power_alignment=%s,"
		   "event_workqueue_stack_capacity_bytes=%u,system_heap_capacity_bytes=%u,"
		   "marker_available=%u,marker_event_gate=%s,marker_bit0=%s,"
		   "marker_bit1=%s,marker_bit2=%s,marker_stage_0=event,"
		   "marker_stage_1=selection,marker_stage_2=encoder,"
		   "marker_stage_3=head_forward_backward,"
		   "marker_stage_4=serialization,marker_stage_5=radio,marker_stage_6=mixing,"
		   "marker_stage_7=control,device_separation_m=operator_capture_context\n",
		   safe_string(metadata->run_id),
		   safe_string(metadata->pair_id),
		   safe_string(board_id),
		   safe_string(metadata->role),
		   safe_string(metadata->build_variant),
		   safe_string(metadata->build_guard),
		   safe_string(metadata->firmware_revision),
		   safe_string(metadata->hardware_revision),
		   STRINGIFY(BUILD_VERSION),
		   KERNEL_VERSION_STRING,
		   __VERSION__,
#if defined(CONFIG_SPEED_OPTIMIZATIONS)
		   "speed",
#elif defined(CONFIG_SIZE_OPTIMIZATIONS)
		   "size",
#else
		   "debug_or_default",
#endif
		   safe_string(metadata->model_id),
		   metadata->model_version,
		   safe_string(metadata->encoder_hash),
		   safe_string(metadata->fixture_hash),
		   safe_string(metadata->artifact_kind),
		   safe_string(metadata->artifact_source),
		   safe_string(metadata->artifact_sha256),
		   metadata->class_count,
		   metadata->parameter_count,
		   metadata->bytes_per_parameter,
		   metadata->head_bytes,
		   metadata->head_bytes,
		   metadata->compute_workspace_bytes,
		   metadata->gradient_buffer_bytes,
		   metadata->received_head_bytes,
		   metadata->utility_table_bytes,
		   metadata->peer_entry_bytes,
		   metadata->cpu_clock_hz,
		   (unsigned long long) pulse_metrics_counter_frequency_hz(),
		   CONFIG_SYS_CLOCK_HW_CYCLES_PER_SEC,
		   metadata->batch_size,
		   metadata->local_steps,
		   (double) metadata->learning_rate,
		   (double) metadata->alpha_max,
		   (double) metadata->utility_momentum,
		   (double) metadata->initial_utility,
		   (double) metadata->ucb_exploration,
		   (double) metadata->norm_epsilon,
		   metadata->peer_limit,
		   safe_string(metadata->tie_break_policy),
		   safe_string(metadata->trial_state_policy),
		   safe_string(metadata->input_mode),
		   safe_string(metadata->enabled_sensors),
		   metadata->warmup_repetitions,
		   metadata->measured_repetitions,
		   metadata->timeout_ms,
		   CONFIG_SENSWEAR_PULSE_DISCOVERY_SETTLE_MS,
		   metadata->tx_power_dbm,
		   metadata->initial_rssi_dbm,
		   metadata->initial_att_mtu,
		   metadata->initial_data_length,
		   metadata->initial_tx_phy,
		   metadata->initial_rx_phy,
		   metadata->initial_connection_interval_units,
		   safe_string(metadata->security_state),
		   safe_string(metadata->local_identity_address),
		   safe_string(metadata->local_identity_address_type),
		   safe_string(metadata->peer_identity_address),
		   safe_string(metadata->peer_identity_address_type),
		   IS_ENABLED(CONFIG_SENSWEAR_PULSE_UART_PACED_CAPTURE)
			   ? "uart_ready_go_dump"
			   : (IS_ENABLED(CONFIG_SENSWEAR_PULSE_KEYSIGHT_FIG4A_CAPTURE)
					  ? "keysight_current_level_trigger_fixed_period"
					  : "automatic_campaign"),
		   PULSE_CAPTURE_QUIET_GUARD_MS,
		   CONFIG_SENSWEAR_PULSE_MIN_EVENT_PERIOD_MS,
		   IS_ENABLED(CONFIG_SENSWEAR_PULSE_UART_PACED_CAPTURE)
			   ? "after_go_guard_through_single_trial_cleanup_before_dump_guard"
			   : (IS_ENABLED(CONFIG_SENSWEAR_PULSE_KEYSIGHT_FIG4A_CAPTURE)
					  ? "current_level_pretrigger_and_4.1s_posttrigger_dmm_window"
					  : "mode_specific_no_uart_capture_envelope"),
		   IS_ENABLED(CONFIG_SENSWEAR_PULSE_UART_PACED_CAPTURE)
			   ? "ngmo2_shared_software_trigger_capture_envelope"
			   : (IS_ENABLED(CONFIG_SENSWEAR_PULSE_KEYSIGHT_FIG4A_CAPTURE)
					  ? "keysight_internal_current_level_fixed_window"
					  : "optional_gpio_or_firmware_timing"),
		   CONFIG_SENSWEAR_PULSE_EVENT_WORKQUEUE_STACK_SIZE,
		   CONFIG_HEAP_MEM_POOL_SIZE,
		   pulse_metrics_marker_available() ? 1U : 0U,
		   pulse_metrics_marker_available() ? "P1.14" : "disabled",
		   pulse_metrics_marker_available() ? "P1.13" : "disabled",
		   pulse_metrics_marker_available() ? "P1.12" : "disabled",
		   pulse_metrics_marker_available() ? "P1.11" : "disabled");
}

uint32_t pulse_metrics_thread_stack_peak(struct k_thread* thread, size_t stack_size) {
	size_t unused = 0U;

	if (thread == NULL || k_thread_stack_space_get(thread, &unused) != 0 || unused > stack_size) {
		return 0U;
	}
	return (uint32_t) (stack_size - unused);
}

void pulse_metrics_system_heap_peak_reset(void) {
#if defined(CONFIG_SYS_HEAP_RUNTIME_STATS) && CONFIG_HEAP_MEM_POOL_SIZE > 0
	/* The benchmark calls this only between events, when its own heap use is quiescent. */
	(void) sys_heap_runtime_stats_reset_max(&_system_heap.heap);
#endif
}

uint32_t pulse_metrics_system_heap_peak_bytes(void) {
#if defined(CONFIG_SYS_HEAP_RUNTIME_STATS) && CONFIG_HEAP_MEM_POOL_SIZE > 0
	struct sys_memory_stats stats;

	if (sys_heap_runtime_stats_get(&_system_heap.heap, &stats) == 0) {
		return stats.max_allocated_bytes > UINT32_MAX ? UINT32_MAX
													  : (uint32_t) stats.max_allocated_bytes;
	}
#endif
	return 0U;
}

uint32_t pulse_metrics_hash32(const void* data, size_t length) {
	const uint8_t* bytes = data;
	uint32_t hash = 2166136261U;

	if (data == NULL) {
		return 0U;
	}
	for (size_t i = 0U; i < length; ++i) {
		hash ^= bytes[i];
		hash *= 16777619U;
	}
	return hash;
}
