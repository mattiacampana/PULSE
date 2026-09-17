/*
 * Copyright (c) 2026
 * SPDX-License-Identifier: Apache-2.0
 *
 * Reproducible two-board PULSE feasibility application for PERCOM 2026 Section V.
 * Nothing is printed while an internally timed or NGMO2 capture event is active.
 */

#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include <zephyr/drivers/hwinfo.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/atomic.h>
#include <zephyr/sys/printk.h>
#include <zephyr/sys/util.h>

#if defined(CONFIG_SENSWEAR_PULSE_UART_PACED_CAPTURE)
#include <zephyr/device.h>
#include <zephyr/devicetree.h>
#include <zephyr/drivers/uart.h>
#endif

#include "device_manager.h"
#include "pulse_link.h"
#include "pulse_metrics.h"
#if defined(CONFIG_SENSWEAR_PULSE_NGMO2_AUTONOMOUS_CAPTURE)
#include "pulse_capture_log.h"
#endif

#if defined(CONFIG_SENSWEAR_PULSE_ALGORITHM)
#include "pulse_benchmark.h"
#include "pulse_fixture.h"
#include "pulse_protocol.h"
#if defined(CONFIG_SENSWEAR_PULSE_CORRECTNESS)
#include "pulse_correctness.h"
#endif
#endif

#define PULSE_BOARD_ID_BYTES 8U
#define PULSE_BOARD_ID_STRING_SIZE (PULSE_BOARD_ID_BYTES * 2U + 1U)
#define PULSE_BT_ADDRESS_STRING_SIZE 18U
#define PULSE_FALLBACK_PEER_ID UINT64_C(0x50554c534e4f4445)
#define PULSE_STATUS_POLL_MS 5U
#define PULSE_EVENT_WORKQUEUE_PRIORITY (CONFIG_BT_RX_PRIO + 1)

#define APP_EVENT_LINK_CONNECTED BIT(0)
#define APP_EVENT_LINK_DISCONNECTED BIT(1)
#define APP_EVENT_SCAN_FOUND BIT(2)
#define APP_EVENT_CLIENT_DISCOVERED BIT(3)
#define APP_EVENT_CLIENT_HEAD_SENT BIT(4)
#define APP_EVENT_CLIENT_REMOTE_PROCESSING BIT(5)
#define APP_EVENT_CLIENT_GRADIENT_READY BIT(6)
#define APP_EVENT_CLIENT_GRADIENT_RECEIVED BIT(7)
#define APP_EVENT_CLIENT_COMPLETE BIT(8)
#define APP_EVENT_CLIENT_CANCELLED BIT(9)
#define APP_EVENT_CLIENT_ERROR BIT(10)
#define APP_EVENT_SERVER_STARTED BIT(11)
#define APP_EVENT_SERVER_HEAD_READY BIT(12)
#define APP_EVENT_SERVER_GRADIENT_RECEIVED BIT(13)
#define APP_EVENT_SERVER_COMPLETE BIT(14)
#define APP_EVENT_SERVER_CANCELLED BIT(15)
#define APP_EVENT_CLIENT_RESULT_RELEASED BIT(16)
#define APP_EVENT_SERVER_RESULT_RELEASED BIT(17)

#define APP_EVENT_CLIENT_TERMINAL \
	(APP_EVENT_CLIENT_ERROR | APP_EVENT_CLIENT_CANCELLED | APP_EVENT_LINK_DISCONNECTED)
#define APP_EVENT_SERVER_TERMINAL (APP_EVENT_SERVER_CANCELLED | APP_EVENT_LINK_DISCONNECTED)
#define APP_EVENT_PROTOCOL_MASK                                                                    \
	(APP_EVENT_CLIENT_DISCOVERED | APP_EVENT_CLIENT_HEAD_SENT |                                    \
	 APP_EVENT_CLIENT_REMOTE_PROCESSING | APP_EVENT_CLIENT_GRADIENT_READY |                        \
	 APP_EVENT_CLIENT_GRADIENT_RECEIVED | APP_EVENT_CLIENT_COMPLETE | APP_EVENT_CLIENT_CANCELLED | \
	 APP_EVENT_CLIENT_ERROR | APP_EVENT_CLIENT_RESULT_RELEASED | APP_EVENT_SERVER_STARTED |        \
	 APP_EVENT_SERVER_HEAD_READY | APP_EVENT_SERVER_GRADIENT_RECEIVED |                            \
	 APP_EVENT_SERVER_COMPLETE | APP_EVENT_SERVER_CANCELLED | APP_EVENT_SERVER_RESULT_RELEASED)

enum measured_stage_bit {
	MEASURED_STAGE_SCAN = 0,
	MEASURED_STAGE_CONNECT,
	MEASURED_STAGE_HEAD_TRANSFER,
	MEASURED_STAGE_GRADIENT_TRANSFER,
};

extern uint32_t SystemCoreClock;

K_EVENT_DEFINE(app_events);

static char board_id[PULSE_BOARD_ID_STRING_SIZE];
static struct pulse_link link_manager;
static struct bt_conn* app_connection;
static enum pulse_link_role app_role = PULSE_LINK_ROLE_UNRESOLVED;
static atomic_t link_result;
static atomic_t link_dropped;

#if defined(CONFIG_SENSWEAR_PULSE_ALGORITHM)
BUILD_ASSERT(PULSE_EVENT_WORKQUEUE_PRIORITY >= 0,
			 "PULSE event workqueue must use a preemptible priority");
BUILD_ASSERT(PULSE_EVENT_WORKQUEUE_PRIORITY < CONFIG_NUM_PREEMPT_PRIORITIES,
			 "PULSE event workqueue priority is outside the configured range");

enum pulse_event_work_kind {
	PULSE_EVENT_WORK_LOCAL = 0,
	PULSE_EVENT_WORK_NO_CONTACT,
	PULSE_EVENT_WORK_ACCEPTED,
	PULSE_EVENT_WORK_RESPONDER,
};

struct pulse_event_work_context {
	struct k_work work;
	struct k_work_sync sync;
	enum pulse_event_work_kind kind;
	enum pulse_event_path path;
	uint32_t iteration;
	uint64_t peer_id;
	int result;
	int infrastructure_result;
	int result_release_result;
	bool record_ready;
	bool reconnect_needed;
	struct pulse_metric_record record_snapshot;
};

static void pulse_event_work_handler(struct k_work* work);
static int disconnect_current_link(void);
static int ensure_connected_client(void);
static enum pulse_failure_reason failure_from_error(int error);

K_THREAD_STACK_DEFINE(pulse_event_workqueue_stack,
					  CONFIG_SENSWEAR_PULSE_EVENT_WORKQUEUE_STACK_SIZE);
K_SEM_DEFINE(responder_dispatch_sem, 0, 1);

static struct k_work_q pulse_event_workqueue;
static struct pulse_event_work_context pulse_event_job;
static atomic_t protocol_result;
static struct pulse_node_state node_state __attribute__((section(".pulse.state")));
static struct pulse_compute_workspace compute_workspace
	__attribute__((section(".pulse.activations")));
static pulse_gradient_t local_gradient __attribute__((section(".pulse.gradient.local")));
static pulse_gradient_t remote_gradient __attribute__((section(".pulse.gradient.remote")));
static pulse_head_t responder_head __attribute__((section(".pulse.protocol")));
static struct pulse_protocol_client protocol_client __attribute__((section(".pulse.protocol")));
static struct pulse_metric_record metric_record;
static struct pulse_protocol_transfer responder_transfer;
static atomic_t measurement_state;
static atomic_t measured_stages;
static atomic_t responder_record_pending;
static uint32_t campaign_measured_failure_count;
#if defined(CONFIG_SENSWEAR_PULSE_NGMO2_AUTONOMOUS_CAPTURE)
#define AUTONOMOUS_RESPONDER_TIMEOUT_WINDOWS 3
#define AUTONOMOUS_RESPONDER_COMPLETION_GUARD_MS 5000
/* run_responder_event() can consume one full timeout waiting for exchange
 * completion, one waiting for RESULT_RELEASE, and one forcing disconnect. */
BUILD_ASSERT(CONFIG_SENSWEAR_PULSE_NGMO2_PERSIST_DEADLINE_MS >
				 CONFIG_SENSWEAR_PULSE_NGMO2_EVENT_DEADLINE_MS +
				 AUTONOMOUS_RESPONDER_TIMEOUT_WINDOWS *
					 CONFIG_SENSWEAR_PULSE_EXCHANGE_TIMEOUT_MS +
					 AUTONOMOUS_RESPONDER_COMPLETION_GUARD_MS,
			 "autonomous proof persistence must follow every responder timeout window");
enum autonomous_responder_admission_state {
	AUTONOMOUS_RESPONDER_DISARMED = 0,
	AUTONOMOUS_RESPONDER_PASSIVE,
	AUTONOMOUS_RESPONDER_ARMED,
	AUTONOMOUS_RESPONDER_CLAIMED,
	AUTONOMOUS_RESPONDER_REJECTED,
	AUTONOMOUS_RESPONDER_SEALED,
};
static int64_t autonomous_boot_ms;
static uint32_t autonomous_sequence;
static struct pulse_capture_proof autonomous_proof;
static struct k_work_delayable autonomous_proof_work;
static uint8_t autonomous_proof_write_attempts;
static atomic_t autonomous_proof_sealed;
static atomic_t autonomous_responder_admission;
static atomic_t autonomous_responder_reject_errno;
static K_MUTEX_DEFINE(autonomous_proof_lock);
#endif
#endif

static void read_board_id(void) {
	uint8_t bytes[PULSE_BOARD_ID_BYTES];
	ssize_t length = hwinfo_get_device_id(bytes, sizeof(bytes));

	if (length <= 0) {
		strcpy(board_id, "unknown");
		return;
	}

	size_t used = MIN((size_t) length, (size_t) PULSE_BOARD_ID_BYTES);
	for (size_t i = 0U; i < used; ++i) {
		(void) snprintk(&board_id[i * 2U], 3U, "%02x", bytes[i]);
	}
	board_id[used * 2U] = '\0';
}

static int start_sensing_baseline(void) {
#if defined(CONFIG_SENSWEAR_PULSE_LIVE_IMU_BASELINE)
	struct device_manager_config_t config;
	int result = device_manager_init();

	if (result != 0) {
		return result;
	}
	result = device_manager_config_device(device_manager_device_Bhi360, NULL);
	if (result != 0) {
		return result;
	}
	device_manager_get_default_config(&config);
	config.imu.phy_streams_enabled = true;
	config.imu.drain_period_ms = 100U;
	config.gauge.update_period_min = 0U;
	config.temperature.update_period_min = 0U;
	result = device_manager_config(&config);
	if (result != 0) {
		return result;
	}
	return device_manager_start();
#else
	return 0;
#endif
}

static const char* link_role_name(enum pulse_link_role role) {
	switch (role) {
	case PULSE_LINK_ROLE_INITIATOR:
		return "initiator";
	case PULSE_LINK_ROLE_RESPONDER:
		return "responder";
	default:
		return "unresolved";
	}
}

static uint16_t snapshot_data_length(const struct pulse_link_snapshot* snapshot) {
	if (snapshot->tx_max_len == 0U) {
		return snapshot->rx_max_len;
	}
	if (snapshot->rx_max_len == 0U) {
		return snapshot->tx_max_len;
	}
	return MIN(snapshot->tx_max_len, snapshot->rx_max_len);
}

static void format_identity_address(const bt_addr_le_t* address,
									char output[PULSE_BT_ADDRESS_STRING_SIZE]) {
	if (bt_addr_le_eq(address, BT_ADDR_LE_ANY)) {
		strcpy(output, "unknown");
		return;
	}

	(void) snprintk(output,
					PULSE_BT_ADDRESS_STRING_SIZE,
					"%02x:%02x:%02x:%02x:%02x:%02x",
					address->a.val[5],
					address->a.val[4],
					address->a.val[3],
					address->a.val[2],
					address->a.val[1],
					address->a.val[0]);
}

static const char* identity_address_type(const bt_addr_le_t* address) {
	if (bt_addr_le_eq(address, BT_ADDR_LE_ANY)) {
		return "unknown";
	}
	if (address->type == BT_ADDR_LE_PUBLIC || address->type == BT_ADDR_LE_PUBLIC_ID) {
		return "public";
	}
	if (address->type == BT_ADDR_LE_RANDOM || address->type == BT_ADDR_LE_RANDOM_ID ||
		address->type == BT_ADDR_LE_UNRESOLVED) {
		if (BT_ADDR_IS_STATIC(&address->a)) {
			return "random_static";
		}
		if (BT_ADDR_IS_RPA(&address->a)) {
			return "random_resolvable";
		}
		if (BT_ADDR_IS_NRPA(&address->a)) {
			return "random_non_resolvable";
		}
		return "random";
	}
	return "unknown";
}

static void dump_run_metadata(const struct pulse_link_snapshot* snapshot) {
	char local_identity_address[PULSE_BT_ADDRESS_STRING_SIZE];
	char peer_identity_address[PULSE_BT_ADDRESS_STRING_SIZE];

	format_identity_address(&snapshot->local_address, local_identity_address);
	format_identity_address(&snapshot->peer_address, peer_identity_address);
	const struct pulse_run_metadata metadata = {
		.run_id = "boot",
		.pair_id = CONFIG_SENSWEAR_PULSE_PAIR_ID,
		.role = link_role_name(app_role),
		.build_guard = SENSWEAR_PULSE_BUILD_GUARD,
#if defined(CONFIG_SENSWEAR_PULSE_ALGORITHM)
		.build_variant = IS_ENABLED(CONFIG_SENSWEAR_PULSE_CORRECTNESS) ? "correctness"
																	   : "pulse_release",
		.model_id = "pulse-har-v1",
		.encoder_hash = pulse_fixture_encoder_sha256,
		.fixture_hash = pulse_fixture_sha256,
		.artifact_kind = pulse_fixture_artifact_kind,
		.artifact_source = pulse_fixture_artifact_source,
		.artifact_sha256 = pulse_fixture_artifact_sha256,
		.input_mode = "internal_flash_versioned_artifact_replay_windows",
		.tie_break_policy = "xorshift32_uniform_positive_exact_ties",
		.trial_state_policy =
			"reset_each_repetition_public_seed_selection_s1_observation_n1_u_plus_or_minus_0.2",
		.model_version = 1U,
		.parameter_count = 4550U,
		.head_bytes = 18200U,
		.compute_workspace_bytes = sizeof(compute_workspace),
		.gradient_buffer_bytes = sizeof(local_gradient),
		.received_head_bytes = sizeof(responder_head),
		.utility_table_bytes = sizeof(node_state.peers),
		.peer_entry_bytes = sizeof(pulse_peer_entry_t),
		.class_count = 6U,
		.batch_size = 16U,
		.local_steps = 2U,
		.bytes_per_parameter = sizeof(float),
		.learning_rate = 0.01f,
		.alpha_max = 0.5f,
		.utility_momentum = 0.2f,
		.initial_utility = 0.0f,
		.ucb_exploration = 0.25f,
		.norm_epsilon = 1.0e-12f,
		.peer_limit = CONFIG_SENSWEAR_PULSE_PEER_LIMIT,
		.warmup_repetitions = CONFIG_SENSWEAR_PULSE_WARMUP_REPETITIONS,
		.measured_repetitions = CONFIG_SENSWEAR_PULSE_REPETITIONS,
#else
		.build_variant = "matched_baseline",
		.model_id = "none",
		.encoder_hash = "none",
		.fixture_hash = "none",
		.artifact_kind = "none",
		.artifact_source = "none",
		.artifact_sha256 = "none",
		.input_mode = "sensing_only_no_model_input",
		.tie_break_policy = "na",
		.trial_state_policy = "na",
		.model_version = 0U,
		.parameter_count = 0U,
		.head_bytes = 0U,
		.compute_workspace_bytes = 0U,
		.gradient_buffer_bytes = 0U,
		.received_head_bytes = 0U,
		.utility_table_bytes = 0U,
		.peer_entry_bytes = 0U,
		.class_count = 0U,
		.batch_size = 0U,
		.local_steps = 0U,
		.bytes_per_parameter = 0U,
		.learning_rate = 0.0f,
		.alpha_max = 0.0f,
		.utility_momentum = 0.0f,
		.initial_utility = 0.0f,
		.ucb_exploration = 0.0f,
		.norm_epsilon = 0.0f,
		.peer_limit = 0U,
		.warmup_repetitions = 0U,
		.measured_repetitions = 0U,
#endif
		.firmware_revision = SENSWEAR_FIRMWARE_REVISION,
		.hardware_revision = CONFIG_SENSWEAR_PULSE_HARDWARE_REVISION,
		.enabled_sensors = IS_ENABLED(CONFIG_SENSWEAR_PULSE_LIVE_IMU_BASELINE)
							   ? "background_bhi360_accel_gyro_quaternion_100hz_not_model_input"
							   : "none",
		.security_state = "unencrypted_unbonded",
		.local_identity_address = local_identity_address,
		.local_identity_address_type = identity_address_type(&snapshot->local_address),
		.peer_identity_address = peer_identity_address,
		.peer_identity_address_type = identity_address_type(&snapshot->peer_address),
		.cpu_clock_hz = SystemCoreClock,
		.timeout_ms = CONFIG_SENSWEAR_PULSE_EXCHANGE_TIMEOUT_MS,
		.tx_power_dbm = CONFIG_BT_CTLR_TX_PWR_DBM,
		.initial_rssi_dbm = snapshot->rssi_dbm,
		.initial_att_mtu = snapshot->att_mtu,
		.initial_data_length = snapshot_data_length(snapshot),
		.initial_connection_interval_units = (uint16_t) (snapshot->interval_us / 1250U),
		.initial_tx_phy = snapshot->tx_phy,
		.initial_rx_phy = snapshot->rx_phy,
	};

	pulse_metrics_dump_metadata(&metadata, board_id);
}

#if defined(CONFIG_SENSWEAR_PULSE_ALGORITHM)
static uint32_t repetition_count(void) {
	return CONFIG_SENSWEAR_PULSE_WARMUP_REPETITIONS + CONFIG_SENSWEAR_PULSE_REPETITIONS;
}

#if defined(CONFIG_SENSWEAR_PULSE_NGMO2_AUTONOMOUS_CAPTURE)
static enum pulse_event_path autonomous_capture_path(void) {
#if defined(CONFIG_SENSWEAR_PULSE_NGMO2_PATH_LOCAL)
	return pulse_event_path_Local;
#elif defined(CONFIG_SENSWEAR_PULSE_NGMO2_PATH_NO_CONTACT)
	return pulse_event_path_NoContact;
#elif defined(CONFIG_SENSWEAR_PULSE_NGMO2_PATH_ACCEPTED_CONNECTED)
	return pulse_event_path_AcceptedConnected;
#elif defined(CONFIG_SENSWEAR_PULSE_NGMO2_PATH_ACCEPTED_DISCOVERY)
	return pulse_event_path_AcceptedDiscovery;
#else
#error "An autonomous NGMO2 event path must be selected"
#endif
}

static int16_t proof_errno(int error) {
	if (error > INT16_MAX) {
		return INT16_MAX;
	}
	if (error < INT16_MIN) {
		return INT16_MIN;
	}
	return (int16_t) error;
}

static uint32_t autonomous_trial_id(uint32_t sequence) {
	return sequence < CONFIG_SENSWEAR_PULSE_WARMUP_REPETITIONS
			   ? sequence
			   : sequence - CONFIG_SENSWEAR_PULSE_WARMUP_REPETITIONS;
}

static uint32_t autonomous_exchange_id(enum pulse_event_path path, uint32_t sequence) {
	if (path == pulse_event_path_AcceptedConnected) {
		return sequence + 1U;
	}
	if (path == pulse_event_path_AcceptedDiscovery) {
		return repetition_count() + sequence + 1U;
	}
	return 0U;
}

static bool autonomous_remote_path(enum pulse_event_path path) {
	return path == pulse_event_path_AcceptedConnected ||
		   path == pulse_event_path_AcceptedDiscovery;
}

static uint8_t proof_u8(uint32_t value) {
	return value > UINT8_MAX ? UINT8_MAX : (uint8_t) value;
}

static uint16_t proof_u16(uint32_t value) {
	return value > UINT16_MAX ? UINT16_MAX : (uint16_t) value;
}

static uint32_t proof_duration_us(uint64_t value) {
	return value > UINT32_MAX ? UINT32_MAX : (uint32_t) value;
}

static uint8_t autonomous_expected_proof_role(enum pulse_event_path path,
										   enum pulse_link_role role) {
	if (role == PULSE_LINK_ROLE_INITIATOR) {
		return path == pulse_event_path_Local || path == pulse_event_path_NoContact
				   ? (uint8_t) pulse_metric_role_Local
				   : (uint8_t) pulse_metric_role_Initiator;
	}
	if (role == PULSE_LINK_ROLE_RESPONDER) {
		return (uint8_t) pulse_metric_role_Responder;
	}
	return PULSE_CAPTURE_PROOF_ROLE_UNRESOLVED;
}

static int autonomous_responder_rejection_error(void) {
	int error = (int) atomic_get(&autonomous_responder_reject_errno);

	return error != 0 ? error : -ESTALE;
}

static void autonomous_responder_reject(enum autonomous_responder_admission_state expected,
										int error) {
	/* Server callbacks are serialized by the protocol lock. Publish the reason
	 * before REJECTED so the proof worker cannot observe a reasonless rejection. */
	if (atomic_get(&autonomous_responder_admission) != expected) {
		return;
	}
	atomic_set(&autonomous_responder_reject_errno, error);
	if (atomic_cas(&autonomous_responder_admission, expected, AUTONOMOUS_RESPONDER_REJECTED)) {
		k_sem_give(&responder_dispatch_sem);
	}
}

static void autonomous_proof_persist(struct k_work* work) {
	ARG_UNUSED(work);
	struct pulse_capture_proof snapshot;
	atomic_val_t responder_state =
		atomic_set(&autonomous_responder_admission, AUTONOMOUS_RESPONDER_SEALED);

	/* Close HEAD_BEGIN admission before freezing the proof snapshot. */
	atomic_set(&autonomous_proof_sealed, 1);

	k_mutex_lock(&autonomous_proof_lock, K_FOREVER);
	if (responder_state == AUTONOMOUS_RESPONDER_REJECTED) {
		int error = autonomous_responder_rejection_error();

		pulse_capture_proof_flag_set(
			&autonomous_proof, PULSE_CAPTURE_PROOF_FLAG_SUCCESS, false);
		autonomous_proof.event_errno = proof_errno(error);
		autonomous_proof.infrastructure_errno = proof_errno(error);
		autonomous_proof.failure_reason = (uint8_t) failure_from_error(error);
	} else if (!pulse_capture_proof_flag_get(
				   &autonomous_proof, PULSE_CAPTURE_PROOF_FLAG_RECORD_READY) &&
		   !pulse_capture_proof_flag_get(
			   &autonomous_proof, PULSE_CAPTURE_PROOF_FLAG_PASSIVE_EXPECTED) &&
		   autonomous_proof.infrastructure_errno == proof_errno(-EINPROGRESS)) {
		autonomous_proof.event_errno = proof_errno(-ETIMEDOUT);
		autonomous_proof.infrastructure_errno = proof_errno(-ETIMEDOUT);
		autonomous_proof.failure_reason = (uint8_t) pulse_failure_Timeout;
	}
	snapshot = autonomous_proof;
	k_mutex_unlock(&autonomous_proof_lock);

	int result = pulse_capture_log_append(&snapshot);
	if (result != 0 && ++autonomous_proof_write_attempts < 3U) {
		(void) k_work_reschedule(&autonomous_proof_work, K_SECONDS(1));
	}
}

static void autonomous_proof_prepare(enum pulse_event_path path,
								 uint32_t sequence,
								 enum pulse_link_role board_role) {
	memset(&autonomous_proof, 0, sizeof(autonomous_proof));
	atomic_set(&autonomous_proof_sealed, 0);
	atomic_set(&autonomous_responder_admission,
			   autonomous_remote_path(path) ? AUTONOMOUS_RESPONDER_ARMED
									: AUTONOMOUS_RESPONDER_PASSIVE);
	atomic_set(&autonomous_responder_reject_errno, 0);
	autonomous_proof_write_attempts = 0U;
	autonomous_proof.magic = PULSE_CAPTURE_PROOF_MAGIC;
	autonomous_proof.schema_version = PULSE_CAPTURE_PROOF_SCHEMA_VERSION;
	autonomous_proof.record_size = (uint8_t) sizeof(autonomous_proof);
	autonomous_proof.sequence = proof_u8(sequence);
	autonomous_proof.trial_id = proof_u8(autonomous_trial_id(sequence));
	autonomous_proof.exchange_id = proof_u16(autonomous_exchange_id(path, sequence));
	autonomous_proof.event_path = (uint8_t) path;
	autonomous_proof.role = autonomous_expected_proof_role(path, board_role);
	pulse_capture_proof_flag_set(
		&autonomous_proof,
		PULSE_CAPTURE_PROOF_FLAG_WARMUP,
		sequence < CONFIG_SENSWEAR_PULSE_WARMUP_REPETITIONS);
	autonomous_proof.failure_reason = (uint8_t) pulse_failure_Internal;
	autonomous_proof.event_errno = proof_errno(-EINPROGRESS);
	autonomous_proof.infrastructure_errno = proof_errno(-EINPROGRESS);
	autonomous_proof.firmware_revision_hash =
		pulse_metrics_hash32(SENSWEAR_FIRMWARE_REVISION, strlen(SENSWEAR_FIRMWARE_REVISION));
	autonomous_proof.artifact_hash = pulse_metrics_hash32(pulse_fixture_artifact_sha256,
											  strlen(pulse_fixture_artifact_sha256));

	k_work_init_delayable(&autonomous_proof_work, autonomous_proof_persist);
	int64_t delay_ms = autonomous_boot_ms +
					   CONFIG_SENSWEAR_PULSE_NGMO2_PERSIST_DEADLINE_MS - k_uptime_get();
	(void) k_work_schedule(&autonomous_proof_work, K_MSEC(MAX(delay_ms, INT64_C(0))));
}

static void autonomous_proof_fail(int error) {
	/* A terminal local/setup outcome must never leave the responder gate open. */
	atomic_set(&autonomous_responder_admission, AUTONOMOUS_RESPONDER_SEALED);
	k_mutex_lock(&autonomous_proof_lock, K_FOREVER);
	if (atomic_get(&autonomous_proof_sealed) != 0) {
		k_mutex_unlock(&autonomous_proof_lock);
		return;
	}
	pulse_capture_proof_flag_set(
		&autonomous_proof, PULSE_CAPTURE_PROOF_FLAG_SUCCESS, false);
	autonomous_proof.failure_reason = (uint8_t) failure_from_error(error);
	autonomous_proof.event_errno = proof_errno(error);
	autonomous_proof.infrastructure_errno = proof_errno(error);
	k_mutex_unlock(&autonomous_proof_lock);
}

static void autonomous_proof_mark_passive(void) {
	/* Resolve PASSIVE versus an unexpected remote claim atomically. */
	if (!atomic_cas(&autonomous_responder_admission,
					AUTONOMOUS_RESPONDER_PASSIVE,
					AUTONOMOUS_RESPONDER_SEALED)) {
		return;
	}
	k_mutex_lock(&autonomous_proof_lock, K_FOREVER);
	if (atomic_get(&autonomous_proof_sealed) != 0) {
		k_mutex_unlock(&autonomous_proof_lock);
		return;
	}
	autonomous_proof.role = (uint8_t) pulse_metric_role_Responder;
	pulse_capture_proof_flag_set(
		&autonomous_proof, PULSE_CAPTURE_PROOF_FLAG_PASSIVE_EXPECTED, true);
	pulse_capture_proof_flag_set(
		&autonomous_proof, PULSE_CAPTURE_PROOF_FLAG_SUCCESS, true);
	autonomous_proof.failure_reason = (uint8_t) pulse_failure_None;
	autonomous_proof.event_errno = 0;
	autonomous_proof.infrastructure_errno = 0;
	k_mutex_unlock(&autonomous_proof_lock);
}

static void autonomous_proof_update_from_job(void) {
	/* The first completed job is the only autonomous event admitted this boot. */
	atomic_set(&autonomous_responder_admission, AUTONOMOUS_RESPONDER_SEALED);
	k_mutex_lock(&autonomous_proof_lock, K_FOREVER);
	if (atomic_get(&autonomous_proof_sealed) != 0) {
		k_mutex_unlock(&autonomous_proof_lock);
		return;
	}
	autonomous_proof.event_errno = proof_errno(pulse_event_job.result);
	autonomous_proof.infrastructure_errno = proof_errno(pulse_event_job.infrastructure_result);
	autonomous_proof.result_release_errno = proof_errno(pulse_event_job.result_release_result);
	pulse_capture_proof_flag_set(&autonomous_proof,
							 PULSE_CAPTURE_PROOF_FLAG_RECORD_READY,
							 pulse_event_job.record_ready);
	if (pulse_event_job.record_ready) {
		const struct pulse_metric_record* record = &pulse_event_job.record_snapshot;
		bool expected_warmup = pulse_capture_proof_flag_get(
			&autonomous_proof, PULSE_CAPTURE_PROOF_FLAG_WARMUP);
		bool identity_matches =
			record->event_path == (enum pulse_event_path) autonomous_proof.event_path &&
			autonomous_proof.role != PULSE_CAPTURE_PROOF_ROLE_UNRESOLVED &&
			(uint8_t) record->role == autonomous_proof.role &&
			record->trial_id == autonomous_proof.trial_id &&
			record->exchange_id == autonomous_proof.exchange_id &&
			record->warmup == expected_warmup;

		if (!identity_matches) {
			pulse_capture_proof_flag_set(
				&autonomous_proof, PULSE_CAPTURE_PROOF_FLAG_SUCCESS, false);
			pulse_capture_proof_flag_set(
				&autonomous_proof, PULSE_CAPTURE_PROOF_FLAG_RECORD_READY, false);
			autonomous_proof.failure_reason = (uint8_t) failure_from_error(-ESTALE);
			autonomous_proof.event_errno = proof_errno(-ESTALE);
			autonomous_proof.infrastructure_errno = proof_errno(-ESTALE);
			k_mutex_unlock(&autonomous_proof_lock);
			return;
		}

		autonomous_proof.trial_id = proof_u8(record->trial_id);
		autonomous_proof.exchange_id = proof_u16(record->exchange_id);
		autonomous_proof.role = (uint8_t) record->role;
		pulse_capture_proof_flag_set(
			&autonomous_proof, PULSE_CAPTURE_PROOF_FLAG_SUCCESS, record->success);
		autonomous_proof.failure_reason = (uint8_t) record->failure_reason;
		pulse_capture_proof_flag_set(&autonomous_proof,
							 PULSE_CAPTURE_PROOF_FLAG_REMOTE_SESSION_STARTED,
							 record->remote_session_started);
		pulse_capture_proof_flag_set(
			&autonomous_proof, PULSE_CAPTURE_PROOF_FLAG_WARMUP, record->warmup);
		autonomous_proof.initial_head_hash = record->initial_head_hash;
		autonomous_proof.result_head_hash = record->result_head_hash;
		if (record->event_wall_end >= record->event_wall_start) {
			autonomous_proof.event_duration_us = proof_duration_us(
				k_cyc_to_us_floor64(record->event_wall_end - record->event_wall_start));
		}
	}
	k_mutex_unlock(&autonomous_proof_lock);
}

static int wait_for_autonomous_event_deadline(void) {
	int64_t remaining_ms = autonomous_boot_ms +
					   CONFIG_SENSWEAR_PULSE_NGMO2_EVENT_DEADLINE_MS - k_uptime_get();

	if (remaining_ms < 0) {
		return -ETIME;
	}
	k_msleep((int32_t) remaining_ms);
	return 0;
}

static void autonomous_hold_forever(void) {
	for (;;) {
		k_sleep(K_FOREVER);
	}
}
#endif

#if defined(CONFIG_SENSWEAR_PULSE_UART_PACED_CAPTURE)
static const struct device* const pulse_capture_console =
	DEVICE_DT_GET(DT_CHOSEN(zephyr_console));

static int wait_capture_command(const char* expected) {
	char command[12];
	size_t used = 0U;

	if (!device_is_ready(pulse_capture_console)) {
		return -ENODEV;
	}
	for (;;) {
		uint8_t byte;
		int result = uart_poll_in(pulse_capture_console, &byte);

		if (result != 0) {
			k_msleep(1);
			continue;
		}
		if (byte == '\r' || byte == '\n') {
			if (used == 0U) {
				continue;
			}
			command[used] = '\0';
			if (strcmp(command, expected) == 0) {
				return 0;
			}
			used = 0U;
			continue;
		}
		if (byte == ' ' || byte == '\t') {
			continue;
		}
		if (used + 1U < sizeof(command)) {
			command[used++] = (char) byte;
		} else {
			used = 0U;
		}
	}
}

static int paced_capture_begin(enum pulse_event_path path,
								uint32_t iteration,
								uint32_t event_index,
								uint32_t trial_id,
								uint32_t exchange_id,
								bool warmup,
								uint64_t* envelope_start) {
	printk("PULSE_READY,run_id=boot,pair_id=%s,board_id=%s,board_role=initiator,"
		   "role=%s,event_path=%s,iteration=%u,event_index=%u,trial_id=%u,"
		   "exchange_id=%u,warmup=%u,capture_quiet_guard_ms=%u\n",
		   CONFIG_SENSWEAR_PULSE_PAIR_ID,
		   board_id,
		   path == pulse_event_path_Local || path == pulse_event_path_NoContact
			   ? "local"
			   : "initiator",
		   pulse_metrics_path_name(path),
		   iteration,
		   event_index,
		   trial_id,
		   exchange_id,
		   warmup ? 1U : 0U,
		   CONFIG_SENSWEAR_PULSE_CAPTURE_QUIET_GUARD_MS);
	int result = wait_capture_command("GO");

	if (result != 0) {
		return result;
	}
	k_msleep(CONFIG_SENSWEAR_PULSE_CAPTURE_QUIET_GUARD_MS);
	*envelope_start = k_cycle_get_64();
	return 0;
}

static int paced_capture_finish(uint64_t envelope_start,
								struct pulse_metric_record* record) {
	uint64_t envelope_end = k_cycle_get_64();

	if (record != NULL && envelope_end >= envelope_start) {
		record->capture_envelope_duration_us =
			k_cyc_to_us_floor64(envelope_end - envelope_start);
	}
	k_msleep(CONFIG_SENSWEAR_PULSE_CAPTURE_QUIET_GUARD_MS);
	return wait_capture_command("DUMP");
}

static int paced_responder_release(void) {
	k_msleep(CONFIG_SENSWEAR_PULSE_CAPTURE_QUIET_GUARD_MS);
	return wait_capture_command("DUMP");
}
#else
static int paced_capture_begin(enum pulse_event_path path,
								uint32_t iteration,
								uint32_t event_index,
								uint32_t trial_id,
								uint32_t exchange_id,
								bool warmup,
								uint64_t* envelope_start) {
	ARG_UNUSED(path);
	ARG_UNUSED(iteration);
	ARG_UNUSED(event_index);
	ARG_UNUSED(trial_id);
	ARG_UNUSED(exchange_id);
	ARG_UNUSED(warmup);
	*envelope_start = 0U;
	return 0;
}

static int paced_capture_finish(uint64_t envelope_start,
								struct pulse_metric_record* record) {
	ARG_UNUSED(envelope_start);
	ARG_UNUSED(record);
	return 0;
}

static int paced_responder_release(void) {
	return 0;
}
#endif

static void measured_stage_begin(enum measured_stage_bit bit,
								 enum pulse_stage stage,
								 enum pulse_marker_stage marker) {
	if (atomic_get(&measurement_state) != 1) {
		return;
	}
	atomic_set_bit(&measured_stages, bit);
	pulse_metrics_stage_begin(&metric_record, stage, marker);
}

static void measured_stage_end(enum measured_stage_bit bit, enum pulse_stage stage) {
	if (atomic_get(&measurement_state) != 0 && atomic_test_and_clear_bit(&measured_stages, bit)) {
		pulse_metrics_stage_end(&metric_record, stage);
	}
}

static void close_measured_stages(void) {
	measured_stage_end(MEASURED_STAGE_SCAN, pulse_stage_BleScan);
	measured_stage_end(MEASURED_STAGE_CONNECT, pulse_stage_BleConnect);
	measured_stage_end(MEASURED_STAGE_HEAD_TRANSFER, pulse_stage_HeadTransfer);
	measured_stage_end(MEASURED_STAGE_GRADIENT_TRANSFER, pulse_stage_GradientTransfer);
}

static bool complete_measured_event(bool success, enum pulse_failure_reason failure) {
	if (!atomic_cas(&measurement_state, 1, 2)) {
		return false;
	}
	close_measured_stages();
	pulse_metrics_event_end(&metric_record, success, failure);
	atomic_set(&measurement_state, 0);
	return true;
}
#endif

static void link_scan_found(const bt_addr_le_t* peer,
							int8_t advertisement_rssi,
							enum pulse_link_role role,
							void* user_data) {
	ARG_UNUSED(peer);
	ARG_UNUSED(user_data);
	k_event_post(&app_events, APP_EVENT_SCAN_FOUND);

#if defined(CONFIG_SENSWEAR_PULSE_ALGORITHM)
	if (role == PULSE_LINK_ROLE_INITIATOR && atomic_get(&measurement_state) == 1 &&
		(metric_record.event_path == pulse_event_path_AcceptedDiscovery ||
		 metric_record.event_path == pulse_event_path_NoContact)) {
		metric_record.rssi_dbm = advertisement_rssi;
		measured_stage_end(MEASURED_STAGE_SCAN, pulse_stage_BleScan);
		if (metric_record.event_path == pulse_event_path_AcceptedDiscovery) {
			measured_stage_begin(MEASURED_STAGE_CONNECT,
								 pulse_stage_BleConnect,
								 pulse_marker_stage_Control);
		}
	}
#else
	ARG_UNUSED(advertisement_rssi);
	ARG_UNUSED(role);
#endif
}

static void link_connected(enum pulse_link_role role, int error, void* user_data) {
	ARG_UNUSED(user_data);
	app_role = role;
	atomic_set(&link_result, error);
	k_event_post(&app_events, APP_EVENT_LINK_CONNECTED);
}

static void link_disconnected(enum pulse_link_role role, uint8_t reason, void* user_data) {
	ARG_UNUSED(reason);
	ARG_UNUSED(user_data);
	atomic_set(&link_dropped, 1);

#if defined(CONFIG_SENSWEAR_PULSE_ALGORITHM)
	if (app_connection != NULL) {
		if (role == PULSE_LINK_ROLE_INITIATOR) {
			pulse_protocol_client_disconnected(&protocol_client, app_connection);
		} else if (role == PULSE_LINK_ROLE_RESPONDER) {
			pulse_protocol_server_disconnected(app_connection);
		}
	}
#else
	ARG_UNUSED(role);
#endif
	k_event_post(&app_events, APP_EVENT_LINK_DISCONNECTED);
#if defined(CONFIG_SENSWEAR_PULSE_ALGORITHM)
	if (role == PULSE_LINK_ROLE_RESPONDER) {
		k_sem_give(&responder_dispatch_sem);
	}
#endif
}

#if defined(CONFIG_SENSWEAR_PULSE_ALGORITHM)
static void reconnect_started(enum pulse_link_role role, void* user_data) {
	ARG_UNUSED(user_data);
	if (role == PULSE_LINK_ROLE_INITIATOR && atomic_get(&measurement_state) == 1) {
		measured_stage_begin(MEASURED_STAGE_SCAN, pulse_stage_BleScan, pulse_marker_stage_Control);
	}
}

static void reconnect_finished(enum pulse_link_role role, int error, void* user_data) {
	ARG_UNUSED(user_data);
	if (role != PULSE_LINK_ROLE_INITIATOR || atomic_get(&measurement_state) != 1) {
		return;
	}
	if (error != 0) {
		measured_stage_end(MEASURED_STAGE_SCAN, pulse_stage_BleScan);
		measured_stage_end(MEASURED_STAGE_CONNECT, pulse_stage_BleConnect);
	}
}

static void protocol_client_event(struct pulse_protocol_client* client,
								  enum pulse_protocol_client_event event,
								  int error,
								  enum pulse_protocol_remote_status remote_status,
								  const struct pulse_protocol_transfer* transfer,
								  void* user_data) {
	ARG_UNUSED(client);
	ARG_UNUSED(remote_status);
	ARG_UNUSED(transfer);
	ARG_UNUSED(user_data);

	switch (event) {
	case PULSE_PROTOCOL_CLIENT_EVENT_DISCOVERY_COMPLETE:
		k_event_post(&app_events, APP_EVENT_CLIENT_DISCOVERED);
		break;
	case PULSE_PROTOCOL_CLIENT_EVENT_HEAD_SENT:
		measured_stage_end(MEASURED_STAGE_HEAD_TRANSFER, pulse_stage_HeadTransfer);
		k_event_post(&app_events, APP_EVENT_CLIENT_HEAD_SENT);
		break;
	case PULSE_PROTOCOL_CLIENT_EVENT_REMOTE_PROCESSING:
		k_event_post(&app_events, APP_EVENT_CLIENT_REMOTE_PROCESSING);
		break;
	case PULSE_PROTOCOL_CLIENT_EVENT_GRADIENT_READY:
		k_event_post(&app_events, APP_EVENT_CLIENT_GRADIENT_READY);
		break;
	case PULSE_PROTOCOL_CLIENT_EVENT_GRADIENT_RECEIVED:
		measured_stage_end(MEASURED_STAGE_GRADIENT_TRANSFER, pulse_stage_GradientTransfer);
		k_event_post(&app_events, APP_EVENT_CLIENT_GRADIENT_RECEIVED);
		break;
	case PULSE_PROTOCOL_CLIENT_EVENT_COMPLETE:
		k_event_post(&app_events, APP_EVENT_CLIENT_COMPLETE);
		break;
	case PULSE_PROTOCOL_CLIENT_EVENT_RESULT_RELEASED:
		k_event_post(&app_events, APP_EVENT_CLIENT_RESULT_RELEASED);
		break;
	case PULSE_PROTOCOL_CLIENT_EVENT_CANCELLED:
		k_event_post(&app_events, APP_EVENT_CLIENT_CANCELLED);
		break;
	case PULSE_PROTOCOL_CLIENT_EVENT_DISCONNECTED:
		atomic_set(&protocol_result, error != 0 ? error : -ENOTCONN);
		k_event_post(&app_events, APP_EVENT_LINK_DISCONNECTED);
		break;
	case PULSE_PROTOCOL_CLIENT_EVENT_ERROR:
	default:
		atomic_set(&protocol_result, error != 0 ? error : -EPROTO);
		k_event_post(&app_events, APP_EVENT_CLIENT_ERROR);
		break;
	}
}

static bool decode_session(uint32_t session_id,
						   enum pulse_event_path* path,
						   uint32_t* event_index,
						   uint32_t* trial_id,
						   bool* warmup) {
	uint32_t total = repetition_count();
	uint32_t iteration;

	if (session_id == 0U || session_id > 2U * total) {
		return false;
	}
	if (session_id <= total) {
		iteration = session_id - 1U;
		*path = pulse_event_path_AcceptedConnected;
		*event_index = 2U * total + iteration;
	} else {
		iteration = session_id - total - 1U;
		*path = pulse_event_path_AcceptedDiscovery;
		*event_index = 3U * total + iteration;
	}
	*warmup = iteration < CONFIG_SENSWEAR_PULSE_WARMUP_REPETITIONS;
	*trial_id = *warmup ? iteration : iteration - CONFIG_SENSWEAR_PULSE_WARMUP_REPETITIONS;
	return true;
}

#if defined(CONFIG_SENSWEAR_PULSE_NGMO2_AUTONOMOUS_CAPTURE)
static bool autonomous_responder_admit_session(const struct pulse_protocol_transfer* transfer,
												void* user_data) {
	enum pulse_event_path expected_path = autonomous_capture_path();
	enum pulse_event_path decoded_path;
	uint32_t event_index;
	uint32_t trial_id;
	bool warmup;

	ARG_UNUSED(user_data);
	if (!autonomous_remote_path(expected_path)) {
		autonomous_responder_reject(AUTONOMOUS_RESPONDER_PASSIVE, -ESTALE);
		return false;
	}

	int64_t claim_cutoff_ms =
		autonomous_boot_ms + CONFIG_SENSWEAR_PULSE_NGMO2_PERSIST_DEADLINE_MS -
		AUTONOMOUS_RESPONDER_TIMEOUT_WINDOWS * CONFIG_SENSWEAR_PULSE_EXCHANGE_TIMEOUT_MS -
		AUTONOMOUS_RESPONDER_COMPLETION_GUARD_MS;
	bool claim_window_open = k_uptime_get() < claim_cutoff_ms;
	bool identity_matches =
		claim_window_open && transfer != NULL &&
		decode_session(transfer->session_id, &decoded_path, &event_index, &trial_id, &warmup) &&
		decoded_path == expected_path &&
		transfer->session_id == autonomous_exchange_id(expected_path, autonomous_sequence) &&
		trial_id == autonomous_trial_id(autonomous_sequence) &&
		warmup == (autonomous_sequence < CONFIG_SENSWEAR_PULSE_WARMUP_REPETITIONS);

	ARG_UNUSED(event_index);
	if (!identity_matches) {
		autonomous_responder_reject(AUTONOMOUS_RESPONDER_ARMED,
									claim_window_open ? -ESTALE : -ETIMEDOUT);
		return false;
	}

	return atomic_cas(&autonomous_responder_admission,
					  AUTONOMOUS_RESPONDER_ARMED,
					  AUTONOMOUS_RESPONDER_CLAIMED);
}
#endif

static void server_session_started(const struct pulse_protocol_transfer* transfer,
								   void* user_data) {
	enum pulse_event_path path;
	uint32_t event_index;
	uint32_t trial_id;
	bool warmup;

	ARG_UNUSED(user_data);
	if (transfer == NULL || atomic_get(&measurement_state) != 0) {
		atomic_set(&protocol_result, -EBUSY);
		k_event_post(&app_events, APP_EVENT_SERVER_CANCELLED);
		return;
	}

	bool valid = decode_session(transfer->session_id, &path, &event_index, &trial_id, &warmup);
	if (!valid) {
		path = pulse_event_path_Failed;
		event_index = transfer->session_id;
		trial_id = 0U;
		warmup = false;
	}

	pulse_metrics_system_heap_peak_reset();
	atomic_set(&measured_stages, 0);
	atomic_set(&link_dropped, 0);
	pulse_metrics_event_begin(&metric_record,
							  path,
							  pulse_metric_role_Responder,
							  pulse_connection_start_Connected,
							  trial_id,
							  transfer->session_id);
	atomic_set(&measurement_state, 1);
	metric_record.event_index = event_index;
	metric_record.warmup = warmup;
	metric_record.input_batch_id = 3U;
	metric_record.reachable_peers = 1U;
	metric_record.head_version_before = transfer->head_version;
	metric_record.head_version_after = transfer->head_version;
	metric_record.head_payload_bytes = PULSE_PROTOCOL_TENSOR_BYTES;
	metric_record.gradient_payload_bytes = PULSE_PROTOCOL_TENSOR_BYTES;
	metric_record.remote_session_started = true;
	responder_transfer = *transfer;
	atomic_set(&responder_record_pending, 1);
	measured_stage_begin(MEASURED_STAGE_HEAD_TRANSFER,
						 pulse_stage_HeadTransfer,
						 pulse_marker_stage_Radio);
	k_event_post(&app_events, APP_EVENT_SERVER_STARTED);
	k_sem_give(&responder_dispatch_sem);
}

static void server_head_ready(const struct pulse_protocol_transfer* transfer,
							  const uint8_t* serialized_head,
							  void* user_data) {
	ARG_UNUSED(serialized_head);
	ARG_UNUSED(user_data);
	if (transfer != NULL) {
		responder_transfer = *transfer;
	}
	measured_stage_end(MEASURED_STAGE_HEAD_TRANSFER, pulse_stage_HeadTransfer);
	k_event_post(&app_events, APP_EVENT_SERVER_HEAD_READY);
}

static void server_gradient_received(const struct pulse_protocol_transfer* transfer,
									 void* user_data) {
	ARG_UNUSED(transfer);
	ARG_UNUSED(user_data);
	measured_stage_end(MEASURED_STAGE_GRADIENT_TRANSFER, pulse_stage_GradientTransfer);
	k_event_post(&app_events, APP_EVENT_SERVER_GRADIENT_RECEIVED);
}

static void server_session_complete(const struct pulse_protocol_transfer* transfer,
									void* user_data) {
	ARG_UNUSED(transfer);
	ARG_UNUSED(user_data);
	/*
	 * Close the successful responder interval in the GATT ACK callback before
	 * the write response can let the initiator send gate-low RESULT_RELEASE.
	 */
	(void) complete_measured_event(true, pulse_failure_None);
	k_event_post(&app_events, APP_EVENT_SERVER_COMPLETE);
}

static void server_result_released(const struct pulse_protocol_transfer* transfer,
								   void* user_data) {
	ARG_UNUSED(transfer);
	ARG_UNUSED(user_data);
	k_event_post(&app_events, APP_EVENT_SERVER_RESULT_RELEASED);
	k_sem_give(&responder_dispatch_sem);
}

static void server_session_cancelled(uint32_t session_id, void* user_data) {
	ARG_UNUSED(session_id);
	ARG_UNUSED(user_data);
	k_event_post(&app_events, APP_EVENT_SERVER_CANCELLED);
	k_sem_give(&responder_dispatch_sem);
}
#endif

static int initialize_link(struct pulse_link_snapshot* snapshot) {
	const struct pulse_link_config config = {
		.callbacks =
			{
				.scan_found = link_scan_found,
				.connected = link_connected,
				.disconnected = link_disconnected,
			},
		.user_data = NULL,
	};
	int result;

	memset(snapshot, 0, sizeof(*snapshot));
	snapshot->rssi_dbm = PULSE_LINK_RSSI_UNAVAILABLE;
	atomic_set(&link_result, -EINPROGRESS);
	result = pulse_link_start(&link_manager, &config);
	if (result != 0) {
		return result;
	}
	result =
		pulse_link_wait_connected(&link_manager, K_MSEC(CONFIG_SENSWEAR_PULSE_EXCHANGE_TIMEOUT_MS));
	app_role = pulse_link_role_get(&link_manager);
	if (result != 0) {
		return result;
	}
	app_connection = pulse_link_connected_ref(&link_manager);
	if (app_connection == NULL) {
		return -ENOTCONN;
	}
	return pulse_link_snapshot_get(&link_manager, snapshot);
}

#if defined(CONFIG_SENSWEAR_PULSE_ALGORITHM)
static enum pulse_failure_reason failure_from_error(int error) {
	switch (error) {
	case -ETIMEDOUT:
	case -ETIME:
	case -EAGAIN:
		return pulse_failure_Timeout;
	case -ENOTCONN:
	case -ECONNRESET:
	case -ECONNABORTED:
	case -ECONNREFUSED:
		return pulse_failure_Disconnected;
	case -EBADMSG:
		return pulse_failure_Crc;
	case -EPROTONOSUPPORT:
		return pulse_failure_ModelMismatch;
	case -ENOMEM:
	case -ENOSPC:
	case -EMSGSIZE:
		return pulse_failure_Resource;
	case -ECANCELED:
	case -ESTALE:
		return pulse_failure_Cancelled;
	case -EPROTO:
	case -EIO:
	case -EBUSY:
		return pulse_failure_Protocol;
	default:
		return pulse_failure_Internal;
	}
}

static uint64_t peer_id_from_address(const bt_addr_le_t* address) {
	uint64_t id = address->type;

	for (int index = (int) sizeof(address->a.val) - 1; index >= 0; --index) {
		id = (id << 8) | address->a.val[index];
	}
	return id != 0U ? id : PULSE_FALLBACK_PEER_ID;
}

static uint64_t connected_peer_id(void) {
	struct pulse_link_snapshot snapshot;

	if (pulse_link_snapshot_get(&link_manager, &snapshot) != 0) {
		return PULSE_FALLBACK_PEER_ID;
	}
	return peer_id_from_address(&snapshot.peer_address);
}

static int64_t exchange_deadline(void) {
	return k_uptime_get() + CONFIG_SENSWEAR_PULSE_EXCHANGE_TIMEOUT_MS;
}

static k_timeout_t remaining_timeout(int64_t deadline) {
	int64_t remaining = deadline - k_uptime_get();

	return remaining > 0 ? K_MSEC(remaining) : K_NO_WAIT;
}

static int wait_client_event(uint32_t expected, int64_t deadline, uint32_t* received) {
	uint32_t events = k_event_wait_safe(&app_events,
										expected | APP_EVENT_CLIENT_TERMINAL,
										false,
										remaining_timeout(deadline));

	if (received != NULL) {
		*received = events;
	}
	if (events == 0U) {
		return -ETIMEDOUT;
	}
	if ((events & APP_EVENT_LINK_DISCONNECTED) != 0U) {
		return -ENOTCONN;
	}
	if ((events & APP_EVENT_CLIENT_ERROR) != 0U) {
		int error = (int) atomic_get(&protocol_result);

		return error != 0 ? error : -EPROTO;
	}
	if ((events & APP_EVENT_CLIENT_CANCELLED) != 0U) {
		return -ECANCELED;
	}
	return (events & expected) != 0U ? 0 : -EPROTO;
}

static int discover_protocol_client(int64_t deadline) {
	int result;

	k_event_clear(&app_events,
				  APP_EVENT_CLIENT_DISCOVERED | APP_EVENT_CLIENT_ERROR |
					  APP_EVENT_CLIENT_CANCELLED | APP_EVENT_LINK_DISCONNECTED);
	atomic_set(&protocol_result, 0);
	result = pulse_protocol_client_discover(&protocol_client, app_connection);
	if (result != 0) {
		return result;
	}
	return wait_client_event(APP_EVENT_CLIENT_DISCOVERED, deadline, NULL);
}

static void populate_link_metrics(struct pulse_metric_record* record) {
	struct pulse_link_snapshot snapshot;

	if (pulse_link_snapshot_get(&link_manager, &snapshot) != 0) {
		return;
	}
	/* Preserve an advertisement RSSI when discovery succeeded but no link exists. */
	if (snapshot.rssi_dbm != PULSE_LINK_RSSI_UNAVAILABLE) {
		record->rssi_dbm = snapshot.rssi_dbm;
	}
	record->att_mtu = snapshot.att_mtu;
	record->data_length = snapshot_data_length(&snapshot);
	record->connection_interval_units = (uint16_t) (snapshot.interval_us / 1250U);
	record->tx_phy = snapshot.tx_phy;
	record->rx_phy = snapshot.rx_phy;
}

static void populate_protocol_metrics(struct pulse_metric_record* record, bool server) {
	struct pulse_protocol_counters counters = {0};

	if (server) {
		pulse_protocol_server_counters_get(&counters);
	} else {
		pulse_protocol_client_counters_get(&protocol_client, &counters);
	}
	record->application_tx_bytes = counters.application_tx_bytes;
	record->application_rx_bytes = counters.application_rx_bytes;
	record->att_tx_bytes = counters.att_value_tx_bytes;
	record->att_rx_bytes = counters.att_value_rx_bytes;
	record->protocol_tx_bytes = counters.att_value_tx_bytes >= counters.application_tx_bytes
									? counters.att_value_tx_bytes - counters.application_tx_bytes
									: 0U;
	record->protocol_rx_bytes = counters.att_value_rx_bytes >= counters.application_rx_bytes
									? counters.att_value_rx_bytes - counters.application_rx_bytes
									: 0U;
	record->tx_chunks = (uint16_t) MIN(counters.tx_chunks, (uint32_t) UINT16_MAX);
	record->rx_chunks = (uint16_t) MIN(counters.rx_chunks, (uint32_t) UINT16_MAX);
}

static void initialize_event_workqueue(void) {
	const struct k_work_queue_config config = {
		.name = "pulse_event",
	};

	k_work_queue_init(&pulse_event_workqueue);
	k_work_init(&pulse_event_job.work, pulse_event_work_handler);
	k_work_queue_start(&pulse_event_workqueue,
					   pulse_event_workqueue_stack,
					   K_THREAD_STACK_SIZEOF(pulse_event_workqueue_stack),
					   PULSE_EVENT_WORKQUEUE_PRIORITY,
					   &config);
}

static uint32_t event_workqueue_stack_peak(void) {
	return pulse_metrics_thread_stack_peak(&pulse_event_workqueue.thread,
										   K_THREAD_STACK_SIZEOF(pulse_event_workqueue_stack));
}

static void capture_record(bool server, bool include_gradient_diagnostics) {
	metric_record.stack_peak_bytes = event_workqueue_stack_peak();
	metric_record.system_heap_peak_bytes = pulse_metrics_system_heap_peak_bytes();
	metric_record.remote_session_started =
		server || pulse_protocol_client_remote_session_started_get(&protocol_client);
	populate_link_metrics(&metric_record);
	populate_protocol_metrics(&metric_record, server);
	if (include_gradient_diagnostics) {
		pulse_benchmark_populate_diagnostics(&node_state,
											 &local_gradient,
											 &remote_gradient,
											 &metric_record);
	}
}

static int emit_record(const struct pulse_metric_record* record) {
	int result = pulse_metrics_store(record);
	uint32_t quiet_ms = CONFIG_SENSWEAR_PULSE_EVENT_GAP_MS;

	if (result != 0) {
		printk("PULSE_STATUS,board_id=%s,error=metric_store,errno=%d\n", board_id, result);
	}
	pulse_metrics_dump_and_reset(board_id);
	if (CONFIG_SENSWEAR_PULSE_MIN_EVENT_PERIOD_MS > 0 &&
		record->event_wall_end >= record->event_wall_start) {
		uint64_t event_duration_ms =
			k_cyc_to_ms_floor64(record->event_wall_end - record->event_wall_start);

		if (event_duration_ms < CONFIG_SENSWEAR_PULSE_MIN_EVENT_PERIOD_MS) {
			uint32_t period_remainder_ms =
				CONFIG_SENSWEAR_PULSE_MIN_EVENT_PERIOD_MS - (uint32_t) event_duration_ms;

			quiet_ms = MAX(quiet_ms, period_remainder_ms);
		}
	}
	k_msleep(quiet_ms);
	return result;
}

static int reset_trial(uint32_t trial_id, uint32_t salt) {
	return pulse_benchmark_reset_trial(&node_state, UINT32_C(0x05eeda11) ^ trial_id ^ salt);
}

static bool connected_client_precondition(void) {
	enum pulse_protocol_client_state state = pulse_protocol_client_state_get(&protocol_client);

	return app_connection != NULL && atomic_get(&link_dropped) == 0 &&
		   (state == PULSE_PROTOCOL_CLIENT_READY || state == PULSE_PROTOCOL_CLIENT_COMPLETE);
}

static int dispatch_event_work(enum pulse_event_work_kind kind,
							   enum pulse_event_path path,
							   uint32_t iteration,
							   uint64_t peer_id);

static void run_local_trial(uint32_t iteration) {
	bool warmup = iteration < CONFIG_SENSWEAR_PULSE_WARMUP_REPETITIONS;
	uint32_t trial_id = warmup ? iteration : iteration - CONFIG_SENSWEAR_PULSE_WARMUP_REPETITIONS;
	int result;

	pulse_metrics_system_heap_peak_reset();
	if (!connected_client_precondition()) {
		pulse_event_job.infrastructure_result = -ENOTCONN;
		return;
	}
	result = reset_trial(trial_id, 0U);
	if (result != 0) {
		pulse_event_job.infrastructure_result = result;
		return;
	}

	result = pulse_benchmark_run_scheduled_local(&node_state,
												 &compute_workspace,
												 &local_gradient,
												 &metric_record,
												 trial_id);
	if (atomic_get(&link_dropped) != 0) {
		metric_record.success = false;
		metric_record.failure_reason = pulse_failure_Disconnected;
		result = -ENOTCONN;
	}
	metric_record.event_index = iteration;
	metric_record.warmup = warmup;
	metric_record.input_batch_id = 0U;
	metric_record.connection_start = pulse_connection_start_Connected;
	metric_record.stack_peak_bytes = event_workqueue_stack_peak();
	metric_record.system_heap_peak_bytes = pulse_metrics_system_heap_peak_bytes();
	populate_link_metrics(&metric_record);
	pulse_event_job.result = result;
	pulse_event_job.record_ready = true;
}

static int __maybe_unused run_local_trials(void) {
	uint32_t total = repetition_count();
	uint32_t emitted = 0U;
	uint32_t measured_failures = 0U;

	for (uint32_t iteration = 0U; iteration < total; ++iteration) {
		uint32_t trial_id = iteration < CONFIG_SENSWEAR_PULSE_WARMUP_REPETITIONS
							? iteration
							: iteration - CONFIG_SENSWEAR_PULSE_WARMUP_REPETITIONS;
		uint64_t capture_start = 0U;
		atomic_set(&link_dropped, 0);
		int result = ensure_connected_client();

		if (result != 0) {
			printk("PULSE_STATUS,board_id=%s,error=infrastructure,path=local,stage=precondition,"
				   "iteration=%u,errno=%d\n",
				   board_id,
				   iteration,
				   result);
			return result;
		}
		result = paced_capture_begin(pulse_event_path_Local,
								 iteration,
								 iteration,
								 trial_id,
								 0U,
								 iteration < CONFIG_SENSWEAR_PULSE_WARMUP_REPETITIONS,
								 &capture_start);
		if (result != 0) {
			printk("PULSE_STATUS,board_id=%s,error=infrastructure,path=local,"
				   "stage=capture_go,iteration=%u,errno=%d\n",
				   board_id,
				   iteration,
				   result);
			return result;
		}
		result = dispatch_event_work(PULSE_EVENT_WORK_LOCAL, pulse_event_path_Local, iteration, 0U);
		int capture_result = paced_capture_finish(
			capture_start,
			pulse_event_job.record_ready ? &pulse_event_job.record_snapshot : NULL);
		if (capture_result != 0) {
			printk("PULSE_STATUS,board_id=%s,error=infrastructure,path=local,"
				   "stage=capture_dump,iteration=%u,errno=%d\n",
				   board_id,
				   iteration,
				   capture_result);
			return capture_result;
		}

		if (result != 0) {
			printk("PULSE_STATUS,board_id=%s,error=infrastructure,path=local,"
				   "stage=event_work_submit,iteration=%u,errno=%d\n",
				   board_id,
				   iteration,
				   result);
			return result;
		}
		if (pulse_event_job.infrastructure_result != 0 || !pulse_event_job.record_ready) {
			result = pulse_event_job.infrastructure_result != 0
						 ? pulse_event_job.infrastructure_result
						 : -EIO;
			printk("PULSE_STATUS,board_id=%s,error=infrastructure,path=local,stage=prepare,"
				   "iteration=%u,errno=%d\n",
				   board_id,
				   iteration,
				   result);
			return result;
		}
		result = emit_record(&pulse_event_job.record_snapshot);
		if (result != 0) {
			return result;
		}
		++emitted;
		if (pulse_event_job.result != 0) {
			printk("PULSE_STATUS,board_id=%s,error=measured_event,path=local,iteration=%u,"
				   "errno=%d\n",
				   board_id,
				   iteration,
				   pulse_event_job.result);
			++measured_failures;
		}
	}
	campaign_measured_failure_count += measured_failures;
	printk("PULSE_STATUS,board_id=%s,state=block_complete,path=local,rows=%u,"
		   "measured_failures=%u\n",
		   board_id,
		   emitted,
		   measured_failures);
	return emitted == total ? 0 : -EIO;
}

static void run_no_contact_trial(uint32_t iteration, uint64_t peer_id) {
	uint32_t total = repetition_count();
	bool warmup = iteration < CONFIG_SENSWEAR_PULSE_WARMUP_REPETITIONS;
	uint32_t trial_id = warmup ? iteration : iteration - CONFIG_SENSWEAR_PULSE_WARMUP_REPETITIONS;
	int result = reset_trial(trial_id, UINT32_C(0x10000000));

	pulse_metrics_system_heap_peak_reset();
	if (result == 0) {
		result = pulse_benchmark_prepare_no_contact(&node_state, peer_id);
	}
	if (result != 0) {
		pulse_event_job.infrastructure_result = result;
		return;
	} else {
		uint32_t initial_hash = pulse_metrics_hash32(&node_state.head, sizeof(node_state.head));

		atomic_set(&measured_stages, 0);
		atomic_set(&link_dropped, 0);
		pulse_metrics_event_begin(&metric_record,
								  pulse_event_path_NoContact,
								  pulse_metric_role_Local,
								  pulse_connection_start_Disconnected,
								  trial_id,
								  0U);
		atomic_set(&measurement_state, 1);
		metric_record.event_index = total + iteration;
		metric_record.warmup = warmup;
		metric_record.input_batch_id = 2U;
		metric_record.head_version_before = node_state.head_version;
		metric_record.initial_head_hash = initial_hash;

		measured_stage_begin(MEASURED_STAGE_SCAN, pulse_stage_BleScan, pulse_marker_stage_Control);
		result = pulse_link_probe_peer(&link_manager,
									   NULL,
									   NULL,
									   K_MSEC(CONFIG_SENSWEAR_PULSE_EXCHANGE_TIMEOUT_MS));
		if (result == 0) {
			result = pulse_benchmark_execute_no_contact(&node_state,
														&compute_workspace,
														&local_gradient,
														&metric_record,
														peer_id);
		}
		if (result != 0) {
			metric_record.head_version_after = node_state.head_version;
		}
		(void) complete_measured_event(result == 0,
									   result == 0 ? pulse_failure_None
												   : failure_from_error(result));
		/* FNV diagnostics are intentionally outside the GPIO-delimited event. */
		metric_record.result_head_hash =
			pulse_metrics_hash32(&node_state.head, sizeof(node_state.head));
	}
	metric_record.event_index = total + iteration;
	metric_record.warmup = warmup;
	metric_record.input_batch_id = 2U;
	metric_record.connection_start = pulse_connection_start_Disconnected;
	metric_record.stack_peak_bytes = event_workqueue_stack_peak();
	metric_record.system_heap_peak_bytes = pulse_metrics_system_heap_peak_bytes();
	populate_link_metrics(&metric_record);
	pulse_event_job.result = result;
	pulse_event_job.record_ready = true;
}

static int __maybe_unused run_no_contact_trials(uint64_t peer_id) {
	uint32_t total = repetition_count();
	uint32_t emitted = 0U;
	uint32_t measured_failures = 0U;
	int result = disconnect_current_link();

	if (result != 0) {
		printk("PULSE_STATUS,board_id=%s,error=infrastructure,path=no_contact,"
			   "stage=precondition_disconnect,errno=%d\n",
			   board_id,
			   result);
		return result;
	}
	k_msleep(CONFIG_SENSWEAR_PULSE_DISCOVERY_SETTLE_MS);

	for (uint32_t iteration = 0U; iteration < total; ++iteration) {
		uint32_t trial_id = iteration < CONFIG_SENSWEAR_PULSE_WARMUP_REPETITIONS
							? iteration
							: iteration - CONFIG_SENSWEAR_PULSE_WARMUP_REPETITIONS;
		uint64_t capture_start = 0U;
		result = paced_capture_begin(pulse_event_path_NoContact,
								 iteration,
								 total + iteration,
								 trial_id,
								 0U,
								 iteration < CONFIG_SENSWEAR_PULSE_WARMUP_REPETITIONS,
								 &capture_start);
		if (result != 0) {
			printk("PULSE_STATUS,board_id=%s,error=infrastructure,path=no_contact,"
				   "stage=capture_go,iteration=%u,errno=%d\n",
				   board_id,
				   iteration,
				   result);
			return result;
		}
		result = dispatch_event_work(PULSE_EVENT_WORK_NO_CONTACT,
									 pulse_event_path_NoContact,
									 iteration,
									 peer_id);
		int capture_result = paced_capture_finish(
			capture_start,
			pulse_event_job.record_ready ? &pulse_event_job.record_snapshot : NULL);
		if (capture_result != 0) {
			printk("PULSE_STATUS,board_id=%s,error=infrastructure,path=no_contact,"
				   "stage=capture_dump,iteration=%u,errno=%d\n",
				   board_id,
				   iteration,
				   capture_result);
			return capture_result;
		}

		if (result != 0) {
			printk("PULSE_STATUS,board_id=%s,error=infrastructure,path=no_contact,"
				   "stage=event_work_submit,iteration=%u,errno=%d\n",
				   board_id,
				   iteration,
				   result);
			return result;
		}
		if (pulse_event_job.infrastructure_result != 0 || !pulse_event_job.record_ready) {
			result = pulse_event_job.infrastructure_result != 0
						 ? pulse_event_job.infrastructure_result
						 : -EIO;
			printk("PULSE_STATUS,board_id=%s,error=infrastructure,path=no_contact,"
				   "stage=prepare,iteration=%u,errno=%d\n",
				   board_id,
				   iteration,
				   result);
			return result;
		}
		result = emit_record(&pulse_event_job.record_snapshot);
		if (result != 0) {
			return result;
		}
		++emitted;
		if (pulse_event_job.result != 0) {
			printk("PULSE_STATUS,board_id=%s,error=measured_event,path=no_contact,"
				   "iteration=%u,errno=%d\n",
				   board_id,
				   iteration,
				   pulse_event_job.result);
			++measured_failures;
		}
	}
	campaign_measured_failure_count += measured_failures;
	printk("PULSE_STATUS,board_id=%s,state=block_complete,path=no_contact,rows=%u,"
		   "measured_failures=%u\n",
		   board_id,
		   emitted,
		   measured_failures);
	return emitted == total ? 0 : -EIO;
}

static int disconnect_current_link(void) {
	int result;

	(void) pulse_protocol_client_reset(&protocol_client);
	k_event_clear(&app_events, APP_EVENT_LINK_DISCONNECTED);
	result = pulse_link_disconnect(&link_manager);
	if (result != 0 && result != -ENOTCONN) {
		return result;
	}
	result = pulse_link_wait_disconnected(&link_manager,
										  K_MSEC(CONFIG_SENSWEAR_PULSE_EXCHANGE_TIMEOUT_MS));
	if (app_connection != NULL) {
		bt_conn_unref(app_connection);
		app_connection = NULL;
	}
	(void) pulse_protocol_client_reset(&protocol_client);
	return result;
}

static int reconnect_and_discover(int64_t deadline, bool measured) {
	const struct pulse_link_timing_callbacks timing = {
		.started = reconnect_started,
		.finished = reconnect_finished,
		.user_data = NULL,
	};
	int result;

	k_event_clear(&app_events,
				  APP_EVENT_LINK_CONNECTED | APP_EVENT_LINK_DISCONNECTED | APP_EVENT_SCAN_FOUND |
					  APP_EVENT_PROTOCOL_MASK);
	atomic_set(&link_dropped, 0);
	result = pulse_link_reconnect(&link_manager, measured ? &timing : NULL);
	if (result != 0) {
		return result;
	}
	result = pulse_link_wait_connected(&link_manager, remaining_timeout(deadline));
	if (result != 0) {
		return result;
	}
	app_connection = pulse_link_connected_ref(&link_manager);
	if (app_connection == NULL) {
		return -ENOTCONN;
	}
	result = discover_protocol_client(deadline);
	if (measured) {
		measured_stage_end(MEASURED_STAGE_CONNECT, pulse_stage_BleConnect);
	}
	return result;
}

static int ensure_connected_client(void) {
	if (connected_client_precondition()) {
		return 0;
	}
	(void) disconnect_current_link();
	return reconnect_and_discover(exchange_deadline(), false);
}

static int release_responder_export(void) {
	int64_t deadline;
	int result;

	if (!pulse_protocol_client_remote_session_started_get(&protocol_client)) {
		return 0;
	}

	k_event_clear(&app_events,
				  APP_EVENT_CLIENT_RESULT_RELEASED | APP_EVENT_CLIENT_ERROR |
					  APP_EVENT_CLIENT_CANCELLED);
	atomic_set(&protocol_result, 0);
	result = pulse_protocol_client_release_result(&protocol_client);
	if (result != 0) {
		return result;
	}
	deadline = exchange_deadline();
	return wait_client_event(APP_EVENT_CLIENT_RESULT_RELEASED, deadline, NULL);
}

static int poll_until_gradient_ready(int64_t deadline) {
	for (;;) {
		uint32_t events = 0U;
		int result;

		if (k_uptime_get() >= deadline) {
			return -ETIMEDOUT;
		}
		k_event_clear(&app_events,
					  APP_EVENT_CLIENT_REMOTE_PROCESSING | APP_EVENT_CLIENT_GRADIENT_READY |
						  APP_EVENT_CLIENT_ERROR);
		result = pulse_protocol_client_poll_status(&protocol_client);
		if (result != 0) {
			return result;
		}
		result =
			wait_client_event(APP_EVENT_CLIENT_REMOTE_PROCESSING | APP_EVENT_CLIENT_GRADIENT_READY,
							  deadline,
							  &events);
		if (result != 0) {
			return result;
		}
		if ((events & APP_EVENT_CLIENT_GRADIENT_READY) != 0U) {
			return 0;
		}
		k_msleep(PULSE_STATUS_POLL_MS);
	}
}

static int wait_until_remote_cancelled(int64_t deadline) {
	int result = wait_client_event(APP_EVENT_CLIENT_REMOTE_PROCESSING, deadline, NULL);

	if (result != 0) {
		return result;
	}

	for (;;) {
		uint32_t events;

		if (k_uptime_get() >= deadline) {
			return -ETIMEDOUT;
		}
		k_event_clear(&app_events,
					  APP_EVENT_CLIENT_REMOTE_PROCESSING | APP_EVENT_CLIENT_CANCELLED |
						  APP_EVENT_CLIENT_ERROR | APP_EVENT_LINK_DISCONNECTED);
		result = pulse_protocol_client_poll_status(&protocol_client);
		if (result != 0) {
			return result;
		}
		events = k_event_wait_safe(&app_events,
								   APP_EVENT_CLIENT_REMOTE_PROCESSING | APP_EVENT_CLIENT_CANCELLED |
									   APP_EVENT_CLIENT_ERROR | APP_EVENT_LINK_DISCONNECTED,
								   false,
								   remaining_timeout(deadline));
		if ((events & APP_EVENT_CLIENT_CANCELLED) != 0U) {
			return 0;
		}
		if ((events & APP_EVENT_LINK_DISCONNECTED) != 0U) {
			return -ENOTCONN;
		}
		if ((events & APP_EVENT_CLIENT_ERROR) != 0U) {
			int error = (int) atomic_get(&protocol_result);

			return error != 0 ? error : -EPROTO;
		}
		if ((events & APP_EVENT_CLIENT_REMOTE_PROCESSING) == 0U) {
			return -ETIMEDOUT;
		}
		k_msleep(PULSE_STATUS_POLL_MS);
	}
}

static int cancel_remote_session(int64_t deadline) {
	k_event_clear(&app_events,
				  APP_EVENT_CLIENT_REMOTE_PROCESSING | APP_EVENT_CLIENT_CANCELLED |
					  APP_EVENT_CLIENT_ERROR | APP_EVENT_LINK_DISCONNECTED);
	atomic_set(&protocol_result, 0);
	int result = pulse_protocol_client_cancel(&protocol_client);

	return result == 0 ? wait_until_remote_cancelled(deadline) : result;
}

static int prepare_accepted_trial(uint32_t trial_id, uint32_t salt, uint64_t peer_id) {
	int result = reset_trial(trial_id, salt);

	if (result != 0) {
		return result;
	}
	return pulse_benchmark_prepare_accepted(&node_state, peer_id);
}

static int run_exchange_body(uint64_t peer_id, uint32_t session_id, int64_t deadline) {
	pulse_peer_selection_t selection;
	uint32_t payload_crc32 = 0U;
	int result;

	result = pulse_benchmark_select_peer(&node_state, peer_id, &metric_record, &selection);
	if (result != 0) {
		return result;
	}
	if (selection.action != PULSE_PEER_SELECTION_CONTACT) {
		return -ECANCELED;
	}

	pulse_metrics_stage_begin(&metric_record,
							  pulse_stage_HeadSerialization,
							  pulse_marker_stage_Serialization);
	result = pulse_protocol_head_serialize(&node_state.head,
										   (uint8_t*) responder_head.parameters,
										   sizeof(responder_head),
										   &payload_crc32);
	pulse_metrics_stage_end(&metric_record, pulse_stage_HeadSerialization);
	if (result != 0) {
		return result;
	}

	measured_stage_begin(MEASURED_STAGE_HEAD_TRANSFER,
						 pulse_stage_HeadTransfer,
						 pulse_marker_stage_Radio);
	result = pulse_protocol_client_send_head(&protocol_client,
											 session_id,
											 node_state.head_version,
											 (const uint8_t*) responder_head.parameters,
											 sizeof(responder_head),
											 payload_crc32);
	if (result != 0) {
		return result;
	}
	result = wait_client_event(APP_EVENT_CLIENT_HEAD_SENT, deadline, NULL);
	if (result != 0) {
		return result;
	}

	result = pulse_benchmark_compute_gradient(&node_state.head,
											  &pulse_fixture_batches[2],
											  &compute_workspace,
											  &local_gradient,
											  NULL,
											  &metric_record,
											  pulse_stage_InitiatorEncoder,
											  pulse_stage_InitiatorHeadBackward);
	if (result != 0) {
		return result;
	}
	result = poll_until_gradient_ready(deadline);
	if (result != 0) {
		return result;
	}

	measured_stage_begin(MEASURED_STAGE_GRADIENT_TRANSFER,
						 pulse_stage_GradientTransfer,
						 pulse_marker_stage_Radio);
	result = pulse_protocol_client_receive_gradient(&protocol_client,
													(uint8_t*) responder_head.parameters,
													sizeof(responder_head));
	if (result != 0) {
		return result;
	}
	result = wait_client_event(APP_EVENT_CLIENT_GRADIENT_RECEIVED, deadline, NULL);
	if (result != 0) {
		return result;
	}
	/* ACK is the responder's explicit proof that the complete gradient reached this peer. */
	result = pulse_protocol_client_acknowledge(&protocol_client);
	if (result != 0) {
		return result;
	}
	result = wait_client_event(APP_EVENT_CLIENT_COMPLETE, deadline, NULL);
	if (result != 0) {
		return result;
	}

	pulse_metrics_stage_begin(&metric_record,
							  pulse_stage_GradientDeserialization,
							  pulse_marker_stage_Serialization);
	result = pulse_protocol_gradient_deserialize((const uint8_t*) responder_head.parameters,
												 sizeof(responder_head),
												 &remote_gradient);
	pulse_metrics_stage_end(&metric_record, pulse_stage_GradientDeserialization);
	if (result != 0) {
		return result;
	}

	result = pulse_benchmark_finish_initiator(&node_state,
											  peer_id,
											  &local_gradient,
											  &remote_gradient,
											  &metric_record);
	if (result != 0) {
		return result;
	}
	return 0;
}

static void begin_initiator_event(enum pulse_event_path path,
								  uint32_t event_index,
								  uint32_t trial_id,
								  uint32_t session_id,
								  bool warmup,
								  uint32_t initial_hash) {
	pulse_metrics_system_heap_peak_reset();
	atomic_set(&measured_stages, 0);
	atomic_set(&link_dropped, 0);
	pulse_metrics_event_begin(&metric_record,
							  path,
							  pulse_metric_role_Initiator,
							  path == pulse_event_path_AcceptedDiscovery
								  ? pulse_connection_start_Disconnected
								  : pulse_connection_start_Connected,
							  trial_id,
							  session_id);
	atomic_set(&measurement_state, 1);
	metric_record.event_index = event_index;
	metric_record.warmup = warmup;
	metric_record.input_batch_id = 2U;
	metric_record.head_version_before = node_state.head_version;
	metric_record.initial_head_hash = initial_hash;
	metric_record.head_payload_bytes = PULSE_PROTOCOL_TENSOR_BYTES;
	metric_record.gradient_payload_bytes = PULSE_PROTOCOL_TENSOR_BYTES;
}

static void run_accepted_trial(enum pulse_event_path path, uint32_t iteration, uint64_t peer_id) {
	uint32_t total = repetition_count();
	uint32_t session_base = path == pulse_event_path_AcceptedConnected ? 1U : total + 1U;
	uint32_t event_base = path == pulse_event_path_AcceptedConnected ? 2U * total : 3U * total;
	uint32_t salt = path == pulse_event_path_AcceptedConnected ? UINT32_C(0x20000000)
															   : UINT32_C(0x30000000);
	bool warmup = iteration < CONFIG_SENSWEAR_PULSE_WARMUP_REPETITIONS;
	uint32_t trial_id = warmup ? iteration : iteration - CONFIG_SENSWEAR_PULSE_WARMUP_REPETITIONS;
	uint32_t session_id = session_base + iteration;
	bool release_ready = false;
	int cancel_result = 0;
	int release_result = 0;
	int result;

	pulse_metrics_system_heap_peak_reset();
	if (path == pulse_event_path_AcceptedDiscovery) {
		/* Autonomous capture establishes its disconnected pre-guard before the
		 * absolute event deadline. Ordinary campaigns disconnect here. */
#if defined(CONFIG_SENSWEAR_PULSE_NGMO2_AUTONOMOUS_CAPTURE)
		result = app_connection == NULL ? 0 : disconnect_current_link();
#else
		result = disconnect_current_link();
#endif
		if (result == 0) {
			k_msleep(CONFIG_SENSWEAR_PULSE_DISCOVERY_SETTLE_MS);
		}
	} else {
		result = ensure_connected_client();
	}
	if (result != 0) {
		pulse_event_job.infrastructure_result = result;
		return;
	}

	result = prepare_accepted_trial(trial_id, salt, peer_id);
	if (result != 0) {
		pulse_event_job.infrastructure_result = result;
		return;
	}
	uint32_t initial_hash = pulse_metrics_hash32(&node_state.head, sizeof(node_state.head));
	pulse_protocol_client_counters_reset(&protocol_client);
	k_event_clear(&app_events, APP_EVENT_PROTOCOL_MASK | APP_EVENT_LINK_DISCONNECTED);
	atomic_set(&protocol_result, 0);
	begin_initiator_event(path, event_base + iteration, trial_id, session_id, warmup, initial_hash);
	int64_t deadline = exchange_deadline();

	if (result == 0 && path == pulse_event_path_AcceptedDiscovery) {
		result = reconnect_and_discover(deadline, true);
	}
	if (result == 0) {
		result = run_exchange_body(peer_id, session_id, deadline);
	}
	if (result == 0) {
		release_ready = true;
	} else if (pulse_protocol_client_remote_session_started_get(&protocol_client)) {
		enum pulse_protocol_client_state state = pulse_protocol_client_state_get(&protocol_client);

		if (state == PULSE_PROTOCOL_CLIENT_COMPLETE || state == PULSE_PROTOCOL_CLIENT_READY) {
			release_ready = true;
		} else {
			/*
			 * CANCEL remains inside the failed measured transaction.  Polling
			 * until CANCELLED proves that the responder worker has stopped and
			 * lowered its gate before the out-of-measurement release is sent.
			 */
			cancel_result = cancel_remote_session(deadline);
			release_ready = cancel_result == 0;
			if (!release_ready) {
				/* Disconnect is the bounded fallback when cancellation cannot be proven. */
				(void) pulse_link_disconnect(&link_manager);
			}
		}
	}
	(void) complete_measured_event(result == 0,
								   result == 0 ? pulse_failure_None : failure_from_error(result));
	capture_record(false, result == 0);
	if (release_ready) {
		release_result = release_responder_export();
	} else if (cancel_result != 0) {
		/* Reuse the export-fallback diagnostic for a cancel-to-disconnect fallback. */
		release_result = cancel_result;
	}

	if (result != 0 || release_result != 0) {
		(void) disconnect_current_link();
		/*
		 * accepted_discovery deliberately starts every repetition disconnected.
		 * A natural scan/connect/discovery failure is therefore already ready for
		 * the next measured repetition; an out-of-gate reconnect here would turn
		 * peer absence into a campaign-aborting infrastructure error.  The
		 * connected-start path does need its retained-link precondition restored.
		 */
		pulse_event_job.reconnect_needed = path == pulse_event_path_AcceptedConnected;
	}
	pulse_event_job.result = result;
	pulse_event_job.result_release_result = release_result;
	/*
	 * RESULT_RELEASE is gate-low capture coordination rather than measured PULSE traffic.  When
	 * the exchange has already failed, a secondary release failure must not abort the campaign:
	 * disconnecting makes the responder's frozen failure row safe to print and both measured rows
	 * remain in the denominator.  A release failure after an otherwise successful exchange still
	 * invalidates capture integrity and remains an infrastructure error.
	 */
	pulse_event_job.infrastructure_result = result == 0 ? release_result : 0;
	pulse_event_job.record_ready = true;
}

static int __maybe_unused run_accepted_trials(enum pulse_event_path path, uint64_t peer_id) {
	uint32_t total = repetition_count();
	uint32_t session_base = path == pulse_event_path_AcceptedConnected ? 1U : total + 1U;
	uint32_t event_base = path == pulse_event_path_AcceptedConnected ? 2U * total : 3U * total;
	uint32_t emitted = 0U;
	uint32_t measured_failures = 0U;
	const char* path_name = path == pulse_event_path_AcceptedConnected ? "accepted_connected"
																	   : "accepted_discovery";

	for (uint32_t iteration = 0U; iteration < total; ++iteration) {
		uint32_t trial_id = iteration < CONFIG_SENSWEAR_PULSE_WARMUP_REPETITIONS
							? iteration
							: iteration - CONFIG_SENSWEAR_PULSE_WARMUP_REPETITIONS;
		uint64_t capture_start = 0U;
		int result = paced_capture_begin(path,
								  iteration,
								  event_base + iteration,
								  trial_id,
								  session_base + iteration,
								  iteration < CONFIG_SENSWEAR_PULSE_WARMUP_REPETITIONS,
								  &capture_start);
		if (result != 0) {
			printk("PULSE_STATUS,board_id=%s,error=infrastructure,path=%s,"
				   "stage=capture_go,iteration=%u,errno=%d\n",
				   board_id,
				   path_name,
				   iteration,
				   result);
			return result;
		}
		result = dispatch_event_work(PULSE_EVENT_WORK_ACCEPTED, path, iteration, peer_id);
		int capture_result = paced_capture_finish(
			capture_start,
			pulse_event_job.record_ready ? &pulse_event_job.record_snapshot : NULL);
		if (capture_result != 0) {
			printk("PULSE_STATUS,board_id=%s,error=infrastructure,path=%s,"
				   "stage=capture_dump,iteration=%u,errno=%d\n",
				   board_id,
				   path_name,
				   iteration,
				   capture_result);
			return capture_result;
		}

		if (result != 0) {
			printk("PULSE_STATUS,board_id=%s,error=infrastructure,path=%s,"
				   "stage=event_work_submit,iteration=%u,errno=%d\n",
				   board_id,
				   path_name,
				   iteration,
				   result);
			return result;
		}
		if (!pulse_event_job.record_ready) {
			result = pulse_event_job.infrastructure_result != 0
						 ? pulse_event_job.infrastructure_result
						 : -EIO;
			printk("PULSE_STATUS,board_id=%s,error=infrastructure,path=%s,stage=prepare,"
				   "iteration=%u,errno=%d\n",
				   board_id,
				   path_name,
				   iteration,
				   result);
			return result;
		}
		result = emit_record(&pulse_event_job.record_snapshot);
		if (result != 0) {
			return result;
		}
		++emitted;
		if (pulse_event_job.result != 0) {
			printk("PULSE_STATUS,board_id=%s,error=measured_event,path=%s,iteration=%u,"
				   "errno=%d\n",
				   board_id,
				   path_name,
				   iteration,
				   pulse_event_job.result);
			++measured_failures;
		}
		if (pulse_event_job.result != 0 && pulse_event_job.result_release_result != 0) {
			printk("PULSE_STATUS,board_id=%s,state=result_release_fallback,path=%s,"
				   "iteration=%u,errno=%d\n",
				   board_id,
				   path_name,
				   iteration,
				   pulse_event_job.result_release_result);
		}
		if (pulse_event_job.infrastructure_result != 0) {
			printk("PULSE_STATUS,board_id=%s,error=infrastructure,path=%s,"
				   "stage=result_release,iteration=%u,errno=%d\n",
				   board_id,
				   path_name,
				   iteration,
				   pulse_event_job.infrastructure_result);
			return pulse_event_job.infrastructure_result;
		}
		if (pulse_event_job.reconnect_needed) {
			result = reconnect_and_discover(exchange_deadline(), false);
			if (result != 0) {
				printk("PULSE_STATUS,board_id=%s,error=infrastructure,path=%s,"
					   "stage=reconnect,iteration=%u,errno=%d\n",
					   board_id,
					   path_name,
					   iteration,
					   result);
				return result;
			}
		}
	}
	campaign_measured_failure_count += measured_failures;
	printk("PULSE_STATUS,board_id=%s,state=block_complete,path=%s,rows=%u,"
		   "measured_failures=%u\n",
		   board_id,
		   path_name,
		   emitted,
		   measured_failures);
	return emitted == total ? 0 : -EIO;
}

static int initialize_protocol(void) {
	if (app_role == PULSE_LINK_ROLE_INITIATOR) {
		const struct pulse_protocol_client_config config = {
			.model_version = 1U,
			.event = protocol_client_event,
			.user_data = NULL,
		};
		int result = pulse_protocol_client_init(&protocol_client, &config);

		if (result != 0) {
			return result;
		}
		return discover_protocol_client(exchange_deadline());
	}
	if (app_role == PULSE_LINK_ROLE_RESPONDER) {
		const struct pulse_protocol_server_config config = {
			.model_version = 1U,
			.serialized_head_receive_buffer = (uint8_t*) responder_head.parameters,
			.serialized_head_receive_buffer_size = sizeof(responder_head),
			.callbacks =
				{
#if defined(CONFIG_SENSWEAR_PULSE_NGMO2_AUTONOMOUS_CAPTURE)
					.admit_session = autonomous_responder_admit_session,
#endif
					.session_started = server_session_started,
					.head_ready = server_head_ready,
					.gradient_received = server_gradient_received,
					.session_complete = server_session_complete,
					.result_released = server_result_released,
					.session_cancelled = server_session_cancelled,
				},
			.user_data = NULL,
		};
		int result = pulse_protocol_server_init(&config);

		if (result == 0) {
			pulse_protocol_server_counters_reset();
		}
		return result;
	}
	return -EINVAL;
}

static void release_connection_reference(void) {
	if (app_connection != NULL) {
		bt_conn_unref(app_connection);
		app_connection = NULL;
	}
}

static int responder_reconnect(void) {
	int result;

	/*
	 * A failed measured session can still have a healthy BLE connection after
	 * its gate-low RESULT_RELEASE.  Retire that link before asking the manager
	 * to advertise again; reconnect otherwise correctly rejects it as busy.
	 * Keep app_connection alive through the disconnect callback so the
	 * protocol server observes the link loss before its reset.
	 */
	result = pulse_link_disconnect(&link_manager);
	if (result == 0 || result == -EALREADY) {
		result = pulse_link_wait_disconnected(&link_manager,
											  K_MSEC(CONFIG_SENSWEAR_PULSE_EXCHANGE_TIMEOUT_MS));
		if (result != 0) {
			return result;
		}
	} else if (result != -ENOTCONN) {
		return result;
	}
	release_connection_reference();
	result = pulse_protocol_server_reset();
	if (result != 0) {
		return result;
	}
	k_event_clear(&app_events, APP_EVENT_LINK_CONNECTED | APP_EVENT_LINK_DISCONNECTED);
	atomic_set(&link_dropped, 0);
	result = pulse_link_reconnect(&link_manager, NULL);
	if (result != 0) {
		return result;
	}
	result = pulse_link_wait_connected(&link_manager, K_FOREVER);
	if (result != 0) {
		return result;
	}
	app_connection = pulse_link_connected_ref(&link_manager);
	return app_connection != NULL ? 0 : -ENOTCONN;
}

static bool freeze_responder_record(void) {
	if (!atomic_cas(&responder_record_pending, 1, 2)) {
		return false;
	}
	metric_record.stack_peak_bytes = event_workqueue_stack_peak();
	metric_record.system_heap_peak_bytes = pulse_metrics_system_heap_peak_bytes();
	populate_link_metrics(&metric_record);
	populate_protocol_metrics(&metric_record, true);
	/* Freeze the exact accepted-session identity and outcome before any reset or reconnect. */
	pulse_event_job.record_snapshot = metric_record;
	return true;
}

static int __maybe_unused reset_frozen_responder_session(void) {
	int result;

	/*
	 * Clear every old-session side channel while the protocol still rejects a
	 * new HEAD_BEGIN.  Reset is deliberately last: it is the transition that
	 * reopens the server's IDLE state to the next initiator request.
	 */
	pulse_protocol_server_counters_reset();
	k_event_clear(&app_events,
				  APP_EVENT_SERVER_STARTED | APP_EVENT_SERVER_HEAD_READY |
					  APP_EVENT_SERVER_GRADIENT_RECEIVED | APP_EVENT_SERVER_COMPLETE |
					  APP_EVENT_SERVER_CANCELLED | APP_EVENT_SERVER_RESULT_RELEASED);
	/* Clear the old application-side claim before reset reopens HEAD_BEGIN. */
	atomic_set(&responder_record_pending, 0);
	result = pulse_protocol_server_reset();
	return result;
}

static void run_responder_event(void) {
	uint32_t events =
		k_event_wait_safe(&app_events,
						  APP_EVENT_SERVER_HEAD_READY | APP_EVENT_SERVER_CANCELLED |
							  APP_EVENT_SERVER_RESULT_RELEASED | APP_EVENT_LINK_DISCONNECTED,
						  false,
						  K_FOREVER);
	bool result_released = (events & APP_EVENT_SERVER_RESULT_RELEASED) != 0U;
	int release_result = 0;
	int result = 0;

	if ((events & (APP_EVENT_SERVER_CANCELLED | APP_EVENT_LINK_DISCONNECTED)) != 0U) {
		result = (events & APP_EVENT_LINK_DISCONNECTED) != 0U ? -ENOTCONN : -ECANCELED;
	} else if ((events & APP_EVENT_SERVER_HEAD_READY) != 0U) {
		uint32_t payload_crc32 = 0U;

		pulse_metrics_stage_begin(&metric_record,
								  pulse_stage_HeadDeserialization,
								  pulse_marker_stage_Serialization);
		result = pulse_protocol_head_deserialize((const uint8_t*) responder_head.parameters,
												 sizeof(responder_head),
												 &responder_head);
		pulse_metrics_stage_end(&metric_record, pulse_stage_HeadDeserialization);
		if (result == 0) {
			/* The wire CRC already validated this tensor; avoid an extra FNV pass in-gate. */
			metric_record.initial_head_hash = 0U;
			metric_record.result_head_hash = 0U;
			result = pulse_benchmark_compute_responder(&responder_head,
													   &compute_workspace,
													   &remote_gradient,
													   &metric_record,
													   NULL);
		}
		if (result == 0) {
			pulse_metrics_stage_begin(&metric_record,
									  pulse_stage_GradientSerialization,
									  pulse_marker_stage_Serialization);
			result = pulse_protocol_gradient_serialize(&remote_gradient,
													   (uint8_t*) responder_head.parameters,
													   sizeof(responder_head),
													   &payload_crc32);
			pulse_metrics_stage_end(&metric_record, pulse_stage_GradientSerialization);
		}
		if (result == 0) {
			measured_stage_begin(MEASURED_STAGE_GRADIENT_TRANSFER,
								 pulse_stage_GradientTransfer,
								 pulse_marker_stage_Radio);
			result =
				pulse_protocol_server_publish_gradient(responder_transfer.session_id,
													   responder_transfer.head_version,
													   (const uint8_t*) responder_head.parameters,
													   sizeof(responder_head),
													   payload_crc32);
		}

		if (result == 0) {
			events = k_event_wait_safe(&app_events,
									   APP_EVENT_SERVER_COMPLETE | APP_EVENT_SERVER_TERMINAL |
										   APP_EVENT_SERVER_RESULT_RELEASED,
									   false,
									   K_MSEC(CONFIG_SENSWEAR_PULSE_EXCHANGE_TIMEOUT_MS));
			result_released = (events & APP_EVENT_SERVER_RESULT_RELEASED) != 0U;
			if (events == 0U) {
				result = -ETIMEDOUT;
			} else if ((events & APP_EVENT_SERVER_COMPLETE) != 0U) {
				/* The ACK callback already closed the successful measurement gate. */
				result = 0;
			} else if ((events & APP_EVENT_LINK_DISCONNECTED) != 0U) {
				result = -ENOTCONN;
			} else if ((events & APP_EVENT_SERVER_CANCELLED) != 0U) {
				result = -ECANCELED;
			} else {
				/* RESULT_RELEASE is invalid before COMPLETE or application-acknowledged cancel. */
				result = -EPROTO;
			}
		}
	}

	if (!complete_measured_event(result == 0,
								 result == 0 ? pulse_failure_None : failure_from_error(result))) {
		/*
		 * The ACK callback is allowed to own successful gate closure.  Resolve
		 * the timeout/ACK boundary from that single frozen metric outcome, and
		 * wait out its short state=2 publication window before snapshotting.
		 */
		while (atomic_get(&measurement_state) == 2) {
			k_yield();
		}
		if (metric_record.success) {
			result = 0;
		} else if (result == 0) {
			result = -EIO;
		}
	}
	/*
	 * Cancellation is a two-phase gate handshake.  The GATT CANCEL callback
	 * only asks this worker to stop; publishing CANCELLED here proves to the
	 * initiator that all responder work and its measurement gate are finished.
	 */
	if ((events & APP_EVENT_SERVER_CANCELLED) != 0U &&
		(events & APP_EVENT_LINK_DISCONNECTED) == 0U) {
		int cancel_complete = pulse_protocol_server_complete_cancel(responder_transfer.session_id);

		if (cancel_complete != 0 && cancel_complete != -EALREADY) {
			release_result = cancel_complete;
		}
	}
	if (!result_released && (events & APP_EVENT_LINK_DISCONNECTED) == 0U) {
		int64_t release_deadline = exchange_deadline();

		while (release_result == 0 && !result_released &&
			   (events & APP_EVENT_LINK_DISCONNECTED) == 0U) {
			uint32_t release_events =
				k_event_wait_safe(&app_events,
								  APP_EVENT_SERVER_CANCELLED | APP_EVENT_SERVER_RESULT_RELEASED |
									  APP_EVENT_LINK_DISCONNECTED,
								  false,
								  remaining_timeout(release_deadline));

			events |= release_events;
			if ((release_events & APP_EVENT_LINK_DISCONNECTED) != 0U) {
				release_result = -ENOTCONN;
				break;
			}
			if ((release_events & APP_EVENT_SERVER_CANCELLED) != 0U) {
				int cancel_complete =
					pulse_protocol_server_complete_cancel(responder_transfer.session_id);

				if (cancel_complete != 0 && cancel_complete != -EALREADY) {
					release_result = cancel_complete;
					break;
				}
			}
			if ((release_events & APP_EVENT_SERVER_RESULT_RELEASED) != 0U) {
				result_released = true;
				break;
			}
			if (release_events == 0U) {
				release_result = -ETIMEDOUT;
			}
		}
	}
	bool record_frozen = freeze_responder_record();
	/*
	 * A successful RESULT_RELEASE proves that the initiator gate is already low.  Without that
	 * proof, keep UART silent until the link has disconnected (or the bounded disconnect wait has
	 * elapsed after the full release timeout).  The snapshot above preserves the pre-disconnect
	 * counters and link/session identity; printing happens only after this work item returns.
	 */
	if (!result_released && app_connection != NULL) {
		(void) pulse_link_disconnect(&link_manager);
		(void) pulse_link_wait_disconnected(&link_manager,
											K_MSEC(CONFIG_SENSWEAR_PULSE_EXCHANGE_TIMEOUT_MS));
		events |= APP_EVENT_LINK_DISCONNECTED;
	}
#if defined(CONFIG_SENSWEAR_PULSE_NGMO2_AUTONOMOUS_CAPTURE)
	/* Keep the protocol terminal: reset would reopen HEAD_BEGIN for a second event. */
	int reset_result = record_frozen ? 0 : -EALREADY;
#else
	int reset_result = record_frozen ? reset_frozen_responder_session() : -EALREADY;
#endif
	pulse_event_job.record_ready = record_frozen;
	pulse_event_job.result = result;
	pulse_event_job.result_release_result = release_result;
	/* A secondary release failure cannot hide or abort an already measured exchange failure. */
	pulse_event_job.infrastructure_result = result == 0 && release_result != 0 ? release_result
																			   : reset_result;
	pulse_event_job.reconnect_needed = (events & APP_EVENT_LINK_DISCONNECTED) != 0U ||
									   app_connection == NULL || result != 0 || release_result != 0;
	if (!pulse_event_job.record_ready && pulse_event_job.infrastructure_result == 0) {
		pulse_event_job.infrastructure_result = -EACCES;
	}
}

static void pulse_event_work_handler(struct k_work* work) {
	struct pulse_event_work_context* job =
		CONTAINER_OF(work, struct pulse_event_work_context, work);

	switch (job->kind) {
	case PULSE_EVENT_WORK_LOCAL:
		run_local_trial(job->iteration);
		break;
	case PULSE_EVENT_WORK_NO_CONTACT:
		run_no_contact_trial(job->iteration, job->peer_id);
		break;
	case PULSE_EVENT_WORK_ACCEPTED:
		run_accepted_trial(job->path, job->iteration, job->peer_id);
		break;
	case PULSE_EVENT_WORK_RESPONDER:
		run_responder_event();
		break;
	default:
		job->result = -EINVAL;
		break;
	}

	if (job->record_ready && job->kind != PULSE_EVENT_WORK_RESPONDER) {
		job->record_snapshot = metric_record;
	}
}

static int dispatch_event_work(enum pulse_event_work_kind kind,
							   enum pulse_event_path path,
							   uint32_t iteration,
							   uint64_t peer_id) {
	int result;

	if (k_work_busy_get(&pulse_event_job.work) != 0U) {
		return -EBUSY;
	}
	pulse_event_job.kind = kind;
	pulse_event_job.path = path;
	pulse_event_job.iteration = iteration;
	pulse_event_job.peer_id = peer_id;
	pulse_event_job.result = 0;
	pulse_event_job.infrastructure_result = 0;
	pulse_event_job.result_release_result = 0;
	pulse_event_job.record_ready = false;
	pulse_event_job.reconnect_needed = false;

	result = k_work_submit_to_queue(&pulse_event_workqueue, &pulse_event_job.work);
	if (result != 1) {
		return result < 0 ? result : -EBUSY;
	}
	(void) k_work_flush(&pulse_event_job.work, &pulse_event_job.sync);
	return 0;
}

#if defined(CONFIG_SENSWEAR_PULSE_NGMO2_AUTONOMOUS_CAPTURE)
static int autonomous_prepare_initiator(enum pulse_event_path path) {
	int result;

	if (path == pulse_event_path_Local || path == pulse_event_path_AcceptedConnected) {
		return ensure_connected_client();
	}
	if (path != pulse_event_path_NoContact && path != pulse_event_path_AcceptedDiscovery) {
		return -EINVAL;
	}

	result = app_connection == NULL ? 0 : disconnect_current_link();
	if (result == 0) {
		k_msleep(CONFIG_SENSWEAR_PULSE_DISCOVERY_SETTLE_MS);
	}
	return result;
}

static void run_autonomous_initiator(enum pulse_event_path path, uint32_t sequence) {
	uint64_t peer_id = connected_peer_id();
	int result = autonomous_prepare_initiator(path);

	if (result == 0) {
		result = wait_for_autonomous_event_deadline();
	}
	if (result != 0) {
		autonomous_proof_fail(result);
		autonomous_hold_forever();
		return;
	}

	enum pulse_event_work_kind kind = PULSE_EVENT_WORK_ACCEPTED;
	if (path == pulse_event_path_Local) {
		kind = PULSE_EVENT_WORK_LOCAL;
	} else if (path == pulse_event_path_NoContact) {
		kind = PULSE_EVENT_WORK_NO_CONTACT;
	}
	result = dispatch_event_work(kind, path, sequence, peer_id);
	if (result != 0) {
		autonomous_proof_fail(result);
		autonomous_hold_forever();
		return;
	}

	/* Make both guards of the discovery path represent the same disconnected,
	 * advertising/scanning state. The cleanup remains inside the capture-wide
	 * incremental-energy envelope but outside the protocol timing interval. */
	if (path == pulse_event_path_AcceptedDiscovery) {
		int cleanup_result = disconnect_current_link();

		if (cleanup_result != 0 && cleanup_result != -ENOTCONN &&
			pulse_event_job.infrastructure_result == 0) {
			pulse_event_job.infrastructure_result = cleanup_result;
		}
	}
	autonomous_proof_update_from_job();
	autonomous_hold_forever();
}
#endif

static void run_responder(void) {
	for (;;) {
		k_sem_take(&responder_dispatch_sem, K_FOREVER);

#if defined(CONFIG_SENSWEAR_PULSE_NGMO2_AUTONOMOUS_CAPTURE)
		if (atomic_get(&autonomous_responder_admission) == AUTONOMOUS_RESPONDER_REJECTED) {
			autonomous_proof_fail(autonomous_responder_rejection_error());
			autonomous_hold_forever();
		}
#endif
		if (atomic_get(&responder_record_pending) != 0) {
			int result =
				dispatch_event_work(PULSE_EVENT_WORK_RESPONDER, pulse_event_path_Failed, 0U, 0U);
			int capture_result = paced_responder_release();

			if (capture_result != 0) {
#if defined(CONFIG_SENSWEAR_PULSE_NGMO2_AUTONOMOUS_CAPTURE)
				autonomous_proof_fail(capture_result);
				autonomous_hold_forever();
#else
				printk("PULSE_STATUS,board_id=%s,error=infrastructure,role=responder,"
					   "stage=capture_dump,errno=%d\n",
					   board_id,
					   capture_result);
				continue;
#endif
			}

#if defined(CONFIG_SENSWEAR_PULSE_NGMO2_AUTONOMOUS_CAPTURE)
			if (result == 0) {
				autonomous_proof_update_from_job();
			} else {
				autonomous_proof_fail(result);
			}
			autonomous_hold_forever();
#else
			if (result != 0) {
				printk("PULSE_STATUS,board_id=%s,error=infrastructure,role=responder,"
					   "stage=event_work_submit,errno=%d\n",
					   board_id,
					   result);
			} else if (pulse_event_job.record_ready) {
				result = emit_record(&pulse_event_job.record_snapshot);
				if (result == 0 && pulse_event_job.result != 0) {
					printk("PULSE_STATUS,board_id=%s,error=measured_event,role=responder,"
						   "errno=%d\n",
						   board_id,
						   pulse_event_job.result);
				}
				if (result == 0 && pulse_event_job.result != 0 &&
					pulse_event_job.result_release_result != 0) {
					printk("PULSE_STATUS,board_id=%s,state=result_release_fallback,"
						   "role=responder,errno=%d\n",
						   board_id,
						   pulse_event_job.result_release_result);
				}
				if (result == 0 && pulse_event_job.infrastructure_result != 0) {
					printk("PULSE_STATUS,board_id=%s,error=infrastructure,role=responder,"
						   "stage=result_release,errno=%d\n",
						   board_id,
						   pulse_event_job.infrastructure_result);
				}
			} else if (pulse_event_job.infrastructure_result != 0) {
				printk("PULSE_STATUS,board_id=%s,error=infrastructure,role=responder,"
					   "stage=result_release,errno=%d\n",
					   board_id,
					   pulse_event_job.infrastructure_result);
			}
#endif
		}

		if (pulse_event_job.reconnect_needed || atomic_get(&link_dropped) != 0 ||
			app_connection == NULL) {
			int result = responder_reconnect();

			pulse_event_job.reconnect_needed = result != 0;
			if (result != 0) {
				k_msleep(CONFIG_SENSWEAR_PULSE_EVENT_GAP_MS);
				k_sem_give(&responder_dispatch_sem);
			}
		}
	}
}
#else
static int baseline_reconnect_start(void) {
	int result;

	if (app_connection != NULL) {
		bt_conn_unref(app_connection);
		app_connection = NULL;
	}
	k_event_clear(&app_events, APP_EVENT_LINK_CONNECTED | APP_EVENT_LINK_DISCONNECTED);
	atomic_set(&link_dropped, 0);
	result = pulse_link_reconnect(&link_manager, NULL);
	return result == -EALREADY ? 0 : result;
}

static void run_baseline_monitor(void) {
	printk("PULSE_STATUS,board_id=%s,state=matched_baseline_idle,connection=retained\n", board_id);
	for (;;) {
		(void) k_event_wait_safe(&app_events, APP_EVENT_LINK_DISCONNECTED, false, K_FOREVER);
		printk("PULSE_STATUS,board_id=%s,error=infrastructure,role=%s,"
			   "stage=baseline_link_loss\n",
			   board_id,
			   link_role_name(app_role));

		for (;;) {
			int result = baseline_reconnect_start();

			if (result == 0) {
				for (;;) {
					result =
						pulse_link_wait_connected(&link_manager,
												  K_MSEC(
													  CONFIG_SENSWEAR_PULSE_EXCHANGE_TIMEOUT_MS));
					if (result == 0) {
						break;
					}
					if (result != -EAGAIN && result != -ETIMEDOUT) {
						break;
					}
					printk("PULSE_STATUS,board_id=%s,state=baseline_reconnect_wait,role=%s\n",
						   board_id,
						   link_role_name(app_role));
				}
			}

			if (result == 0) {
				app_connection = pulse_link_connected_ref(&link_manager);
				if (app_connection != NULL) {
					break;
				}
				result = -ENOTCONN;
			}
			printk("PULSE_STATUS,board_id=%s,error=infrastructure,role=%s,"
				   "stage=baseline_reconnect,errno=%d\n",
				   board_id,
				   link_role_name(app_role),
				   result);
			k_msleep(CONFIG_SENSWEAR_PULSE_EVENT_GAP_MS);
		}
		printk("PULSE_STATUS,board_id=%s,state=baseline_link_recovered,role=%s\n",
			   board_id,
			   link_role_name(app_role));
	}
}
#endif

int main(void) {
	struct pulse_link_snapshot initial_snapshot = {0};
	int marker_result;
	int sensing_result;
	int link_init_result;
#if defined(CONFIG_SENSWEAR_PULSE_NGMO2_AUTONOMOUS_CAPTURE)
	enum pulse_event_path autonomous_path = autonomous_capture_path();
	int capture_log_result;

	autonomous_boot_ms = k_uptime_get();
#endif

	read_board_id();
#if defined(CONFIG_SENSWEAR_PULSE_NGMO2_AUTONOMOUS_CAPTURE)
	capture_log_result = pulse_capture_log_init();
#if defined(CONFIG_SENSWEAR_PULSE_NGMO2_EXPORT_ONLY)
	/* Export the complete compile-time measurement contract before the proof
	 * records so the host parser can bind every recovered record to it.  A
	 * zero snapshot deliberately reports unavailable link observations: this
	 * branch must not start sensing, Bluetooth, protocol, or model work. */
	marker_result = pulse_metrics_init();
	dump_run_metadata(&initial_snapshot);
	printk("PULSE_STATUS,board_id=%s,state=capture_export_metadata,"
		   "marker_errno=%d,role=unresolved,link=not_initialized\n",
		   board_id,
		   marker_result);
	if (capture_log_result == 0) {
		pulse_capture_log_dump(board_id);
	} else {
		printk("PULSE_CAPTURE_STATUS,board_id=%s,error=storage_init,errno=%d\n",
			   board_id,
			   capture_log_result);
	}
	autonomous_hold_forever();
	return capture_log_result;
#endif
	if (capture_log_result != 0 ||
		pulse_capture_log_next_sequence(autonomous_path, &autonomous_sequence) != 0 ||
		autonomous_sequence >= repetition_count()) {
		autonomous_hold_forever();
		return capture_log_result;
	}
#endif
	marker_result = pulse_metrics_init();
	sensing_result = start_sensing_baseline();
	link_init_result = initialize_link(&initial_snapshot);
#if defined(CONFIG_SENSWEAR_PULSE_NGMO2_AUTONOMOUS_CAPTURE)
	/* Role election happens during link setup. Initialize the proof immediately
	 * afterward, before any setup error can be persisted or returned. */
	autonomous_proof_prepare(autonomous_path, autonomous_sequence, app_role);
#endif
	dump_run_metadata(&initial_snapshot);
	printk("PULSE_STATUS,board_id=%s,marker_errno=%d,sensing_errno=%d,link_errno=%d,role=%s\n",
		   board_id,
		   marker_result,
		   sensing_result,
		   link_init_result,
		   link_role_name(app_role));
	if (marker_result != 0 || sensing_result != 0 || link_init_result != 0) {
		int result = marker_result != 0 ? marker_result
								: (sensing_result != 0 ? sensing_result : link_init_result);
#if defined(CONFIG_SENSWEAR_PULSE_NGMO2_AUTONOMOUS_CAPTURE)
		autonomous_proof_fail(result);
		autonomous_hold_forever();
#endif
		return result;
	}

#if defined(CONFIG_SENSWEAR_PULSE_ALGORITHM)
	if (pulse_encoder_artifact_validate(&pulse_fixture_encoder) != PULSE_STATUS_OK ||
		pulse_head_artifact_validate(&pulse_fixture_initial_head) != PULSE_STATUS_OK ||
		pulse_benchmark_node_init(&node_state, UINT32_C(0x05eeda11)) != 0) {
		printk("PULSE_STATUS,board_id=%s,error=fixture_validation\n", board_id);
#if defined(CONFIG_SENSWEAR_PULSE_NGMO2_AUTONOMOUS_CAPTURE)
		autonomous_proof_fail(-EINVAL);
		autonomous_hold_forever();
#endif
		return -EINVAL;
	}

	initialize_event_workqueue();
	int protocol_init_result = initialize_protocol();
	printk("PULSE_STATUS,board_id=%s,protocol_errno=%d\n", board_id, protocol_init_result);
	if (protocol_init_result != 0) {
#if defined(CONFIG_SENSWEAR_PULSE_NGMO2_AUTONOMOUS_CAPTURE)
		autonomous_proof_fail(protocol_init_result);
		autonomous_hold_forever();
#endif
		return protocol_init_result;
	}

#if defined(CONFIG_SENSWEAR_PULSE_CORRECTNESS)
	int correctness_result = pulse_correctness_run(board_id);
	if (correctness_result != 0) {
#if defined(CONFIG_SENSWEAR_PULSE_NGMO2_AUTONOMOUS_CAPTURE)
		autonomous_proof_fail(correctness_result);
		autonomous_hold_forever();
#endif
		return correctness_result;
	}
#endif

#if defined(CONFIG_SENSWEAR_PULSE_NGMO2_AUTONOMOUS_CAPTURE)
	if (app_role == PULSE_LINK_ROLE_RESPONDER) {
		if (autonomous_path == pulse_event_path_Local ||
			autonomous_path == pulse_event_path_NoContact) {
			autonomous_proof_mark_passive();
		}
		run_responder();
	} else {
		run_autonomous_initiator(autonomous_path, autonomous_sequence);
	}
#else
	k_msleep(CONFIG_SENSWEAR_PULSE_START_DELAY_MS);
	if (app_role == PULSE_LINK_ROLE_RESPONDER) {
		run_responder();
	} else {
		uint64_t peer_id = connected_peer_id();
		int campaign_result;

		campaign_result = run_local_trials();
#if !defined(CONFIG_SENSWEAR_PULSE_KEYSIGHT_FIG4A_CAPTURE)
		if (campaign_result == 0) {
			campaign_result = run_no_contact_trials(peer_id);
		}
#endif
		if (campaign_result == 0) {
			campaign_result = run_accepted_trials(pulse_event_path_AcceptedConnected, peer_id);
		}
#if !defined(CONFIG_SENSWEAR_PULSE_KEYSIGHT_FIG4A_CAPTURE)
		if (campaign_result == 0) {
			campaign_result = run_accepted_trials(pulse_event_path_AcceptedDiscovery, peer_id);
		}
#endif
		if (campaign_result == 0) {
			printk("PULSE_STATUS,board_id=%s,state=campaign_complete,measured_failures=%u\n",
				   board_id,
				   campaign_measured_failure_count);
		} else {
			printk("PULSE_STATUS,board_id=%s,state=campaign_failed,errno=%d\n",
				   board_id,
				   campaign_result);
		}
	}
#endif
#else
	run_baseline_monitor();
#endif

	for (;;) {
		k_sleep(K_FOREVER);
	}
	return 0;
}
