/*
 * Copyright (c) 2026
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pulse_capture_log.h"

#include <errno.h>
#include <stddef.h>
#include <string.h>

#include <zephyr/device.h>
#include <zephyr/drivers/flash.h>
#include <zephyr/kvss/nvs.h>
#include <zephyr/storage/flash_map.h>
#include <zephyr/sys/printk.h>
#include <zephyr/sys/util.h>

#define PULSE_CAPTURE_LOG_FIRST_ID UINT16_C(0x0100)
#define PULSE_CAPTURE_LOG_IDS_PER_PATH UINT16_C(0x0080)
#define PULSE_CAPTURE_LOG_PATH_COUNT 4U
#define PULSE_CAPTURE_NVS_ATE_BYTES 8U
#define PULSE_CAPTURE_NVS_DATA_CRC_BYTES 4U

BUILD_ASSERT(sizeof(struct pulse_capture_proof) == 40U,
			 "capture proof layout changed; re-audit the four-path NVS capacity");
BUILD_ASSERT(IS_ENABLED(CONFIG_NVS_DATA_CRC),
			 "capture proofs require CONFIG_NVS_DATA_CRC");
BUILD_ASSERT(CONFIG_SENSWEAR_PULSE_WARMUP_REPETITIONS +
				 CONFIG_SENSWEAR_PULSE_REPETITIONS <=
			 PULSE_CAPTURE_LOG_IDS_PER_PATH,
			 "capture repetition count exceeds the reserved NVS IDs per path");

static struct nvs_fs capture_fs;
static const struct flash_area* capture_partition;
static bool capture_fs_ready;

static uint32_t capture_count(void) {
	return CONFIG_SENSWEAR_PULSE_WARMUP_REPETITIONS +
		   CONFIG_SENSWEAR_PULSE_REPETITIONS;
}

static size_t aligned_size(size_t size, size_t alignment) {
	return ((size + alignment - 1U) / alignment) * alignment;
}

static bool capture_capacity_sufficient(size_t write_block_size,
									uint32_t sector_size,
									uint16_t sector_count) {
	if (write_block_size == 0U || sector_count < 2U) {
		return false;
	}

	size_t ate_size = aligned_size(PULSE_CAPTURE_NVS_ATE_BYTES, write_block_size);
	size_t data_size = aligned_size(sizeof(struct pulse_capture_proof) +
									  PULSE_CAPTURE_NVS_DATA_CRC_BYTES,
								  write_block_size);
	if (sector_size <= 2U * ate_size || data_size + ate_size > sector_size) {
		return false;
	}

	size_t records_per_sector = (sector_size - 2U * ate_size) / (data_size + ate_size);
	size_t record_capacity = (size_t) (sector_count - 1U) * records_per_sector;
	size_t records_required = (size_t) PULSE_CAPTURE_LOG_PATH_COUNT * capture_count();

	return record_capacity >= records_required;
}

static bool persistent_path(enum pulse_event_path path) {
	return path >= pulse_event_path_Local && path <= pulse_event_path_AcceptedDiscovery;
}

static uint16_t proof_id(enum pulse_event_path path, uint32_t sequence) {
	return (uint16_t) (PULSE_CAPTURE_LOG_FIRST_ID +
				   (uint16_t) path * PULSE_CAPTURE_LOG_IDS_PER_PATH +
				   (uint16_t) sequence);
}

static bool valid_proof(const struct pulse_capture_proof* proof,
						enum pulse_event_path path,
						uint32_t sequence) {
	return proof->magic == PULSE_CAPTURE_PROOF_MAGIC &&
		   proof->schema_version == PULSE_CAPTURE_PROOF_SCHEMA_VERSION &&
		   proof->record_size == sizeof(*proof) && proof->event_path == (uint8_t) path &&
		   proof->sequence == sequence;
}

int pulse_capture_log_init(void) {
	struct flash_pages_info page;
	int result;

	if (capture_fs_ready) {
		return 0;
	}

	result = flash_area_open(PARTITION_ID(storage_partition), &capture_partition);
	if (result != 0) {
		return result;
	}
	if (!device_is_ready(capture_partition->fa_dev)) {
		flash_area_close(capture_partition);
		capture_partition = NULL;
		return -ENODEV;
	}
	result = flash_get_page_info_by_offs(capture_partition->fa_dev,
									 capture_partition->fa_off,
									 &page);
	if (result != 0 || page.size == 0U || capture_partition->fa_size % page.size != 0U) {
		flash_area_close(capture_partition);
		capture_partition = NULL;
		return result != 0 ? result : -EINVAL;
	}

	capture_fs.flash_device = capture_partition->fa_dev;
	capture_fs.offset = capture_partition->fa_off;
	capture_fs.sector_size = page.size;
	capture_fs.sector_count = capture_partition->fa_size / page.size;
	if (!capture_capacity_sufficient(flash_get_write_block_size(capture_partition->fa_dev),
									 capture_fs.sector_size,
									 capture_fs.sector_count)) {
		flash_area_close(capture_partition);
		capture_partition = NULL;
		return -ENOSPC;
	}

	result = nvs_mount(&capture_fs);
	if (result != 0) {
		flash_area_close(capture_partition);
		capture_partition = NULL;
		return result;
	}
	capture_fs_ready = true;
	return 0;
}

int pulse_capture_log_next_sequence(enum pulse_event_path path, uint32_t* sequence) {
	struct pulse_capture_proof proof;
	uint32_t next = 0U;

	if (!capture_fs_ready || sequence == NULL || !persistent_path(path)) {
		return -EINVAL;
	}

	for (uint32_t candidate = 0U; candidate < capture_count(); ++candidate) {
		ssize_t length = nvs_read(&capture_fs, proof_id(path, candidate), &proof, sizeof(proof));

		if (length == -ENOENT) {
			continue;
		}
		if (length < 0) {
			return (int) length;
		}
		if ((size_t) length != sizeof(proof) || !valid_proof(&proof, path, candidate)) {
			return -EBADMSG;
		}
		next = candidate + 1U;
	}

	*sequence = next;
	return 0;
}

int pulse_capture_log_append(const struct pulse_capture_proof* proof) {
	ssize_t length;

	if (!capture_fs_ready || proof == NULL ||
		!persistent_path((enum pulse_event_path) proof->event_path) ||
		proof->sequence >= capture_count() ||
		!valid_proof(proof, (enum pulse_event_path) proof->event_path, proof->sequence)) {
		return -EINVAL;
	}

	length = nvs_write(&capture_fs,
				   proof_id((enum pulse_event_path) proof->event_path, proof->sequence),
				   proof,
				   sizeof(*proof));
	if (length < 0) {
		return (int) length;
	}
	return (size_t) length == sizeof(*proof) ? 0 : -EIO;
}

void pulse_capture_log_dump(const char* board_id) {
	struct pulse_capture_proof proof;
	const char* id = board_id != NULL ? board_id : "unknown";

	if (!capture_fs_ready) {
		printk("PULSE_CAPTURE_STATUS,board_id=%s,error=storage_not_ready\n", id);
		return;
	}

	for (uint8_t path_value = (uint8_t) pulse_event_path_Local;
		 path_value <= (uint8_t) pulse_event_path_AcceptedDiscovery;
		 ++path_value) {
		enum pulse_event_path path = (enum pulse_event_path) path_value;

		for (uint32_t sequence = 0U; sequence < capture_count(); ++sequence) {
			ssize_t length =
				nvs_read(&capture_fs, proof_id(path, sequence), &proof, sizeof(proof));

			if (length == -ENOENT) {
				continue;
			}
			if ((size_t) length != sizeof(proof) || !valid_proof(&proof, path, sequence)) {
				printk("PULSE_CAPTURE_STATUS,board_id=%s,error=invalid_record,path=%s,"
					   "sequence=%u,read_result=%d\n",
					   id,
					   pulse_metrics_path_name(path),
					   sequence,
					   (int) length);
				continue;
			}

			const char* role = proof.role <= (uint8_t) pulse_metric_role_Responder
							   ? pulse_metrics_role_name((enum pulse_metric_role) proof.role)
							   : "unresolved";
			printk("PULSE_CAPTURE_PROOF,board_id=%s,schema=%u,path=%s,firmware_sequence=%u,sequence=%u,"
				   "trial_id=%u,exchange_id=%u,role=%s,warmup=%u,record_ready=%u,"
				   "passive_expected=%u,success=%u,failure_reason=%u,"
				   "remote_session_started=%u,event_duration_us=%u,event_errno=%d,"
				   "infrastructure_errno=%d,result_release_errno=%d,"
				   "initial_head_hash=%08x,result_head_hash=%08x,"
				   "firmware_revision_hash=%08x,artifact_hash=%08x\n",
				   id,
				   proof.schema_version,
				   pulse_metrics_path_name(path),
				   proof.sequence,
				   proof.sequence,
				   proof.trial_id,
				   proof.exchange_id,
				   role,
				   pulse_capture_proof_flag_get(&proof, PULSE_CAPTURE_PROOF_FLAG_WARMUP),
				   pulse_capture_proof_flag_get(&proof,
										PULSE_CAPTURE_PROOF_FLAG_RECORD_READY),
				   pulse_capture_proof_flag_get(&proof,
										PULSE_CAPTURE_PROOF_FLAG_PASSIVE_EXPECTED),
				   pulse_capture_proof_flag_get(&proof, PULSE_CAPTURE_PROOF_FLAG_SUCCESS),
				   proof.failure_reason,
				   pulse_capture_proof_flag_get(
					   &proof, PULSE_CAPTURE_PROOF_FLAG_REMOTE_SESSION_STARTED),
				   (unsigned int) proof.event_duration_us,
				   proof.event_errno,
				   proof.infrastructure_errno,
				   proof.result_release_errno,
				   proof.initial_head_hash,
				   proof.result_head_hash,
				   proof.firmware_revision_hash,
				   proof.artifact_hash);
		}
	}
	printk("PULSE_CAPTURE_STATUS,board_id=%s,state=export_complete\n", id);
}
