/*
 * Copyright (c) 2026
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pulse_benchmark.h"

#include <errno.h>
#include <math.h>

#include <zephyr/sys/util.h>

BUILD_ASSERT(PULSE_HEAD_PARAMETER_COUNT == 4550U, "PULSE HAR head layout changed");
BUILD_ASSERT(sizeof(pulse_head_t) == PULSE_HEAD_BYTES, "head wire layout has padding");
BUILD_ASSERT(sizeof(pulse_gradient_t) == PULSE_HEAD_BYTES, "gradient wire layout has padding");

static int status_to_errno(pulse_status_t status) {
	switch (status) {
	case PULSE_STATUS_OK:
		return 0;
	case PULSE_STATUS_BAD_ARGUMENT:
	case PULSE_STATUS_BAD_ARTIFACT:
	case PULSE_STATUS_BAD_LABEL:
		return -EINVAL;
	case PULSE_STATUS_CAPACITY:
		return -ENOSPC;
	case PULSE_STATUS_NUMERIC_ERROR:
	default:
		return -ERANGE;
	}
}

int pulse_benchmark_node_init(struct pulse_node_state* node, uint32_t tie_break_seed) {
	return pulse_benchmark_reset_trial(node, tie_break_seed);
}

int pulse_benchmark_reset_trial(struct pulse_node_state* node, uint32_t tie_break_seed) {
	pulse_status_t status;

	if (node == NULL) {
		return -EINVAL;
	}

	status = pulse_head_load(&node->head, &pulse_fixture_initial_head);
	if (status != PULSE_STATUS_OK) {
		return status_to_errno(status);
	}
	node->head_version = 0U;
	status = pulse_peer_table_init(&node->peers,
								   PULSE_INITIAL_UTILITY,
								   PULSE_UCB_EXPLORATION,
								   tie_break_seed);
	return status_to_errno(status);
}

int pulse_benchmark_compute_gradient(const pulse_head_t* head,
									 const pulse_batch_t* batch,
									 struct pulse_compute_workspace* workspace,
									 pulse_gradient_t* gradient,
									 float* loss,
									 struct pulse_metric_record* record,
									 enum pulse_stage encoder_stage,
									 enum pulse_stage head_stage) {
	pulse_sample_batch_t samples;
	pulse_status_t status;

	if (head == NULL || batch == NULL || workspace == NULL || gradient == NULL || record == NULL) {
		return -EINVAL;
	}

	samples.samples = batch->samples;
	samples.sample_float_count = batch->sample_float_count;
	samples.sample_stride_floats = batch->sample_stride_floats;
	samples.batch_size = batch->batch_size;

	pulse_metrics_stage_begin(record, encoder_stage, pulse_marker_stage_Encoder);
	status = pulse_encoder_batch_forward(&pulse_fixture_encoder,
										 &samples,
										 &workspace->model,
										 &workspace->embeddings);
	pulse_metrics_stage_end(record, encoder_stage);
	if (status != PULSE_STATUS_OK) {
		return status_to_errno(status);
	}

	pulse_metrics_stage_begin(record, head_stage, pulse_marker_stage_HeadBackward);
	status = pulse_head_batch_gradient_from_embeddings(head,
													   &workspace->embeddings,
													   batch->labels,
													   batch->label_count,
													   &workspace->model,
													   gradient,
													   loss);
	pulse_metrics_stage_end(record, head_stage);
	return status_to_errno(status);
}

static int run_local_step(struct pulse_node_state* node,
						  const pulse_batch_t* batch,
						  struct pulse_compute_workspace* workspace,
						  pulse_gradient_t* gradient,
						  struct pulse_metric_record* record,
						  enum pulse_stage encoder_stage,
						  enum pulse_stage head_stage,
						  enum pulse_stage update_stage) {
	int ret = pulse_benchmark_compute_gradient(&node->head,
											   batch,
											   workspace,
											   gradient,
											   NULL,
											   record,
											   encoder_stage,
											   head_stage);

	if (ret != 0) {
		return ret;
	}

	pulse_metrics_stage_begin(record, update_stage, pulse_marker_stage_Mixing);
	pulse_status_t status =
		pulse_head_apply_local_gradient(&node->head, gradient, PULSE_LEARNING_RATE);
	pulse_metrics_stage_end(record, update_stage);
	if (status != PULSE_STATUS_OK) {
		return status_to_errno(status);
	}
	node->head_version++;
	return 0;
}

int pulse_benchmark_run_scheduled_local(struct pulse_node_state* node,
										struct pulse_compute_workspace* workspace,
										pulse_gradient_t* gradient,
										struct pulse_metric_record* record,
										uint32_t trial_id) {
	int ret;

	if (node == NULL || workspace == NULL || gradient == NULL || record == NULL) {
		return -EINVAL;
	}

	uint32_t initial_hash = pulse_metrics_hash32(&node->head, sizeof(node->head));
	pulse_metrics_event_begin(record,
							  pulse_event_path_Local,
							  pulse_metric_role_Local,
							  pulse_connection_start_NotApplicable,
							  trial_id,
							  0U);
	record->head_version_before = node->head_version;
	record->initial_head_hash = initial_hash;

	ret = run_local_step(node,
						 &pulse_fixture_batches[0],
						 workspace,
						 gradient,
						 record,
						 pulse_stage_LocalStep1Encoder,
						 pulse_stage_LocalStep1HeadBackward,
						 pulse_stage_LocalStep1Update);
	if (ret == 0) {
		ret = run_local_step(node,
							 &pulse_fixture_batches[1],
							 workspace,
							 gradient,
							 record,
							 pulse_stage_LocalStep2Encoder,
							 pulse_stage_LocalStep2HeadBackward,
							 pulse_stage_LocalStep2Update);
	}

	record->head_version_after = node->head_version;
	pulse_metrics_event_end(record,
							ret == 0,
							ret == 0 ? pulse_failure_None : pulse_failure_Internal);
	record->result_head_hash = pulse_metrics_hash32(&node->head, sizeof(node->head));
	return ret;
}

int pulse_benchmark_select_peer(struct pulse_node_state* node,
								uint64_t peer_id,
								struct pulse_metric_record* record,
								pulse_peer_selection_t* selection) {
	const pulse_peer_entry_t* before;
	pulse_status_t status;

	if (node == NULL || record == NULL || selection == NULL) {
		return -EINVAL;
	}

	record->reachable_peers = 1U;
	record->selection_opportunities_before = node->peers.total_selection_opportunities;
	before = pulse_peer_find(&node->peers, peer_id);
	record->utility_before = before != NULL ? before->utility : node->peers.initial_utility;
	record->peer_visits_before = before != NULL ? before->visits : 0U;

	pulse_metrics_stage_begin(record, pulse_stage_PeerSelection, pulse_marker_stage_Selection);
	status = pulse_peer_select(&node->peers, &peer_id, 1U, selection);
	pulse_metrics_stage_end(record, pulse_stage_PeerSelection);
	record->selection_opportunities_after = node->peers.total_selection_opportunities;
	record->ucb_index = selection->ucb_index;
	return status_to_errno(status);
}

static int prepare_peer_history(struct pulse_node_state* node, uint64_t peer_id, float agreement) {
	pulse_peer_selection_t selection;
	pulse_status_t status;

	if (node == NULL) {
		return -EINVAL;
	}

	/* Recreate public policy transitions: s=1, followed by n=1 and U=+/-0.2. */
	status = pulse_peer_select(&node->peers, &peer_id, 1U, &selection);
	if (status != PULSE_STATUS_OK) {
		return status_to_errno(status);
	}
	if (selection.action != PULSE_PEER_SELECTION_CONTACT || selection.peer_id != peer_id ||
		node->peers.total_selection_opportunities != 1U) {
		return -EPROTO;
	}

	status = pulse_peer_observe(&node->peers, peer_id, agreement, PULSE_UTILITY_MOMENTUM);
	return status_to_errno(status);
}

int pulse_benchmark_prepare_accepted(struct pulse_node_state* node, uint64_t peer_id) {
	return prepare_peer_history(node, peer_id, 1.0f);
}

int pulse_benchmark_prepare_no_contact(struct pulse_node_state* node, uint64_t peer_id) {
	return prepare_peer_history(node, peer_id, -1.0f);
}

int pulse_benchmark_execute_no_contact(struct pulse_node_state* node,
									   struct pulse_compute_workspace* workspace,
									   pulse_gradient_t* gradient,
									   struct pulse_metric_record* record,
									   uint64_t peer_id) {
	pulse_peer_selection_t selection;
	int ret;

	if (node == NULL || workspace == NULL || gradient == NULL || record == NULL) {
		return -EINVAL;
	}

	ret = pulse_benchmark_select_peer(node, peer_id, record, &selection);
	if (ret == 0 && selection.action != PULSE_PEER_SELECTION_NO_CONTACT) {
		ret = -EPROTO;
	}
	if (ret == 0) {
		ret = run_local_step(node,
							 &pulse_fixture_batches[2],
							 workspace,
							 gradient,
							 record,
							 pulse_stage_LocalStep1Encoder,
							 pulse_stage_LocalStep1HeadBackward,
							 pulse_stage_LocalStep1Update);
	}

	const pulse_peer_entry_t* after = pulse_peer_find(&node->peers, peer_id);
	record->utility_after = after != NULL ? after->utility : node->peers.initial_utility;
	record->peer_visits_after = after != NULL ? after->visits : 0U;
	record->head_version_after = node->head_version;
	return ret;
}

int pulse_benchmark_finish_initiator(struct pulse_node_state* node,
									 uint64_t peer_id,
									 const pulse_gradient_t* local_gradient,
									 const pulse_gradient_t* remote_gradient,
									 struct pulse_metric_record* record) {
	pulse_gradient_products_t products;
	const pulse_peer_entry_t* after;
	float agreement;
	float alpha;
	float remote_scale;
	pulse_status_t status;

	if (node == NULL || local_gradient == NULL || remote_gradient == NULL || record == NULL) {
		return -EINVAL;
	}
	if (pulse_peer_find(&node->peers, peer_id) == NULL &&
		pulse_peer_count(&node->peers) >= PULSE_MAX_PEERS) {
		return -ENOSPC;
	}

	pulse_metrics_stage_begin(record, pulse_stage_Agreement, pulse_marker_stage_Mixing);
	status = pulse_gradient_products(local_gradient, remote_gradient, &products);
	if (status == PULSE_STATUS_OK) {
		status = pulse_agreement_from_products(&products, PULSE_NORM_EPSILON, &agreement);
	}
	if (status == PULSE_STATUS_OK) {
		status =
			pulse_compute_mixing_alpha(PULSE_ALPHA_MAX, record->utility_before, agreement, &alpha);
	}
	pulse_metrics_stage_end(record, pulse_stage_Agreement);
	if (status != PULSE_STATUS_OK) {
		return status_to_errno(status);
	}

	pulse_metrics_stage_begin(record, pulse_stage_Normalization, pulse_marker_stage_Mixing);
	status = pulse_remote_scale_from_products(&products, PULSE_NORM_EPSILON, &remote_scale);
	pulse_metrics_stage_end(record, pulse_stage_Normalization);
	if (status != PULSE_STATUS_OK) {
		return status_to_errno(status);
	}

	pulse_metrics_stage_begin(record, pulse_stage_Mixing, pulse_marker_stage_Mixing);
	status = pulse_head_apply_weighted_gradient(&node->head,
												local_gradient,
												remote_gradient,
												PULSE_LEARNING_RATE,
												alpha,
												remote_scale);
	pulse_metrics_stage_end(record, pulse_stage_Mixing);
	if (status != PULSE_STATUS_OK) {
		return status_to_errno(status);
	}
	node->head_version++;

	pulse_metrics_stage_begin(record, pulse_stage_UtilityUpdate, pulse_marker_stage_Mixing);
	status = pulse_peer_observe(&node->peers, peer_id, agreement, PULSE_UTILITY_MOMENTUM);
	pulse_metrics_stage_end(record, pulse_stage_UtilityUpdate);
	if (status != PULSE_STATUS_OK) {
		return status_to_errno(status);
	}

	after = pulse_peer_find(&node->peers, peer_id);
	record->utility_after = after != NULL ? after->utility : record->utility_before;
	record->peer_visits_after = after != NULL ? after->visits : record->peer_visits_before;
	record->agreement = agreement;
	record->mixing_alpha = alpha;
	record->normalization_scale = remote_scale;
	record->head_version_after = node->head_version;
	return 0;
}

void pulse_benchmark_populate_diagnostics(const struct pulse_node_state* node,
										  const pulse_gradient_t* local_gradient,
										  const pulse_gradient_t* remote_gradient,
										  struct pulse_metric_record* record) {
	pulse_gradient_products_t products;

	if (node == NULL || local_gradient == NULL || remote_gradient == NULL || record == NULL) {
		return;
	}
	if (pulse_gradient_products(local_gradient, remote_gradient, &products) != PULSE_STATUS_OK) {
		return;
	}
	record->local_gradient_norm = (float) sqrt(products.local_norm_sq);
	record->remote_gradient_norm = (float) sqrt(products.remote_norm_sq);
	record->gradient_dot = (float) products.dot_product;
	record->result_head_hash = pulse_metrics_hash32(&node->head, sizeof(node->head));
}

int pulse_benchmark_compute_responder(const pulse_head_t* request_head,
									  struct pulse_compute_workspace* workspace,
									  pulse_gradient_t* response_gradient,
									  struct pulse_metric_record* record,
									  float* loss) {
	return pulse_benchmark_compute_gradient(request_head,
											&pulse_fixture_batches[3],
											workspace,
											response_gradient,
											loss,
											record,
											pulse_stage_ResponderEncoder,
											pulse_stage_ResponderHeadBackward);
}
