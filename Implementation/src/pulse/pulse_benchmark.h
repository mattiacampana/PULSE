/*
 * Copyright (c) 2026
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef SENSWEAR_PULSE_BENCHMARK_H_
#define SENSWEAR_PULSE_BENCHMARK_H_

#include "pulse_fixture.h"
#include "pulse_metrics.h"
#include "pulse_peers.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PULSE_LEARNING_RATE 0.01f
#define PULSE_ALPHA_MAX 0.5f
#define PULSE_UTILITY_MOMENTUM 0.2f
#define PULSE_INITIAL_UTILITY 0.0f
#define PULSE_UCB_EXPLORATION 0.25f
#define PULSE_NORM_EPSILON 1.0e-12f
#define PULSE_HEAD_BYTES (PULSE_HEAD_PARAMETER_COUNT * sizeof(float))

struct pulse_node_state {
	pulse_head_t head;
	pulse_peer_table_t peers;
	uint32_t head_version;
};

struct pulse_compute_workspace {
	pulse_model_workspace_t model;
	pulse_embedding_batch_t embeddings;
};

int pulse_benchmark_node_init(struct pulse_node_state* node, uint32_t tie_break_seed);
int pulse_benchmark_reset_trial(struct pulse_node_state* node, uint32_t tie_break_seed);

int pulse_benchmark_compute_gradient(const pulse_head_t* head,
									 const pulse_batch_t* batch,
									 struct pulse_compute_workspace* workspace,
									 pulse_gradient_t* gradient,
									 float* loss,
									 struct pulse_metric_record* record,
									 enum pulse_stage encoder_stage,
									 enum pulse_stage head_stage);

int pulse_benchmark_run_scheduled_local(struct pulse_node_state* node,
										struct pulse_compute_workspace* workspace,
										pulse_gradient_t* gradient,
										struct pulse_metric_record* record,
										uint32_t trial_id);

/* Build deterministic one-opportunity peer histories before the event gate rises. */
int pulse_benchmark_prepare_accepted(struct pulse_node_state* node, uint64_t peer_id);
int pulse_benchmark_prepare_no_contact(struct pulse_node_state* node, uint64_t peer_id);

/* Execute selection and its one-step local fallback inside an already active event. */
int pulse_benchmark_execute_no_contact(struct pulse_node_state* node,
									   struct pulse_compute_workspace* workspace,
									   pulse_gradient_t* gradient,
									   struct pulse_metric_record* record,
									   uint64_t peer_id);

int pulse_benchmark_select_peer(struct pulse_node_state* node,
								uint64_t peer_id,
								struct pulse_metric_record* record,
								pulse_peer_selection_t* selection);

int pulse_benchmark_finish_initiator(struct pulse_node_state* node,
									 uint64_t peer_id,
									 const pulse_gradient_t* local_gradient,
									 const pulse_gradient_t* remote_gradient,
									 struct pulse_metric_record* record);

void pulse_benchmark_populate_diagnostics(const struct pulse_node_state* node,
										  const pulse_gradient_t* local_gradient,
										  const pulse_gradient_t* remote_gradient,
										  struct pulse_metric_record* record);

int pulse_benchmark_compute_responder(const pulse_head_t* request_head,
									  struct pulse_compute_workspace* workspace,
									  pulse_gradient_t* response_gradient,
									  struct pulse_metric_record* record,
									  float* loss);

#ifdef __cplusplus
}
#endif

#endif /* SENSWEAR_PULSE_BENCHMARK_H_ */
