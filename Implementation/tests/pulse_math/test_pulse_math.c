#include "pulse_math.h"
#include "pulse_peers.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define TEST_TOLERANCE 1.0e-5f

static float conv1_weight[PULSE_ENCODER_CONV1_WEIGHT_COUNT];
static float conv1_bias[PULSE_ENCODER_CONV1_BIAS_COUNT];
static float conv2_weight[PULSE_ENCODER_CONV2_WEIGHT_COUNT];
static float conv2_bias[PULSE_ENCODER_CONV2_BIAS_COUNT];
static float projection_weight[PULSE_ENCODER_PROJECTION_WEIGHT_COUNT];
static float projection_bias[PULSE_ENCODER_PROJECTION_BIAS_COUNT];
static float samples[PULSE_HAR_BATCH_SIZE * PULSE_HAR_SAMPLE_FLOATS];
static uint8_t labels[PULSE_HAR_BATCH_SIZE];
static pulse_head_t head;
static pulse_head_t trained_head;
static pulse_gradient_t gradient;
static pulse_gradient_t split_gradient;
static pulse_gradient_t local_gradient;
static pulse_gradient_t remote_gradient;
static pulse_model_workspace_t workspace;
static pulse_embedding_batch_t embedding_batch;

static int failures;

#define CHECK(condition)                                                         \
	do {                                                                         \
		if (!(condition)) {                                                      \
			fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
			++failures;                                                          \
		}                                                                        \
	} while (0)

static void check_close(float actual, float expected, float tolerance) {
	if (!isfinite(actual) || fabsf(actual - expected) > tolerance) {
		fprintf(stderr,
				"FAIL %s:%d: actual %.9g, expected %.9g (tol %.9g)\n",
				__FILE__,
				__LINE__,
				(double) actual,
				(double) expected,
				(double) tolerance);
		++failures;
	}
}

static pulse_encoder_artifact_t make_encoder(void) {
	pulse_encoder_artifact_t encoder = {
		.conv1_weight = {conv1_weight, PULSE_ENCODER_CONV1_WEIGHT_COUNT},
		.conv1_bias = {conv1_bias, PULSE_ENCODER_CONV1_BIAS_COUNT},
		.conv2_weight = {conv2_weight, PULSE_ENCODER_CONV2_WEIGHT_COUNT},
		.conv2_bias = {conv2_bias, PULSE_ENCODER_CONV2_BIAS_COUNT},
		.projection_weight =
			{
				projection_weight,
				PULSE_ENCODER_PROJECTION_WEIGHT_COUNT,
			},
		.projection_bias = {projection_bias, PULSE_ENCODER_PROJECTION_BIAS_COUNT},
	};

	return encoder;
}

static pulse_batch_t make_batch(void) {
	pulse_batch_t batch = {
		.samples = samples,
		.sample_float_count = PULSE_HAR_BATCH_SIZE * PULSE_HAR_SAMPLE_FLOATS,
		.sample_stride_floats = PULSE_HAR_SAMPLE_FLOATS,
		.labels = labels,
		.label_count = PULSE_HAR_BATCH_SIZE,
		.batch_size = PULSE_HAR_BATCH_SIZE,
	};

	return batch;
}

static void set_up_identity_path(void) {
	size_t sample_index;
	size_t time_index;

	memset(conv1_weight, 0, sizeof(conv1_weight));
	memset(conv1_bias, 0, sizeof(conv1_bias));
	memset(conv2_weight, 0, sizeof(conv2_weight));
	memset(conv2_bias, 0, sizeof(conv2_bias));
	memset(projection_weight, 0, sizeof(projection_weight));
	memset(projection_bias, 0, sizeof(projection_bias));
	memset(samples, 0, sizeof(samples));
	memset(labels, 0, sizeof(labels));
	memset(&head, 0, sizeof(head));

	/* Conv1[0,0,center], Conv2[0,0,center], projection[0,0] form an identity path. */
	conv1_weight[2U] = 1.0f;
	conv2_weight[2U] = 1.0f;
	projection_weight[0U] = 1.0f;

	for (sample_index = 0U; sample_index < PULSE_HAR_BATCH_SIZE; ++sample_index) {
		for (time_index = 0U; time_index < PULSE_HAR_WINDOW_LENGTH; ++time_index) {
			samples[sample_index * PULSE_HAR_SAMPLE_FLOATS + time_index] = 2.0f;
		}
	}

	head.parameters[PULSE_HEAD_W1_OFFSET] = 1.0f;
	head.parameters[PULSE_HEAD_W2_OFFSET] = 1.0f;
}

static void test_layout_and_artifact_loading(void) {
	float artifact_values[PULSE_HEAD_PARAMETER_COUNT];
	pulse_head_artifact_t artifact = {
		.parameters = {artifact_values, PULSE_HEAD_PARAMETER_COUNT},
	};
	pulse_head_t loaded;
	size_t index;

	CHECK(PULSE_ENCODER_PARAMETER_COUNT == 14976U);
	CHECK(PULSE_HEAD_W1_OFFSET == 0U);
	CHECK(PULSE_HEAD_B1_OFFSET == 4096U);
	CHECK(PULSE_HEAD_W2_OFFSET == 4160U);
	CHECK(PULSE_HEAD_B2_OFFSET == 4544U);
	CHECK(PULSE_HEAD_PARAMETER_COUNT == 4550U);
	CHECK(pulse_head_storage_size() == PULSE_HEAD_PARAMETER_COUNT * sizeof(float));
	CHECK(pulse_gradient_storage_size() == PULSE_HEAD_PARAMETER_COUNT * sizeof(float));
	CHECK(pulse_model_workspace_size() == sizeof(pulse_model_workspace_t));
	CHECK(pulse_embedding_batch_storage_size() ==
		  PULSE_HAR_BATCH_SIZE * PULSE_EMBEDDING_SIZE * sizeof(float));

	for (index = 0U; index < PULSE_HEAD_PARAMETER_COUNT; ++index) {
		artifact_values[index] = (float) index * 0.001f;
	}

	CHECK(pulse_head_load(&loaded, &artifact) == PULSE_STATUS_OK);
	check_close(loaded.parameters[0], 0.0f, 0.0f);
	check_close(loaded.parameters[PULSE_HEAD_B2_OFFSET], 4.544f, TEST_TOLERANCE);
	check_close(loaded.parameters[PULSE_HEAD_PARAMETER_COUNT - 1U], 4.549f, TEST_TOLERANCE);

	artifact.parameters.count = PULSE_HEAD_PARAMETER_COUNT - 1U;
	CHECK(pulse_head_load(&loaded, &artifact) == PULSE_STATUS_BAD_ARTIFACT);
}

static void test_encoder_and_batch_gradient(void) {
	pulse_encoder_artifact_t encoder = make_encoder();
	pulse_batch_t batch = make_batch();
	pulse_sample_batch_t sample_batch = {
		.samples = samples,
		.sample_float_count = PULSE_HAR_BATCH_SIZE * PULSE_HAR_SAMPLE_FLOATS,
		.sample_stride_floats = PULSE_HAR_SAMPLE_FLOATS,
		.batch_size = PULSE_HAR_BATCH_SIZE,
	};
	float embedding[PULSE_EMBEDDING_SIZE];
	float loss;
	float split_loss;
	float denominator = expf(2.0f) + 5.0f;
	float delta_zero = expf(2.0f) / denominator - 1.0f;
	float delta_other = 1.0f / denominator;
	size_t index;

	set_up_identity_path();
	CHECK(pulse_encoder_artifact_validate(&encoder) == PULSE_STATUS_OK);
	CHECK(pulse_batch_validate(&batch) == PULSE_STATUS_OK);
	CHECK(
		pulse_encoder_forward(&encoder, samples, PULSE_HAR_SAMPLE_FLOATS, &workspace, embedding) ==
		PULSE_STATUS_OK);
	check_close(embedding[0], 2.0f, TEST_TOLERANCE);
	for (index = 1U; index < PULSE_EMBEDDING_SIZE; ++index) {
		check_close(embedding[index], 0.0f, 0.0f);
	}

	CHECK(pulse_head_batch_gradient(&encoder, &head, &batch, &workspace, &gradient, &loss) ==
		  PULSE_STATUS_OK);
	check_close(loss, logf(denominator) - 2.0f, TEST_TOLERANCE);
	check_close(gradient.parameters[PULSE_HEAD_W1_OFFSET], 2.0f * delta_zero, TEST_TOLERANCE);
	check_close(gradient.parameters[PULSE_HEAD_B1_OFFSET], delta_zero, TEST_TOLERANCE);
	check_close(gradient.parameters[PULSE_HEAD_W2_OFFSET], 2.0f * delta_zero, TEST_TOLERANCE);
	check_close(gradient.parameters[PULSE_HEAD_B2_OFFSET], delta_zero, TEST_TOLERANCE);
	check_close(gradient.parameters[PULSE_HEAD_W2_OFFSET + PULSE_HEAD_HIDDEN_SIZE],
				2.0f * delta_other,
				TEST_TOLERANCE);
	check_close(gradient.parameters[PULSE_HEAD_B2_OFFSET + 1U], delta_other, TEST_TOLERANCE);

	CHECK(pulse_encoder_batch_forward(&encoder, &sample_batch, &workspace, &embedding_batch) ==
		  PULSE_STATUS_OK);
	for (index = 0U; index < PULSE_HAR_BATCH_SIZE; ++index) {
		check_close(embedding_batch.values[index * PULSE_EMBEDDING_SIZE], 2.0f, TEST_TOLERANCE);
	}
	CHECK(pulse_head_batch_gradient_from_embeddings(&head,
													&embedding_batch,
													labels,
													PULSE_HAR_BATCH_SIZE,
													&workspace,
													&split_gradient,
													&split_loss) == PULSE_STATUS_OK);
	check_close(split_loss, loss, TEST_TOLERANCE);
	for (index = 0U; index < PULSE_HEAD_PARAMETER_COUNT; ++index) {
		check_close(split_gradient.parameters[index], gradient.parameters[index], 0.0f);
	}

	labels[0] = PULSE_HAR_CLASS_COUNT;
	CHECK(pulse_head_batch_gradient(&encoder, &head, &batch, &workspace, &gradient, &loss) ==
		  PULSE_STATUS_BAD_LABEL);
	labels[0] = 0U;
}

static void test_r_step_local_sgd(void) {
	pulse_encoder_artifact_t encoder = make_encoder();
	pulse_batch_t batches[2] = {make_batch(), make_batch()};
	pulse_local_sgd_stats_t stats;
	float before;
	float after;

	set_up_identity_path();
	trained_head = head;
	CHECK(pulse_head_batch_gradient(&encoder,
									&trained_head,
									&batches[0],
									&workspace,
									&gradient,
									&before) == PULSE_STATUS_OK);
	CHECK(pulse_head_local_sgd(&encoder,
							   &trained_head,
							   batches,
							   2U,
							   2U,
							   0.01f,
							   &workspace,
							   &gradient,
							   &stats) == PULSE_STATUS_OK);
	CHECK(stats.steps_completed == 2U);
	CHECK(pulse_head_batch_gradient(&encoder,
									&trained_head,
									&batches[0],
									&workspace,
									&gradient,
									&after) == PULSE_STATUS_OK);
	CHECK(after < before);
}

static void test_agreement_and_mixing(void) {
	pulse_gradient_products_t products;
	pulse_gradient_products_t floor_products = {
		.local_norm_sq = 4.0e-12,
		.remote_norm_sq = 1.0e-14,
		.dot_product = 0.0,
	};
	pulse_mix_stats_t stats;
	float agreement;
	float alpha;
	float floor_scale;
	float remote_scale;

	memset(&head, 0, sizeof(head));
	pulse_gradient_zero(&local_gradient);
	pulse_gradient_zero(&remote_gradient);
	local_gradient.parameters[0] = 3.0f;
	local_gradient.parameters[1] = 4.0f;
	remote_gradient.parameters[1] = 10.0f;

	CHECK(pulse_cosine_agreement(&local_gradient, &remote_gradient, 1.0e-12f, &agreement) ==
		  PULSE_STATUS_OK);
	check_close(agreement, 0.8f, TEST_TOLERANCE);
	CHECK(pulse_gradient_products(&local_gradient, &remote_gradient, &products) == PULSE_STATUS_OK);
	check_close((float) products.local_norm_sq, 25.0f, TEST_TOLERANCE);
	check_close((float) products.remote_norm_sq, 100.0f, TEST_TOLERANCE);
	check_close((float) products.dot_product, 40.0f, TEST_TOLERANCE);
	CHECK(pulse_agreement_from_products(&products, 1.0e-12f, &agreement) == PULSE_STATUS_OK);
	check_close(agreement, 0.8f, TEST_TOLERANCE);
	CHECK(pulse_remote_scale_from_products(&products, 1.0e-12f, &remote_scale) == PULSE_STATUS_OK);
	check_close(remote_scale, 0.5f, TEST_TOLERANCE);
	CHECK(pulse_remote_scale_from_products(&floor_products, 1.0e-12f, &floor_scale) ==
		  PULSE_STATUS_OK);
	check_close(floor_scale, 2.0f, TEST_TOLERANCE);
	CHECK(pulse_compute_mixing_alpha(0.5f, 0.5f, agreement, &alpha) == PULSE_STATUS_OK);
	check_close(alpha, 0.2f, TEST_TOLERANCE);
	memset(&trained_head, 0, sizeof(trained_head));
	CHECK(pulse_head_apply_weighted_gradient(&trained_head,
											 &local_gradient,
											 &remote_gradient,
											 0.1f,
											 alpha,
											 remote_scale) == PULSE_STATUS_OK);
	check_close(trained_head.parameters[0], -0.24f, TEST_TOLERANCE);
	check_close(trained_head.parameters[1], -0.42f, TEST_TOLERANCE);

	CHECK(pulse_head_apply_peer_update(&head,
									   &local_gradient,
									   &remote_gradient,
									   0.1f,
									   0.5f,
									   0.5f,
									   1.0e-12f,
									   &stats) == PULSE_STATUS_OK);
	check_close(stats.agreement, 0.8f, TEST_TOLERANCE);
	check_close(stats.alpha, 0.2f, TEST_TOLERANCE);
	check_close(stats.remote_scale, 0.5f, TEST_TOLERANCE);
	check_close(head.parameters[0], -0.24f, TEST_TOLERANCE);
	check_close(head.parameters[1], -0.42f, TEST_TOLERANCE);

	pulse_gradient_zero(&remote_gradient);
	CHECK(pulse_cosine_agreement(&local_gradient, &remote_gradient, 1.0e-12f, &agreement) ==
		  PULSE_STATUS_OK);
	check_close(agreement, 0.0f, 0.0f);

	remote_gradient.parameters[0] = -3.0f;
	remote_gradient.parameters[1] = -4.0f;
	CHECK(pulse_cosine_agreement(&local_gradient, &remote_gradient, 1.0e-12f, &agreement) ==
		  PULSE_STATUS_OK);
	check_close(agreement, -1.0f, TEST_TOLERANCE);
	CHECK(pulse_compute_mixing_alpha(0.5f, 0.5f, agreement, &alpha) == PULSE_STATUS_OK);
	check_close(alpha, 0.0f, 0.0f);
}

static void test_peer_policy(void) {
	const uint64_t candidates[] = {11U, 22U, 33U};
	pulse_peer_table_t first;
	pulse_peer_table_t second;
	pulse_peer_table_t conservative;
	pulse_peer_table_t seeded;
	pulse_peer_selection_t first_selection;
	pulse_peer_selection_t second_selection;
	const pulse_peer_entry_t* entry;
	size_t index;

	CHECK(pulse_peer_table_init(&first, 0.0f, 0.25f, 1234U) == PULSE_STATUS_OK);
	CHECK(pulse_peer_table_init(&second, 0.0f, 0.25f, 1234U) == PULSE_STATUS_OK);
	CHECK(pulse_peer_select(&first, candidates, 3U, &first_selection) == PULSE_STATUS_OK);
	CHECK(pulse_peer_select(&second, candidates, 3U, &second_selection) == PULSE_STATUS_OK);
	CHECK(first_selection.action == PULSE_PEER_SELECTION_CONTACT);
	CHECK(first_selection.peer_id == second_selection.peer_id);
	CHECK(first_selection.candidate_index == second_selection.candidate_index);
	CHECK(first.total_selection_opportunities == 1U);
	CHECK(second.total_selection_opportunities == 1U);

	/* Benchmark preconditions must be reachable through public policy transitions. */
	CHECK(pulse_peer_table_init(&seeded, 0.0f, 0.25f, 42U) == PULSE_STATUS_OK);
	CHECK(pulse_peer_select(&seeded, candidates, 1U, &first_selection) == PULSE_STATUS_OK);
	CHECK(first_selection.action == PULSE_PEER_SELECTION_CONTACT);
	CHECK(pulse_peer_observe(&seeded, candidates[0], 1.0f, 0.2f) == PULSE_STATUS_OK);
	entry = pulse_peer_find(&seeded, candidates[0]);
	CHECK(entry != NULL);
	check_close(entry->utility, 0.2f, TEST_TOLERANCE);
	CHECK(entry->visits == 1U);
	CHECK(seeded.total_selection_opportunities == 1U);

	CHECK(pulse_peer_table_init(&seeded, 0.0f, 0.25f, 43U) == PULSE_STATUS_OK);
	CHECK(pulse_peer_select(&seeded, candidates, 1U, &first_selection) == PULSE_STATUS_OK);
	CHECK(first_selection.action == PULSE_PEER_SELECTION_CONTACT);
	CHECK(pulse_peer_observe(&seeded, candidates[0], -1.0f, 0.2f) == PULSE_STATUS_OK);
	entry = pulse_peer_find(&seeded, candidates[0]);
	CHECK(entry != NULL);
	check_close(entry->utility, -0.2f, TEST_TOLERANCE);
	CHECK(entry->visits == 1U);
	CHECK(seeded.total_selection_opportunities == 1U);
	CHECK(pulse_peer_select(&seeded, candidates, 1U, &first_selection) == PULSE_STATUS_OK);
	CHECK(first_selection.action == PULSE_PEER_SELECTION_NO_CONTACT);

	CHECK(pulse_peer_table_init(&conservative, 0.0f, 0.0f, 7U) == PULSE_STATUS_OK);
	CHECK(pulse_peer_select(&conservative, candidates, 1U, &first_selection) == PULSE_STATUS_OK);
	CHECK(first_selection.action == PULSE_PEER_SELECTION_NO_CONTACT);
	CHECK(conservative.total_selection_opportunities == 1U);

	CHECK(pulse_peer_observe(&conservative, 42U, -1.0f, 0.2f) == PULSE_STATUS_OK);
	entry = pulse_peer_find(&conservative, 42U);
	CHECK(entry != NULL);
	check_close(entry->utility, -0.2f, TEST_TOLERANCE);
	CHECK(entry->visits == 1U);
	CHECK(pulse_peer_select(&conservative, &entry->peer_id, 1U, &first_selection) ==
		  PULSE_STATUS_OK);
	CHECK(first_selection.action == PULSE_PEER_SELECTION_NO_CONTACT);
	CHECK(pulse_peer_observe(&conservative, 42U, 1.0f, 0.2f) == PULSE_STATUS_OK);
	entry = pulse_peer_find(&conservative, 42U);
	check_close(entry->utility, 0.04f, TEST_TOLERANCE);
	CHECK(entry->visits == 2U);

	CHECK(pulse_peer_select(&conservative, NULL, 0U, &first_selection) == PULSE_STATUS_OK);
	CHECK(first_selection.action == PULSE_PEER_SELECTION_NO_CANDIDATES);
	CHECK(conservative.total_selection_opportunities == 2U);

	CHECK(pulse_peer_table_init(&conservative, 0.0f, 0.25f, 9U) == PULSE_STATUS_OK);
	for (index = 0U; index < PULSE_MAX_PEERS; ++index) {
		CHECK(pulse_peer_observe(&conservative, (uint64_t) index, 0.0f, 0.2f) == PULSE_STATUS_OK);
	}
	CHECK(pulse_peer_count(&conservative) == PULSE_MAX_PEERS);
	CHECK(pulse_peer_observe(&conservative, 1000U, 0.0f, 0.2f) == PULSE_STATUS_CAPACITY);
}

int main(void) {
	test_layout_and_artifact_loading();
	test_encoder_and_batch_gradient();
	test_r_step_local_sgd();
	test_agreement_and_mixing();
	test_peer_policy();

	if (failures != 0) {
		fprintf(stderr, "%d PULSE math test(s) failed\n", failures);
		return 1;
	}

	printf("PULSE math tests passed\n");
	return 0;
}
