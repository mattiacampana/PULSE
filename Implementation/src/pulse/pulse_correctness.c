/*
 * Copyright (c) 2026
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pulse_correctness.h"

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

#include <zephyr/sys/printk.h>

#include "pulse_benchmark.h"
#include "pulse_protocol.h"

#define PULSE_CORRECTNESS_MAX_ERROR_LIMIT 1.0e-5f
#define PULSE_CORRECTNESS_MIN_COSINE 0.99999f

static float maximum_error(const float* actual, const float* expected, size_t count) {
	float maximum = 0.0f;

	for (size_t i = 0U; i < count; ++i) {
		float error = fabsf(actual[i] - expected[i]);

		if (error > maximum) {
			maximum = error;
		}
	}
	return maximum;
}

static float gradient_cosine(const pulse_gradient_t* actual, const pulse_gradient_t* expected) {
	pulse_gradient_products_t products;
	float cosine = NAN;

	if (pulse_gradient_products(actual, expected, &products) != PULSE_STATUS_OK ||
		pulse_agreement_from_products(&products, PULSE_NORM_EPSILON, &cosine) != PULSE_STATUS_OK) {
		return NAN;
	}
	return cosine;
}

int pulse_correctness_run(const char* board_id) {
	static struct pulse_compute_workspace workspace;
	static pulse_gradient_t local_gradient;
	static pulse_gradient_t remote_gradient;
	static pulse_model_workspace_t single_workspace;
	static pulse_head_t head;
	static float embedding[PULSE_EMBEDDING_SIZE];
	static float logits[PULSE_HAR_CLASS_COUNT];
	float local_loss = 0.0f;
	float remote_loss = 0.0f;
	pulse_mix_stats_t mix_stats;
	float embedding_error;
	float logits_error;
	float local_gradient_error;
	float remote_gradient_error;
	float local_gradient_cosine;
	float remote_gradient_cosine;
	float local_loss_error;
	float remote_loss_error;
	float head_error;
	float agreement_error;
	float alpha_error;
	float remote_scale_error;
	uint32_t codec_crc32 = 0U;
	uint32_t gradient_codec_crc32 = 0U;
	int codec_serialize_status;
	int codec_deserialize_status;
	int gradient_codec_serialize_status;
	int gradient_codec_deserialize_status;
	bool codec_pass;
	bool gradient_codec_pass;
	bool passed;

	if (pulse_head_load(&head, &pulse_fixture_initial_head) != PULSE_STATUS_OK) {
		printk("PULSE_CORRECTNESS,board_id=%s,pass=0,error=initial_head_load_failed,"
			   "codec_pass=0,codec_crc32=%08x,codec_expected_crc32=%08x\n",
			   board_id,
			   codec_crc32,
			   pulse_fixture_initial_head_fp32le_crc32);
		return -EIO;
	}

	/*
	 * Exercise the exact wire codec in-place before numerical validation.  The
	 * generated ROM fixture is the bit-exact reference, so this needs no second
	 * mutable tensor-sized buffer in the correctness image.
	 */
	codec_serialize_status =
		pulse_protocol_head_serialize(&head, (uint8_t*) &head, sizeof(head), &codec_crc32);
	codec_deserialize_status =
		codec_serialize_status == 0
			? pulse_protocol_head_deserialize((const uint8_t*) &head, sizeof(head), &head)
			: codec_serialize_status;
	codec_pass = codec_serialize_status == 0 && codec_deserialize_status == 0 &&
				 codec_crc32 == pulse_fixture_initial_head_fp32le_crc32 &&
				 memcmp(head.parameters,
						pulse_fixture_initial_head.parameters.data,
						sizeof(head.parameters)) == 0;
	if (!codec_pass) {
		printk("PULSE_CORRECTNESS,board_id=%s,pass=0,error=fp32le_codec_failed,"
			   "codec_pass=0,codec_crc32=%08x,codec_expected_crc32=%08x\n",
			   board_id,
			   codec_crc32,
			   pulse_fixture_initial_head_fp32le_crc32);
		return -EIO;
	}

	if (pulse_encoder_forward(&pulse_fixture_encoder,
							  pulse_fixture_batches[0].samples,
							  PULSE_HAR_SAMPLE_FLOATS,
							  &single_workspace,
							  embedding) != PULSE_STATUS_OK ||
		pulse_head_forward(&head, embedding, &single_workspace, logits) != PULSE_STATUS_OK ||
		pulse_head_batch_gradient(&pulse_fixture_encoder,
								  &head,
								  &pulse_fixture_batches[2],
								  &workspace.model,
								  &local_gradient,
								  &local_loss) != PULSE_STATUS_OK ||
		pulse_head_batch_gradient(&pulse_fixture_encoder,
								  &head,
								  &pulse_fixture_batches[3],
								  &workspace.model,
								  &remote_gradient,
								  &remote_loss) != PULSE_STATUS_OK ||
		pulse_head_apply_peer_update(&head,
									 &local_gradient,
									 &remote_gradient,
									 PULSE_LEARNING_RATE,
									 PULSE_ALPHA_MAX,
									 pulse_fixture_golden_utility_before,
									 PULSE_NORM_EPSILON,
									 &mix_stats) != PULSE_STATUS_OK) {
		printk("PULSE_CORRECTNESS,board_id=%s,pass=0,error=embedded_computation_failed,"
			   "codec_pass=1,codec_crc32=%08x,codec_expected_crc32=%08x\n",
			   board_id,
			   codec_crc32,
			   pulse_fixture_initial_head_fp32le_crc32);
		return -EIO;
	}

	embedding_error =
		maximum_error(embedding, pulse_fixture_golden_embedding, PULSE_EMBEDDING_SIZE);
	logits_error = maximum_error(logits, pulse_fixture_golden_logits, PULSE_HAR_CLASS_COUNT);
	local_gradient_error = maximum_error(local_gradient.parameters,
										 pulse_fixture_golden_local_gradient.parameters,
										 PULSE_HEAD_PARAMETER_COUNT);
	remote_gradient_error = maximum_error(remote_gradient.parameters,
										  pulse_fixture_golden_remote_gradient.parameters,
										  PULSE_HEAD_PARAMETER_COUNT);
	local_gradient_cosine = gradient_cosine(&local_gradient, &pulse_fixture_golden_local_gradient);
	remote_gradient_cosine =
		gradient_cosine(&remote_gradient, &pulse_fixture_golden_remote_gradient);
	local_loss_error = fabsf(local_loss - pulse_fixture_golden_local_loss);
	remote_loss_error = fabsf(remote_loss - pulse_fixture_golden_remote_loss);
	head_error = maximum_error(head.parameters,
							   pulse_fixture_golden_mixed_head.parameters,
							   PULSE_HEAD_PARAMETER_COUNT);
	agreement_error = fabsf(mix_stats.agreement - pulse_fixture_golden_agreement);
	alpha_error = fabsf(mix_stats.alpha - pulse_fixture_golden_alpha);
	remote_scale_error = fabsf(mix_stats.remote_scale - pulse_fixture_golden_remote_scale);

	/*
	 * Exercise the gradient-specific entry points against bit-exact generated
	 * bytes after the numerical comparison no longer needs remote_gradient.
	 */
	remote_gradient = pulse_fixture_golden_remote_gradient;
	gradient_codec_serialize_status = pulse_protocol_gradient_serialize(&remote_gradient,
																		(uint8_t*) &remote_gradient,
																		sizeof(remote_gradient),
																		&gradient_codec_crc32);
	gradient_codec_deserialize_status =
		gradient_codec_serialize_status == 0
			? pulse_protocol_gradient_deserialize((const uint8_t*) &remote_gradient,
												  sizeof(remote_gradient),
												  &remote_gradient)
			: gradient_codec_serialize_status;
	gradient_codec_pass =
		gradient_codec_serialize_status == 0 && gradient_codec_deserialize_status == 0 &&
		gradient_codec_crc32 == pulse_fixture_golden_remote_gradient_fp32le_crc32 &&
		pulse_protocol_gradient_crc32(&remote_gradient) ==
			pulse_fixture_golden_remote_gradient_fp32le_crc32 &&
		memcmp(remote_gradient.parameters,
			   pulse_fixture_golden_remote_gradient.parameters,
			   sizeof(remote_gradient.parameters)) == 0;

	passed = codec_pass && gradient_codec_pass && isfinite(local_gradient_cosine) &&
			 isfinite(remote_gradient_cosine) && isfinite(agreement_error) &&
			 isfinite(alpha_error) && isfinite(remote_scale_error) &&
			 embedding_error <= PULSE_CORRECTNESS_MAX_ERROR_LIMIT &&
			 logits_error <= PULSE_CORRECTNESS_MAX_ERROR_LIMIT &&
			 local_gradient_error <= PULSE_CORRECTNESS_MAX_ERROR_LIMIT &&
			 remote_gradient_error <= PULSE_CORRECTNESS_MAX_ERROR_LIMIT &&
			 local_loss_error <= PULSE_CORRECTNESS_MAX_ERROR_LIMIT &&
			 remote_loss_error <= PULSE_CORRECTNESS_MAX_ERROR_LIMIT &&
			 head_error <= PULSE_CORRECTNESS_MAX_ERROR_LIMIT &&
			 agreement_error <= PULSE_CORRECTNESS_MAX_ERROR_LIMIT &&
			 alpha_error <= PULSE_CORRECTNESS_MAX_ERROR_LIMIT &&
			 remote_scale_error <= PULSE_CORRECTNESS_MAX_ERROR_LIMIT &&
			 local_gradient_cosine >= PULSE_CORRECTNESS_MIN_COSINE &&
			 remote_gradient_cosine >= PULSE_CORRECTNESS_MIN_COSINE;

	printk("PULSE_CORRECTNESS,board_id=%s,pass=%u,reference=%s,"
		   "fixture_sha256=%s,codec_pass=%u,codec_crc32=%08x,"
		   "codec_expected_crc32=%08x,gradient_codec_pass=%u,"
		   "gradient_codec_crc32=%08x,gradient_codec_expected_crc32=%08x,"
		   "embedding_max_abs_error=%.9g,"
		   "logits_max_abs_error=%.9g,"
		   "local_gradient_cosine=%.9g,local_gradient_max_abs_error=%.9g,"
		   "remote_gradient_cosine=%.9g,remote_gradient_max_abs_error=%.9g,"
		   "local_loss_abs_error=%.9g,remote_loss_abs_error=%.9g,"
		   "updated_head_max_abs_error=%.9g,utility_before=%.9g,agreement=%.9g,"
		   "golden_agreement=%.9g,agreement_abs_error=%.9g,"
		   "alpha=%.9g,golden_alpha=%.9g,alpha_abs_error=%.9g,"
		   "remote_scale=%.9g,golden_remote_scale=%.9g,remote_scale_abs_error=%.9g,"
		   "max_abs_error_limit=%.9g,min_cosine=%.9g\n",
		   board_id,
		   passed ? 1U : 0U,
		   pulse_fixture_reference_name,
		   pulse_fixture_sha256,
		   codec_pass ? 1U : 0U,
		   codec_crc32,
		   pulse_fixture_initial_head_fp32le_crc32,
		   gradient_codec_pass ? 1U : 0U,
		   gradient_codec_crc32,
		   pulse_fixture_golden_remote_gradient_fp32le_crc32,
		   (double) embedding_error,
		   (double) logits_error,
		   (double) local_gradient_cosine,
		   (double) local_gradient_error,
		   (double) remote_gradient_cosine,
		   (double) remote_gradient_error,
		   (double) local_loss_error,
		   (double) remote_loss_error,
		   (double) head_error,
		   (double) pulse_fixture_golden_utility_before,
		   (double) mix_stats.agreement,
		   (double) pulse_fixture_golden_agreement,
		   (double) agreement_error,
		   (double) mix_stats.alpha,
		   (double) pulse_fixture_golden_alpha,
		   (double) alpha_error,
		   (double) mix_stats.remote_scale,
		   (double) pulse_fixture_golden_remote_scale,
		   (double) remote_scale_error,
		   (double) PULSE_CORRECTNESS_MAX_ERROR_LIMIT,
		   (double) PULSE_CORRECTNESS_MIN_COSINE);

	return passed ? 0 : -ERANGE;
}
