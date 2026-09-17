#include "pulse_math.h"

#include <float.h>
#include <math.h>
#include <string.h>

static int tensor_has_exact_count(const pulse_const_tensor_t* tensor, size_t expected_count) {
	return tensor != NULL && tensor->data != NULL && tensor->count == expected_count;
}

static int vector_is_finite(const float* values, size_t count) {
	size_t index;

	for (index = 0U; index < count; ++index) {
		if (!isfinite(values[index])) {
			return 0;
		}
	}

	return 1;
}

static int double_is_finite(double value) {
	return value <= DBL_MAX && value >= -DBL_MAX;
}

static size_t conv1_weight_index(size_t output, size_t input, size_t kernel) {
	return ((output * PULSE_HAR_CHANNELS + input) * PULSE_ENCODER_CONV1_KERNEL_SIZE) + kernel;
}

static size_t conv2_weight_index(size_t output, size_t input, size_t kernel) {
	return ((output * PULSE_ENCODER_CONV1_OUT_CHANNELS + input) * PULSE_ENCODER_CONV2_KERNEL_SIZE) +
		   kernel;
}

static size_t head_w1_index(size_t output, size_t input) {
	return PULSE_HEAD_W1_OFFSET + output * PULSE_EMBEDDING_SIZE + input;
}

static size_t head_w2_index(size_t output, size_t input) {
	return PULSE_HEAD_W2_OFFSET + output * PULSE_HEAD_HIDDEN_SIZE + input;
}

static float conv1_value(const pulse_encoder_artifact_t* encoder,
						 const float* sample,
						 size_t output_channel,
						 size_t output_time) {
	float value = encoder->conv1_bias.data[output_channel];
	size_t input_channel;
	size_t kernel;

	for (input_channel = 0U; input_channel < PULSE_HAR_CHANNELS; ++input_channel) {
		for (kernel = 0U; kernel < PULSE_ENCODER_CONV1_KERNEL_SIZE; ++kernel) {
			int input_time = (int) output_time + (int) kernel - 2;

			if (input_time >= 0 && input_time < (int) PULSE_HAR_WINDOW_LENGTH) {
				value += encoder->conv1_weight
							 .data[conv1_weight_index(output_channel, input_channel, kernel)] *
						 sample[input_channel * PULSE_HAR_WINDOW_LENGTH + (size_t) input_time];
			}
		}
	}

	return value > 0.0f ? value : 0.0f;
}

static pulse_status_t encoder_forward_unchecked(const pulse_encoder_artifact_t* encoder,
												const float* sample,
												pulse_model_workspace_t* workspace) {
	size_t output_channel;
	size_t pooled_time;
	size_t output_time;
	size_t input_channel;
	size_t kernel;
	size_t embedding_index;

	/* ReLU and MaxPool1d(2) are fused, so the 32 x 128 Conv1 output is not retained. */
	for (output_channel = 0U; output_channel < PULSE_ENCODER_CONV1_OUT_CHANNELS; ++output_channel) {
		for (pooled_time = 0U; pooled_time < PULSE_ENCODER_POOL_LENGTH; ++pooled_time) {
			float first = conv1_value(encoder, sample, output_channel, 2U * pooled_time);
			float second = conv1_value(encoder, sample, output_channel, 2U * pooled_time + 1U);

			workspace->pooled[output_channel * PULSE_ENCODER_POOL_LENGTH + pooled_time] =
				first > second ? first : second;
		}
	}

	/* Conv2, ReLU, and time averaging are fused; only the 64 channel means remain. */
	for (output_channel = 0U; output_channel < PULSE_ENCODER_CONV2_OUT_CHANNELS; ++output_channel) {
		float time_sum = 0.0f;

		for (output_time = 0U; output_time < PULSE_ENCODER_POOL_LENGTH; ++output_time) {
			float value = encoder->conv2_bias.data[output_channel];

			for (input_channel = 0U; input_channel < PULSE_ENCODER_CONV1_OUT_CHANNELS;
				 ++input_channel) {
				for (kernel = 0U; kernel < PULSE_ENCODER_CONV2_KERNEL_SIZE; ++kernel) {
					int input_time = (int) output_time + (int) kernel - 2;

					if (input_time >= 0 && input_time < (int) PULSE_ENCODER_POOL_LENGTH) {
						value +=
							encoder->conv2_weight
								.data[conv2_weight_index(output_channel, input_channel, kernel)] *
							workspace->pooled[input_channel * PULSE_ENCODER_POOL_LENGTH +
											  (size_t) input_time];
					}
				}
			}

			if (value > 0.0f) {
				time_sum += value;
			}
		}

		workspace->encoder_features[output_channel] = time_sum / (float) PULSE_ENCODER_POOL_LENGTH;
	}

	for (embedding_index = 0U; embedding_index < PULSE_EMBEDDING_SIZE; ++embedding_index) {
		float value = encoder->projection_bias.data[embedding_index];

		for (input_channel = 0U; input_channel < PULSE_ENCODER_CONV2_OUT_CHANNELS;
			 ++input_channel) {
			value += encoder->projection_weight
						 .data[embedding_index * PULSE_ENCODER_CONV2_OUT_CHANNELS + input_channel] *
					 workspace->encoder_features[input_channel];
		}

		workspace->embedding[embedding_index] = value;
	}

	if (!vector_is_finite(workspace->embedding, PULSE_EMBEDDING_SIZE)) {
		return PULSE_STATUS_NUMERIC_ERROR;
	}

	return PULSE_STATUS_OK;
}

static pulse_status_t head_forward_unchecked(const pulse_head_t* head,
											 const float* embedding,
											 pulse_model_workspace_t* workspace) {
	size_t hidden_index;
	size_t input_index;
	size_t class_index;

	for (hidden_index = 0U; hidden_index < PULSE_HEAD_HIDDEN_SIZE; ++hidden_index) {
		float value = head->parameters[PULSE_HEAD_B1_OFFSET + hidden_index];

		for (input_index = 0U; input_index < PULSE_EMBEDDING_SIZE; ++input_index) {
			value +=
				head->parameters[head_w1_index(hidden_index, input_index)] * embedding[input_index];
		}

		workspace->hidden_pre[hidden_index] = value;
		workspace->hidden[hidden_index] = value > 0.0f ? value : 0.0f;
	}

	for (class_index = 0U; class_index < PULSE_HAR_CLASS_COUNT; ++class_index) {
		float value = head->parameters[PULSE_HEAD_B2_OFFSET + class_index];

		for (hidden_index = 0U; hidden_index < PULSE_HEAD_HIDDEN_SIZE; ++hidden_index) {
			value += head->parameters[head_w2_index(class_index, hidden_index)] *
					 workspace->hidden[hidden_index];
		}

		workspace->logits[class_index] = value;
	}

	if (!vector_is_finite(workspace->logits, PULSE_HAR_CLASS_COUNT)) {
		return PULSE_STATUS_NUMERIC_ERROR;
	}

	return PULSE_STATUS_OK;
}

static pulse_status_t labels_validate(const uint8_t* labels, size_t label_count) {
	size_t sample_index;

	if (labels == NULL) {
		return PULSE_STATUS_BAD_ARGUMENT;
	}

	if (label_count < PULSE_HAR_BATCH_SIZE) {
		return PULSE_STATUS_BAD_ARTIFACT;
	}

	for (sample_index = 0U; sample_index < PULSE_HAR_BATCH_SIZE; ++sample_index) {
		if (labels[sample_index] >= PULSE_HAR_CLASS_COUNT) {
			return PULSE_STATUS_BAD_LABEL;
		}
	}

	return PULSE_STATUS_OK;
}

static pulse_status_t head_gradient_accumulate(const pulse_head_t* head,
											   const float* embedding,
											   uint8_t label,
											   pulse_model_workspace_t* workspace,
											   pulse_gradient_t* gradient,
											   double* loss_sum) {
	const float inverse_batch_size = 1.0f / (float) PULSE_HAR_BATCH_SIZE;
	float maximum_logit;
	float exponential_sum = 0.0f;
	pulse_status_t status;
	size_t class_index;
	size_t hidden_index;
	size_t input_index;

	status = head_forward_unchecked(head, embedding, workspace);
	if (status != PULSE_STATUS_OK) {
		return status;
	}

	maximum_logit = workspace->logits[0];
	for (class_index = 1U; class_index < PULSE_HAR_CLASS_COUNT; ++class_index) {
		if (workspace->logits[class_index] > maximum_logit) {
			maximum_logit = workspace->logits[class_index];
		}
	}

	for (class_index = 0U; class_index < PULSE_HAR_CLASS_COUNT; ++class_index) {
		float exponential = expf(workspace->logits[class_index] - maximum_logit);

		workspace->probabilities[class_index] = exponential;
		exponential_sum += exponential;
	}

	if (!isfinite(exponential_sum) || exponential_sum <= 0.0f) {
		return PULSE_STATUS_NUMERIC_ERROR;
	}

	*loss_sum +=
		(double) maximum_logit + (double) logf(exponential_sum) - (double) workspace->logits[label];

	for (hidden_index = 0U; hidden_index < PULSE_HEAD_HIDDEN_SIZE; ++hidden_index) {
		workspace->hidden_gradient[hidden_index] = 0.0f;
	}

	for (class_index = 0U; class_index < PULSE_HAR_CLASS_COUNT; ++class_index) {
		float probability = workspace->probabilities[class_index] / exponential_sum;
		float delta = probability - (class_index == (size_t) label ? 1.0f : 0.0f);

		delta *= inverse_batch_size;
		gradient->parameters[PULSE_HEAD_B2_OFFSET + class_index] += delta;

		for (hidden_index = 0U; hidden_index < PULSE_HEAD_HIDDEN_SIZE; ++hidden_index) {
			gradient->parameters[head_w2_index(class_index, hidden_index)] +=
				delta * workspace->hidden[hidden_index];
			workspace->hidden_gradient[hidden_index] +=
				head->parameters[head_w2_index(class_index, hidden_index)] * delta;
		}
	}

	for (hidden_index = 0U; hidden_index < PULSE_HEAD_HIDDEN_SIZE; ++hidden_index) {
		float delta = workspace->hidden_pre[hidden_index] > 0.0f
						  ? workspace->hidden_gradient[hidden_index]
						  : 0.0f;

		gradient->parameters[PULSE_HEAD_B1_OFFSET + hidden_index] += delta;
		for (input_index = 0U; input_index < PULSE_EMBEDDING_SIZE; ++input_index) {
			gradient->parameters[head_w1_index(hidden_index, input_index)] +=
				delta * embedding[input_index];
		}
	}

	return PULSE_STATUS_OK;
}

static pulse_status_t head_gradient_finish(double loss_sum,
										   const pulse_gradient_t* gradient,
										   float* mean_loss) {
	if (!double_is_finite(loss_sum) ||
		!vector_is_finite(gradient->parameters, PULSE_HEAD_PARAMETER_COUNT)) {
		return PULSE_STATUS_NUMERIC_ERROR;
	}

	if (mean_loss != NULL) {
		*mean_loss = (float) (loss_sum / (double) PULSE_HAR_BATCH_SIZE);
	}

	return PULSE_STATUS_OK;
}

/* Public so benchmark code can time this traversal independently. */
pulse_status_t pulse_gradient_products(const pulse_gradient_t* local_gradient,
									   const pulse_gradient_t* remote_gradient,
									   pulse_gradient_products_t* products) {
	double local_sum = 0.0;
	double remote_sum = 0.0;
	double dot_sum = 0.0;
	size_t index;

	if (local_gradient == NULL || remote_gradient == NULL || products == NULL) {
		return PULSE_STATUS_BAD_ARGUMENT;
	}

	for (index = 0U; index < PULSE_HEAD_PARAMETER_COUNT; ++index) {
		double local = (double) local_gradient->parameters[index];
		double remote = (double) remote_gradient->parameters[index];

		if (!double_is_finite(local) || !double_is_finite(remote)) {
			return PULSE_STATUS_NUMERIC_ERROR;
		}

		local_sum += local * local;
		remote_sum += remote * remote;
		dot_sum += local * remote;
	}

	if (!double_is_finite(local_sum) || !double_is_finite(remote_sum) ||
		!double_is_finite(dot_sum)) {
		return PULSE_STATUS_NUMERIC_ERROR;
	}

	products->local_norm_sq = local_sum;
	products->remote_norm_sq = remote_sum;
	products->dot_product = dot_sum;
	return PULSE_STATUS_OK;
}

size_t pulse_model_workspace_size(void) {
	return sizeof(pulse_model_workspace_t);
}

size_t pulse_head_storage_size(void) {
	return sizeof(pulse_head_t);
}

size_t pulse_gradient_storage_size(void) {
	return sizeof(pulse_gradient_t);
}

size_t pulse_embedding_batch_storage_size(void) {
	return sizeof(pulse_embedding_batch_t);
}

pulse_status_t pulse_encoder_artifact_validate(const pulse_encoder_artifact_t* encoder) {
	if (encoder == NULL) {
		return PULSE_STATUS_BAD_ARGUMENT;
	}

	if (!tensor_has_exact_count(&encoder->conv1_weight, PULSE_ENCODER_CONV1_WEIGHT_COUNT) ||
		!tensor_has_exact_count(&encoder->conv1_bias, PULSE_ENCODER_CONV1_BIAS_COUNT) ||
		!tensor_has_exact_count(&encoder->conv2_weight, PULSE_ENCODER_CONV2_WEIGHT_COUNT) ||
		!tensor_has_exact_count(&encoder->conv2_bias, PULSE_ENCODER_CONV2_BIAS_COUNT) ||
		!tensor_has_exact_count(&encoder->projection_weight,
								PULSE_ENCODER_PROJECTION_WEIGHT_COUNT) ||
		!tensor_has_exact_count(&encoder->projection_bias, PULSE_ENCODER_PROJECTION_BIAS_COUNT)) {
		return PULSE_STATUS_BAD_ARTIFACT;
	}

	return PULSE_STATUS_OK;
}

pulse_status_t pulse_head_artifact_validate(const pulse_head_artifact_t* head_artifact) {
	if (head_artifact == NULL) {
		return PULSE_STATUS_BAD_ARGUMENT;
	}

	if (!tensor_has_exact_count(&head_artifact->parameters, PULSE_HEAD_PARAMETER_COUNT)) {
		return PULSE_STATUS_BAD_ARTIFACT;
	}

	return PULSE_STATUS_OK;
}

pulse_status_t pulse_sample_batch_validate(const pulse_sample_batch_t* batch) {
	size_t required_float_count;

	if (batch == NULL || batch->samples == NULL) {
		return PULSE_STATUS_BAD_ARGUMENT;
	}

	if (batch->batch_size != PULSE_HAR_BATCH_SIZE ||
		batch->sample_stride_floats < PULSE_HAR_SAMPLE_FLOATS) {
		return PULSE_STATUS_BAD_ARTIFACT;
	}

	if (batch->sample_stride_floats >
		(SIZE_MAX - PULSE_HAR_SAMPLE_FLOATS) / (PULSE_HAR_BATCH_SIZE - 1U)) {
		return PULSE_STATUS_BAD_ARTIFACT;
	}

	required_float_count =
		(PULSE_HAR_BATCH_SIZE - 1U) * batch->sample_stride_floats + PULSE_HAR_SAMPLE_FLOATS;
	if (batch->sample_float_count < required_float_count) {
		return PULSE_STATUS_BAD_ARTIFACT;
	}

	return PULSE_STATUS_OK;
}

pulse_status_t pulse_batch_validate(const pulse_batch_t* batch) {
	pulse_sample_batch_t samples;
	pulse_status_t status;

	if (batch == NULL || batch->labels == NULL) {
		return PULSE_STATUS_BAD_ARGUMENT;
	}

	samples.samples = batch->samples;
	samples.sample_float_count = batch->sample_float_count;
	samples.sample_stride_floats = batch->sample_stride_floats;
	samples.batch_size = batch->batch_size;
	status = pulse_sample_batch_validate(&samples);
	if (status != PULSE_STATUS_OK) {
		return status;
	}

	if (batch->label_count < PULSE_HAR_BATCH_SIZE) {
		return PULSE_STATUS_BAD_ARTIFACT;
	}

	return PULSE_STATUS_OK;
}

pulse_status_t pulse_head_load(pulse_head_t* destination, const pulse_head_artifact_t* source) {
	pulse_status_t status;

	if (destination == NULL) {
		return PULSE_STATUS_BAD_ARGUMENT;
	}

	status = pulse_head_artifact_validate(source);
	if (status != PULSE_STATUS_OK) {
		return status;
	}

	memcpy(destination->parameters, source->parameters.data, sizeof(destination->parameters));
	return PULSE_STATUS_OK;
}

pulse_status_t pulse_encoder_forward(const pulse_encoder_artifact_t* encoder,
									 const float* sample,
									 size_t sample_float_count,
									 pulse_model_workspace_t* workspace,
									 float embedding[PULSE_EMBEDDING_SIZE]) {
	pulse_status_t status;

	if (sample == NULL || workspace == NULL || embedding == NULL) {
		return PULSE_STATUS_BAD_ARGUMENT;
	}

	if (sample_float_count < PULSE_HAR_SAMPLE_FLOATS) {
		return PULSE_STATUS_BAD_ARTIFACT;
	}

	status = pulse_encoder_artifact_validate(encoder);
	if (status != PULSE_STATUS_OK) {
		return status;
	}

	status = encoder_forward_unchecked(encoder, sample, workspace);
	if (status != PULSE_STATUS_OK) {
		return status;
	}

	memcpy(embedding, workspace->embedding, sizeof(workspace->embedding));
	return PULSE_STATUS_OK;
}

pulse_status_t pulse_encoder_batch_forward(const pulse_encoder_artifact_t* encoder,
										   const pulse_sample_batch_t* batch,
										   pulse_model_workspace_t* workspace,
										   pulse_embedding_batch_t* embeddings) {
	pulse_status_t status;
	size_t sample_index;

	if (workspace == NULL || embeddings == NULL) {
		return PULSE_STATUS_BAD_ARGUMENT;
	}

	status = pulse_encoder_artifact_validate(encoder);
	if (status != PULSE_STATUS_OK) {
		return status;
	}

	status = pulse_sample_batch_validate(batch);
	if (status != PULSE_STATUS_OK) {
		return status;
	}

	for (sample_index = 0U; sample_index < PULSE_HAR_BATCH_SIZE; ++sample_index) {
		const float* sample = batch->samples + sample_index * batch->sample_stride_floats;

		status = encoder_forward_unchecked(encoder, sample, workspace);
		if (status != PULSE_STATUS_OK) {
			return status;
		}

		memcpy(&embeddings->values[sample_index * PULSE_EMBEDDING_SIZE],
			   workspace->embedding,
			   sizeof(workspace->embedding));
	}

	return PULSE_STATUS_OK;
}

pulse_status_t pulse_head_forward(const pulse_head_t* head,
								  const float embedding[PULSE_EMBEDDING_SIZE],
								  pulse_model_workspace_t* workspace,
								  float logits[PULSE_HAR_CLASS_COUNT]) {
	pulse_status_t status;

	if (head == NULL || embedding == NULL || workspace == NULL || logits == NULL) {
		return PULSE_STATUS_BAD_ARGUMENT;
	}

	status = head_forward_unchecked(head, embedding, workspace);
	if (status != PULSE_STATUS_OK) {
		return status;
	}

	memcpy(logits, workspace->logits, sizeof(workspace->logits));
	return PULSE_STATUS_OK;
}

pulse_status_t pulse_head_predict(const pulse_encoder_artifact_t* encoder,
								  const pulse_head_t* head,
								  const float* sample,
								  size_t sample_float_count,
								  pulse_model_workspace_t* workspace,
								  uint8_t* class_index) {
	pulse_status_t status;
	size_t index;
	size_t best = 0U;

	if (head == NULL || workspace == NULL || class_index == NULL) {
		return PULSE_STATUS_BAD_ARGUMENT;
	}

	status =
		pulse_encoder_forward(encoder, sample, sample_float_count, workspace, workspace->embedding);
	if (status != PULSE_STATUS_OK) {
		return status;
	}

	status = pulse_head_forward(head, workspace->embedding, workspace, workspace->logits);
	if (status != PULSE_STATUS_OK) {
		return status;
	}

	for (index = 1U; index < PULSE_HAR_CLASS_COUNT; ++index) {
		if (workspace->logits[index] > workspace->logits[best]) {
			best = index;
		}
	}

	*class_index = (uint8_t) best;
	return PULSE_STATUS_OK;
}

void pulse_gradient_zero(pulse_gradient_t* gradient) {
	if (gradient != NULL) {
		memset(gradient->parameters, 0, sizeof(gradient->parameters));
	}
}

pulse_status_t pulse_head_batch_gradient_from_embeddings(const pulse_head_t* head,
														 const pulse_embedding_batch_t* embeddings,
														 const uint8_t* labels,
														 size_t label_count,
														 pulse_model_workspace_t* workspace,
														 pulse_gradient_t* gradient,
														 float* mean_loss) {
	double loss_sum = 0.0;
	pulse_status_t status;
	size_t sample_index;

	if (head == NULL || embeddings == NULL || workspace == NULL || gradient == NULL) {
		return PULSE_STATUS_BAD_ARGUMENT;
	}

	status = labels_validate(labels, label_count);
	if (status != PULSE_STATUS_OK) {
		return status;
	}

	if (!vector_is_finite(head->parameters, PULSE_HEAD_PARAMETER_COUNT) ||
		!vector_is_finite(embeddings->values, PULSE_HAR_BATCH_SIZE * PULSE_EMBEDDING_SIZE)) {
		return PULSE_STATUS_NUMERIC_ERROR;
	}

	pulse_gradient_zero(gradient);
	for (sample_index = 0U; sample_index < PULSE_HAR_BATCH_SIZE; ++sample_index) {
		status = head_gradient_accumulate(head,
										  &embeddings->values[sample_index * PULSE_EMBEDDING_SIZE],
										  labels[sample_index],
										  workspace,
										  gradient,
										  &loss_sum);
		if (status != PULSE_STATUS_OK) {
			return status;
		}
	}

	return head_gradient_finish(loss_sum, gradient, mean_loss);
}

pulse_status_t pulse_head_batch_gradient(const pulse_encoder_artifact_t* encoder,
										 const pulse_head_t* head,
										 const pulse_batch_t* batch,
										 pulse_model_workspace_t* workspace,
										 pulse_gradient_t* gradient,
										 float* mean_loss) {
	pulse_status_t status;
	double loss_sum = 0.0;
	size_t sample_index;

	if (head == NULL || workspace == NULL || gradient == NULL) {
		return PULSE_STATUS_BAD_ARGUMENT;
	}

	status = pulse_encoder_artifact_validate(encoder);
	if (status != PULSE_STATUS_OK) {
		return status;
	}

	status = pulse_batch_validate(batch);
	if (status != PULSE_STATUS_OK) {
		return status;
	}

	if (!vector_is_finite(head->parameters, PULSE_HEAD_PARAMETER_COUNT)) {
		return PULSE_STATUS_NUMERIC_ERROR;
	}

	status = labels_validate(batch->labels, batch->label_count);
	if (status != PULSE_STATUS_OK) {
		return status;
	}

	pulse_gradient_zero(gradient);

	for (sample_index = 0U; sample_index < PULSE_HAR_BATCH_SIZE; ++sample_index) {
		const float* sample = batch->samples + sample_index * batch->sample_stride_floats;

		status = encoder_forward_unchecked(encoder, sample, workspace);
		if (status != PULSE_STATUS_OK) {
			return status;
		}

		status = head_gradient_accumulate(head,
										  workspace->embedding,
										  batch->labels[sample_index],
										  workspace,
										  gradient,
										  &loss_sum);
		if (status != PULSE_STATUS_OK) {
			return status;
		}
	}

	return head_gradient_finish(loss_sum, gradient, mean_loss);
}

pulse_status_t pulse_head_apply_local_gradient(pulse_head_t* head,
											   const pulse_gradient_t* gradient,
											   float learning_rate) {
	size_t index;

	if (head == NULL || gradient == NULL || !isfinite(learning_rate) || learning_rate < 0.0f) {
		return PULSE_STATUS_BAD_ARGUMENT;
	}

	if (!vector_is_finite(head->parameters, PULSE_HEAD_PARAMETER_COUNT) ||
		!vector_is_finite(gradient->parameters, PULSE_HEAD_PARAMETER_COUNT)) {
		return PULSE_STATUS_NUMERIC_ERROR;
	}

	for (index = 0U; index < PULSE_HEAD_PARAMETER_COUNT; ++index) {
		double updated = (double) head->parameters[index] -
						 (double) learning_rate * (double) gradient->parameters[index];

		if (!double_is_finite(updated) || updated > (double) FLT_MAX ||
			updated < -(double) FLT_MAX) {
			return PULSE_STATUS_NUMERIC_ERROR;
		}
	}

	for (index = 0U; index < PULSE_HEAD_PARAMETER_COUNT; ++index) {
		head->parameters[index] -= learning_rate * gradient->parameters[index];
	}

	return PULSE_STATUS_OK;
}

pulse_status_t pulse_head_local_sgd(const pulse_encoder_artifact_t* encoder,
									pulse_head_t* head,
									const pulse_batch_t* batches,
									size_t batch_count,
									uint32_t local_steps,
									float learning_rate,
									pulse_model_workspace_t* model_workspace,
									pulse_gradient_t* gradient_workspace,
									pulse_local_sgd_stats_t* stats) {
	double loss_sum = 0.0;
	uint32_t step;

	if (head == NULL || batches == NULL || model_workspace == NULL || gradient_workspace == NULL ||
		local_steps == 0U || (size_t) local_steps > batch_count) {
		return PULSE_STATUS_BAD_ARGUMENT;
	}

	if (stats != NULL) {
		memset(stats, 0, sizeof(*stats));
	}

	for (step = 0U; step < local_steps; ++step) {
		float loss;
		pulse_status_t status = pulse_head_batch_gradient(encoder,
														  head,
														  &batches[step],
														  model_workspace,
														  gradient_workspace,
														  &loss);

		if (status != PULSE_STATUS_OK) {
			return status;
		}

		status = pulse_head_apply_local_gradient(head, gradient_workspace, learning_rate);
		if (status != PULSE_STATUS_OK) {
			return status;
		}

		loss_sum += (double) loss;
		if (stats != NULL) {
			stats->steps_completed = step + 1U;
			stats->last_loss = loss;
			stats->mean_loss = (float) (loss_sum / (double) (step + 1U));
		}
	}

	return PULSE_STATUS_OK;
}

pulse_status_t pulse_cosine_agreement(const pulse_gradient_t* local_gradient,
									  const pulse_gradient_t* remote_gradient,
									  float norm_epsilon,
									  float* agreement) {
	pulse_gradient_products_t products;
	pulse_status_t status;

	status = pulse_gradient_products(local_gradient, remote_gradient, &products);
	if (status != PULSE_STATUS_OK) {
		return status;
	}

	return pulse_agreement_from_products(&products, norm_epsilon, agreement);
}

pulse_status_t pulse_agreement_from_products(const pulse_gradient_products_t* products,
											 float norm_epsilon,
											 float* agreement) {
	double denominator;
	double value;

	if (products == NULL || agreement == NULL || !isfinite(norm_epsilon) || norm_epsilon <= 0.0f ||
		!double_is_finite(products->local_norm_sq) || !double_is_finite(products->remote_norm_sq) ||
		!double_is_finite(products->dot_product) || products->local_norm_sq < 0.0 ||
		products->remote_norm_sq < 0.0) {
		return PULSE_STATUS_BAD_ARGUMENT;
	}

	denominator = sqrt(products->local_norm_sq) * sqrt(products->remote_norm_sq);
	if (denominator <= (double) norm_epsilon) {
		*agreement = 0.0f;
		return PULSE_STATUS_OK;
	}

	value = products->dot_product / denominator;
	if (value > 1.0) {
		value = 1.0;
	} else if (value < -1.0) {
		value = -1.0;
	}

	if (!double_is_finite(value)) {
		return PULSE_STATUS_NUMERIC_ERROR;
	}

	*agreement = (float) value;
	return PULSE_STATUS_OK;
}

pulse_status_t pulse_remote_scale_from_products(const pulse_gradient_products_t* products,
												float norm_epsilon,
												float* remote_scale) {
	double scale;

	if (products == NULL || remote_scale == NULL || !isfinite(norm_epsilon) ||
		norm_epsilon <= 0.0f || !double_is_finite(products->local_norm_sq) ||
		!double_is_finite(products->remote_norm_sq) || products->local_norm_sq < 0.0 ||
		products->remote_norm_sq < 0.0) {
		return PULSE_STATUS_BAD_ARGUMENT;
	}

	scale = sqrt(products->local_norm_sq / (products->remote_norm_sq > (double) norm_epsilon
												? products->remote_norm_sq
												: (double) norm_epsilon));
	if (!double_is_finite(scale) || scale > (double) FLT_MAX) {
		return PULSE_STATUS_NUMERIC_ERROR;
	}

	*remote_scale = (float) scale;
	return PULSE_STATUS_OK;
}

pulse_status_t pulse_compute_mixing_alpha(float alpha_max,
										  float utility_before,
										  float agreement,
										  float* alpha) {
	float positive_utility;
	float positive_agreement;

	if (alpha == NULL || !isfinite(alpha_max) || !isfinite(utility_before) ||
		!isfinite(agreement) || alpha_max < 0.0f || alpha_max > 1.0f || utility_before < -1.0f ||
		utility_before > 1.0f || agreement < -1.0f || agreement > 1.0f) {
		return PULSE_STATUS_BAD_ARGUMENT;
	}

	positive_utility = utility_before > 0.0f ? utility_before : 0.0f;
	positive_agreement = agreement > 0.0f ? agreement : 0.0f;
	*alpha = alpha_max * positive_utility * positive_agreement;
	return PULSE_STATUS_OK;
}

pulse_status_t pulse_head_apply_weighted_gradient(pulse_head_t* head,
												  const pulse_gradient_t* local_gradient,
												  const pulse_gradient_t* remote_gradient,
												  float learning_rate,
												  float alpha,
												  float remote_scale) {
	size_t index;

	if (head == NULL || local_gradient == NULL || remote_gradient == NULL ||
		!isfinite(learning_rate) || learning_rate < 0.0f || !isfinite(alpha) || alpha < 0.0f ||
		alpha > 1.0f || !isfinite(remote_scale) || remote_scale < 0.0f) {
		return PULSE_STATUS_BAD_ARGUMENT;
	}

	if (!vector_is_finite(head->parameters, PULSE_HEAD_PARAMETER_COUNT) ||
		!vector_is_finite(local_gradient->parameters, PULSE_HEAD_PARAMETER_COUNT) ||
		!vector_is_finite(remote_gradient->parameters, PULSE_HEAD_PARAMETER_COUNT)) {
		return PULSE_STATUS_NUMERIC_ERROR;
	}

	/* Preflight every result so a numeric failure cannot leave a partially updated head. */
	for (index = 0U; index < PULSE_HEAD_PARAMETER_COUNT; ++index) {
		double local = (double) local_gradient->parameters[index];
		double remote = (double) remote_gradient->parameters[index] * (double) remote_scale;
		double aggregate = (1.0 - (double) alpha) * local + (double) alpha * remote;
		double updated = (double) head->parameters[index] - (double) learning_rate * aggregate;

		if (!double_is_finite(updated) || updated > (double) FLT_MAX ||
			updated < -(double) FLT_MAX) {
			return PULSE_STATUS_NUMERIC_ERROR;
		}
	}

	for (index = 0U; index < PULSE_HEAD_PARAMETER_COUNT; ++index) {
		float scaled_remote = remote_gradient->parameters[index] * remote_scale;
		float aggregate =
			(1.0f - alpha) * local_gradient->parameters[index] + alpha * scaled_remote;

		head->parameters[index] -= learning_rate * aggregate;
	}

	return PULSE_STATUS_OK;
}

pulse_status_t pulse_head_apply_mixed_gradient(pulse_head_t* head,
											   const pulse_gradient_t* local_gradient,
											   const pulse_gradient_t* remote_gradient,
											   float learning_rate,
											   float alpha,
											   float norm_epsilon,
											   pulse_mix_stats_t* stats) {
	pulse_gradient_products_t products;
	float agreement;
	float remote_scale;
	pulse_status_t status;

	status = pulse_gradient_products(local_gradient, remote_gradient, &products);
	if (status != PULSE_STATUS_OK) {
		return status;
	}

	status = pulse_agreement_from_products(&products, norm_epsilon, &agreement);
	if (status != PULSE_STATUS_OK) {
		return status;
	}

	status = pulse_remote_scale_from_products(&products, norm_epsilon, &remote_scale);
	if (status != PULSE_STATUS_OK) {
		return status;
	}

	status = pulse_head_apply_weighted_gradient(head,
												local_gradient,
												remote_gradient,
												learning_rate,
												alpha,
												remote_scale);
	if (status != PULSE_STATUS_OK) {
		return status;
	}

	if (stats != NULL) {
		stats->agreement = agreement;
		stats->alpha = alpha;
		stats->remote_scale = remote_scale;
		stats->local_norm_sq = (float) products.local_norm_sq;
		stats->remote_norm_sq = (float) products.remote_norm_sq;
	}

	return PULSE_STATUS_OK;
}

pulse_status_t pulse_head_apply_peer_update(pulse_head_t* head,
											const pulse_gradient_t* local_gradient,
											const pulse_gradient_t* remote_gradient,
											float learning_rate,
											float alpha_max,
											float utility_before,
											float norm_epsilon,
											pulse_mix_stats_t* stats) {
	pulse_gradient_products_t products;
	float agreement;
	float alpha;
	float remote_scale;
	pulse_status_t status;

	status = pulse_gradient_products(local_gradient, remote_gradient, &products);
	if (status != PULSE_STATUS_OK) {
		return status;
	}

	status = pulse_agreement_from_products(&products, norm_epsilon, &agreement);
	if (status != PULSE_STATUS_OK) {
		return status;
	}

	status = pulse_compute_mixing_alpha(alpha_max, utility_before, agreement, &alpha);
	if (status != PULSE_STATUS_OK) {
		return status;
	}

	status = pulse_remote_scale_from_products(&products, norm_epsilon, &remote_scale);
	if (status != PULSE_STATUS_OK) {
		return status;
	}

	status = pulse_head_apply_weighted_gradient(head,
												local_gradient,
												remote_gradient,
												learning_rate,
												alpha,
												remote_scale);
	if (status != PULSE_STATUS_OK) {
		return status;
	}

	if (stats != NULL) {
		stats->agreement = agreement;
		stats->alpha = alpha;
		stats->remote_scale = remote_scale;
		stats->local_norm_sq = (float) products.local_norm_sq;
		stats->remote_norm_sq = (float) products.remote_norm_sq;
	}

	return PULSE_STATUS_OK;
}
