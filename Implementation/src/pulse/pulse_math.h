#ifndef SENSWEAR_PULSE_MATH_H_
#define SENSWEAR_PULSE_MATH_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Firmware HAR topology; deploy Conv1d+BatchNorm pairs are folded into these tensors. */
#define PULSE_HAR_CHANNELS 3U
#define PULSE_HAR_WINDOW_LENGTH 128U
#define PULSE_HAR_SAMPLE_FLOATS (PULSE_HAR_CHANNELS * PULSE_HAR_WINDOW_LENGTH)
#define PULSE_HAR_BATCH_SIZE 16U
#define PULSE_HAR_CLASS_COUNT 6U
#define PULSE_EMBEDDING_SIZE 64U
#define PULSE_HEAD_HIDDEN_SIZE 64U

#define PULSE_ENCODER_CONV1_OUT_CHANNELS 32U
#define PULSE_ENCODER_CONV1_KERNEL_SIZE 5U
#define PULSE_ENCODER_POOL_LENGTH 64U
#define PULSE_ENCODER_CONV2_OUT_CHANNELS 64U
#define PULSE_ENCODER_CONV2_KERNEL_SIZE 5U

#define PULSE_ENCODER_CONV1_WEIGHT_COUNT \
	(PULSE_ENCODER_CONV1_OUT_CHANNELS * PULSE_HAR_CHANNELS * PULSE_ENCODER_CONV1_KERNEL_SIZE)
#define PULSE_ENCODER_CONV1_BIAS_COUNT PULSE_ENCODER_CONV1_OUT_CHANNELS
#define PULSE_ENCODER_CONV2_WEIGHT_COUNT                                   \
	(PULSE_ENCODER_CONV2_OUT_CHANNELS * PULSE_ENCODER_CONV1_OUT_CHANNELS * \
	 PULSE_ENCODER_CONV2_KERNEL_SIZE)
#define PULSE_ENCODER_CONV2_BIAS_COUNT PULSE_ENCODER_CONV2_OUT_CHANNELS
#define PULSE_ENCODER_PROJECTION_WEIGHT_COUNT \
	(PULSE_EMBEDDING_SIZE * PULSE_ENCODER_CONV2_OUT_CHANNELS)
#define PULSE_ENCODER_PROJECTION_BIAS_COUNT PULSE_EMBEDDING_SIZE
#define PULSE_ENCODER_PARAMETER_COUNT                                    \
	(PULSE_ENCODER_CONV1_WEIGHT_COUNT + PULSE_ENCODER_CONV1_BIAS_COUNT + \
	 PULSE_ENCODER_CONV2_WEIGHT_COUNT + PULSE_ENCODER_CONV2_BIAS_COUNT + \
	 PULSE_ENCODER_PROJECTION_WEIGHT_COUNT + PULSE_ENCODER_PROJECTION_BIAS_COUNT)

/* PyTorch named_parameters() order: W1, b1, W2, b2. */
#define PULSE_HEAD_W1_OFFSET 0U
#define PULSE_HEAD_W1_COUNT (PULSE_HEAD_HIDDEN_SIZE * PULSE_EMBEDDING_SIZE)
#define PULSE_HEAD_B1_OFFSET (PULSE_HEAD_W1_OFFSET + PULSE_HEAD_W1_COUNT)
#define PULSE_HEAD_B1_COUNT PULSE_HEAD_HIDDEN_SIZE
#define PULSE_HEAD_W2_OFFSET (PULSE_HEAD_B1_OFFSET + PULSE_HEAD_B1_COUNT)
#define PULSE_HEAD_W2_COUNT (PULSE_HAR_CLASS_COUNT * PULSE_HEAD_HIDDEN_SIZE)
#define PULSE_HEAD_B2_OFFSET (PULSE_HEAD_W2_OFFSET + PULSE_HEAD_W2_COUNT)
#define PULSE_HEAD_B2_COUNT PULSE_HAR_CLASS_COUNT
#define PULSE_HEAD_PARAMETER_COUNT (PULSE_HEAD_B2_OFFSET + PULSE_HEAD_B2_COUNT)

typedef enum {
	PULSE_STATUS_OK = 0,
	PULSE_STATUS_BAD_ARGUMENT,
	PULSE_STATUS_BAD_ARTIFACT,
	PULSE_STATUS_BAD_LABEL,
	PULSE_STATUS_CAPACITY,
	PULSE_STATUS_NUMERIC_ERROR
} pulse_status_t;

/* A generated artifact supplies immutable storage and its exact element count. */
typedef struct {
	const float* data;
	size_t count;
} pulse_const_tensor_t;

/* All arrays use PyTorch row-major order: [out][in][kernel] or [out][in]. */
typedef struct {
	pulse_const_tensor_t conv1_weight;
	pulse_const_tensor_t conv1_bias;
	pulse_const_tensor_t conv2_weight;
	pulse_const_tensor_t conv2_bias;
	pulse_const_tensor_t projection_weight;
	pulse_const_tensor_t projection_bias;
} pulse_encoder_artifact_t;

typedef struct {
	pulse_const_tensor_t parameters;
} pulse_head_artifact_t;

/* Wire/storage order is exactly W1, b1, W2, b2 with no struct padding in the payload array. */
typedef struct {
	float parameters[PULSE_HEAD_PARAMETER_COUNT];
} pulse_head_t;

typedef struct {
	float parameters[PULSE_HEAD_PARAMETER_COUNT];
} pulse_gradient_t;

typedef struct {
	const float* samples;
	size_t sample_float_count;
	size_t sample_stride_floats;
	size_t batch_size;
} pulse_sample_batch_t;

/*
 * Samples are normalized FP32 windows in contiguous [channel][time] order. A stride larger
 * than PULSE_HAR_SAMPLE_FLOATS permits records with trailing metadata.
 */
typedef struct {
	const float* samples;
	size_t sample_float_count;
	size_t sample_stride_floats;
	const uint8_t* labels;
	size_t label_count;
	size_t batch_size;
} pulse_batch_t;

typedef struct {
	float values[PULSE_HAR_BATCH_SIZE * PULSE_EMBEDDING_SIZE];
} pulse_embedding_batch_t;

/* Caller-owned scratch storage. It may be static, on a worker stack, or in a fixed arena. */
typedef struct {
	float pooled[PULSE_ENCODER_CONV1_OUT_CHANNELS * PULSE_ENCODER_POOL_LENGTH];
	float encoder_features[PULSE_ENCODER_CONV2_OUT_CHANNELS];
	float embedding[PULSE_EMBEDDING_SIZE];
	float hidden_pre[PULSE_HEAD_HIDDEN_SIZE];
	float hidden[PULSE_HEAD_HIDDEN_SIZE];
	float hidden_gradient[PULSE_HEAD_HIDDEN_SIZE];
	float logits[PULSE_HAR_CLASS_COUNT];
	float probabilities[PULSE_HAR_CLASS_COUNT];
} pulse_model_workspace_t;

typedef struct {
	uint32_t steps_completed;
	float mean_loss;
	float last_loss;
} pulse_local_sgd_stats_t;

typedef struct {
	float agreement;
	float alpha;
	float remote_scale;
	float local_norm_sq;
	float remote_norm_sq;
} pulse_mix_stats_t;

typedef struct {
	double local_norm_sq;
	double remote_norm_sq;
	double dot_product;
} pulse_gradient_products_t;

size_t pulse_model_workspace_size(void);
size_t pulse_head_storage_size(void);
size_t pulse_gradient_storage_size(void);
size_t pulse_embedding_batch_storage_size(void);

pulse_status_t pulse_encoder_artifact_validate(const pulse_encoder_artifact_t* encoder);
pulse_status_t pulse_head_artifact_validate(const pulse_head_artifact_t* head_artifact);
pulse_status_t pulse_sample_batch_validate(const pulse_sample_batch_t* batch);
pulse_status_t pulse_batch_validate(const pulse_batch_t* batch);

pulse_status_t pulse_head_load(pulse_head_t* destination, const pulse_head_artifact_t* source);

pulse_status_t pulse_encoder_forward(const pulse_encoder_artifact_t* encoder,
									 const float* sample,
									 size_t sample_float_count,
									 pulse_model_workspace_t* workspace,
									 float embedding[PULSE_EMBEDDING_SIZE]);

/* Separates the encoder stage for GPIO/cycle timing and reusable frozen embeddings. */
pulse_status_t pulse_encoder_batch_forward(const pulse_encoder_artifact_t* encoder,
										   const pulse_sample_batch_t* batch,
										   pulse_model_workspace_t* workspace,
										   pulse_embedding_batch_t* embeddings);

pulse_status_t pulse_head_forward(const pulse_head_t* head,
								  const float embedding[PULSE_EMBEDDING_SIZE],
								  pulse_model_workspace_t* workspace,
								  float logits[PULSE_HAR_CLASS_COUNT]);

pulse_status_t pulse_head_predict(const pulse_encoder_artifact_t* encoder,
								  const pulse_head_t* head,
								  const float* sample,
								  size_t sample_float_count,
								  pulse_model_workspace_t* workspace,
								  uint8_t* class_index);

void pulse_gradient_zero(pulse_gradient_t* gradient);

/* Head-only mean cross-entropy/backprop stage over caller-owned 16 x 64 embeddings. */
pulse_status_t pulse_head_batch_gradient_from_embeddings(const pulse_head_t* head,
														 const pulse_embedding_batch_t* embeddings,
														 const uint8_t* labels,
														 size_t label_count,
														 pulse_model_workspace_t* workspace,
														 pulse_gradient_t* gradient,
														 float* mean_loss);

/* Computes d(mean cross-entropy)/d[W1,b1,W2,b2] for exactly 16 samples. */
pulse_status_t pulse_head_batch_gradient(const pulse_encoder_artifact_t* encoder,
										 const pulse_head_t* head,
										 const pulse_batch_t* batch,
										 pulse_model_workspace_t* workspace,
										 pulse_gradient_t* gradient,
										 float* mean_loss);

pulse_status_t pulse_head_apply_local_gradient(pulse_head_t* head,
											   const pulse_gradient_t* gradient,
											   float learning_rate);

/*
 * Runs one gradient/update per supplied batch. local_steps is R and must not exceed
 * batch_count. gradient_workspace is overwritten at every step.
 */
pulse_status_t pulse_head_local_sgd(const pulse_encoder_artifact_t* encoder,
									pulse_head_t* head,
									const pulse_batch_t* batches,
									size_t batch_count,
									uint32_t local_steps,
									float learning_rate,
									pulse_model_workspace_t* model_workspace,
									pulse_gradient_t* gradient_workspace,
									pulse_local_sgd_stats_t* stats);

pulse_status_t pulse_cosine_agreement(const pulse_gradient_t* local_gradient,
									  const pulse_gradient_t* remote_gradient,
									  float norm_epsilon,
									  float* agreement);

/* One traversal shared by separately timed agreement and normalization stages. */
pulse_status_t pulse_gradient_products(const pulse_gradient_t* local_gradient,
									   const pulse_gradient_t* remote_gradient,
									   pulse_gradient_products_t* products);

pulse_status_t pulse_agreement_from_products(const pulse_gradient_products_t* products,
											 float norm_epsilon,
											 float* agreement);

/* Repository contract: epsilon floors the squared remote norm before sqrt. */
pulse_status_t pulse_remote_scale_from_products(const pulse_gradient_products_t* products,
												float norm_epsilon,
												float* remote_scale);

pulse_status_t pulse_compute_mixing_alpha(float alpha_max,
										  float utility_before,
										  float agreement,
										  float* alpha);

/* Pure mixing/update stage using a previously measured remote scale. */
pulse_status_t pulse_head_apply_weighted_gradient(pulse_head_t* head,
												  const pulse_gradient_t* local_gradient,
												  const pulse_gradient_t* remote_gradient,
												  float learning_rate,
												  float alpha,
												  float remote_scale);

/* Applies h <- h - lr*((1-alpha)*g_local + alpha*scale*g_remote). */
pulse_status_t pulse_head_apply_mixed_gradient(pulse_head_t* head,
											   const pulse_gradient_t* local_gradient,
											   const pulse_gradient_t* remote_gradient,
											   float learning_rate,
											   float alpha,
											   float norm_epsilon,
											   pulse_mix_stats_t* stats);

/* Computes agreement and alpha from the pre-update utility, then applies the mixed update. */
pulse_status_t pulse_head_apply_peer_update(pulse_head_t* head,
											const pulse_gradient_t* local_gradient,
											const pulse_gradient_t* remote_gradient,
											float learning_rate,
											float alpha_max,
											float utility_before,
											float norm_epsilon,
											pulse_mix_stats_t* stats);

#ifdef __cplusplus
}
#endif

#endif /* SENSWEAR_PULSE_MATH_H_ */
