#include "pulse_protocol.h"

#include <errno.h>
#include <string.h>

#include <zephyr/bluetooth/att.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/byteorder.h>
#include <zephyr/sys/util.h>

BUILD_ASSERT(sizeof(float) == 4U, "PULSE wire protocol requires 32-bit float");
BUILD_ASSERT(PULSE_HEAD_PARAMETER_COUNT == 4550U, "PULSE HAR wire shape changed");
BUILD_ASSERT(sizeof(pulse_head_t) == PULSE_PROTOCOL_TENSOR_BYTES,
			 "PULSE head must be a padding-free FP32 array");
BUILD_ASSERT(sizeof(pulse_gradient_t) == PULSE_PROTOCOL_TENSOR_BYTES,
			 "PULSE gradient must be a padding-free FP32 array");
BUILD_ASSERT(PULSE_PROTOCOL_CHUNK_BUFFER_SIZE > PULSE_PROTOCOL_CHUNK_HEADER_SIZE,
			 "PULSE chunk buffer cannot hold a payload");

#define CONTROL_MAGIC_OFFSET 0U
#define CONTROL_PROTOCOL_VERSION_OFFSET 4U
#define CONTROL_FRAME_TYPE_OFFSET 6U
#define CONTROL_DTYPE_OFFSET 7U
#define CONTROL_MODEL_VERSION_OFFSET 8U
#define CONTROL_SESSION_ID_OFFSET 12U
#define CONTROL_HEAD_VERSION_OFFSET 16U
#define CONTROL_TOTAL_SIZE_OFFSET 20U
#define CONTROL_PAYLOAD_CRC_OFFSET 24U
#define CONTROL_STATUS_OFFSET 28U
#define CONTROL_FLAGS_OFFSET 30U
#define CONTROL_RECEIVED_SIZE_OFFSET 32U
#define CONTROL_FRAME_CRC_OFFSET 36U

#define CHUNK_MAGIC_OFFSET 0U
#define CHUNK_PROTOCOL_VERSION_OFFSET 4U
#define CHUNK_FRAME_TYPE_OFFSET 6U
#define CHUNK_DTYPE_OFFSET 7U
#define CHUNK_MODEL_VERSION_OFFSET 8U
#define CHUNK_SESSION_ID_OFFSET 12U
#define CHUNK_HEAD_VERSION_OFFSET 16U
#define CHUNK_TOTAL_SIZE_OFFSET 20U
#define CHUNK_PAYLOAD_OFFSET_OFFSET 24U
#define CHUNK_PAYLOAD_SIZE_OFFSET 28U
#define CHUNK_RESERVED_OFFSET 30U
#define CHUNK_TOTAL_CRC_OFFSET 32U
#define CHUNK_PAYLOAD_CRC_OFFSET 36U
#define CHUNK_HEADER_CRC_OFFSET 40U

struct decoded_control_frame {
	enum pulse_protocol_frame_type frame_type;
	struct pulse_protocol_transfer transfer;
	enum pulse_protocol_remote_status status;
	uint32_t received_size;
};

struct decoded_chunk_frame {
	struct pulse_protocol_transfer transfer;
	uint32_t payload_offset;
	uint16_t payload_size;
	uint32_t chunk_crc32;
};

struct pulse_protocol_server_state {
	bool initialized;
	bool cancel_pending;
	struct pulse_protocol_server_config config;
	enum pulse_protocol_remote_status status;
	struct pulse_protocol_transfer head_transfer;
	struct pulse_protocol_transfer gradient_transfer;
	uint32_t received_size;
	uint32_t received_crc32;
	const uint8_t* serialized_gradient;
	struct bt_conn* conn;
	struct pulse_protocol_counters counters;
};

static struct pulse_protocol_server_state server_state;
K_MUTEX_DEFINE(server_lock);

#if defined(CONFIG_BT_GATT_CLIENT)
static struct bt_uuid_128 pulse_service_uuid = BT_UUID_INIT_128(BT_UUID_PULSE_SERVICE_VAL);
static struct bt_uuid_128 pulse_control_uuid = BT_UUID_INIT_128(BT_UUID_PULSE_CONTROL_VAL);
static struct bt_uuid_128 pulse_head_rx_uuid = BT_UUID_INIT_128(BT_UUID_PULSE_HEAD_RX_VAL);
static struct bt_uuid_128 pulse_gradient_tx_uuid = BT_UUID_INIT_128(BT_UUID_PULSE_GRADIENT_TX_VAL);
#endif

static uint32_t saturating_add_u32(uint32_t value, size_t increment) {
	if (increment > UINT32_MAX || value > UINT32_MAX - (uint32_t) increment) {
		return UINT32_MAX;
	}

	return value + (uint32_t) increment;
}

/* Reflected CRC-32/IEEE (polynomial 0xedb88320, initial/final XOR 0xffffffff). */
static uint32_t protocol_crc32_update(uint32_t crc, const uint8_t* data, size_t length) {
	static const uint32_t nibble_table[16] = {
		0x00000000U,
		0x1db71064U,
		0x3b6e20c8U,
		0x26d930acU,
		0x76dc4190U,
		0x6b6b51f4U,
		0x4db26158U,
		0x5005713cU,
		0xedb88320U,
		0xf00f9344U,
		0xd6d6a3e8U,
		0xcb61b38cU,
		0x9b64c2b0U,
		0x86d3d2d4U,
		0xa00ae278U,
		0xbdbdf21cU,
	};

	crc = ~crc;
	while (length > 0U) {
		crc ^= *data++;
		crc = (crc >> 4) ^ nibble_table[crc & 0x0fU];
		crc = (crc >> 4) ^ nibble_table[crc & 0x0fU];
		--length;
	}

	return ~crc;
}

static uint32_t protocol_crc32(const uint8_t* data, size_t length) {
	return protocol_crc32_update(0U, data, length);
}

static uint32_t float_array_crc32(const float* values, size_t count) {
	uint32_t crc = 0U;
	size_t index;

	if (values == NULL) {
		return 0U;
	}

	for (index = 0U; index < count; ++index) {
		uint32_t bits;
		uint8_t encoded[sizeof(float)];

		memcpy(&bits, &values[index], sizeof(bits));
		sys_put_le32(bits, encoded);
		crc = protocol_crc32_update(crc, encoded, sizeof(encoded));
	}

	return crc;
}

static int float_array_serialize(const float* values,
								 uint8_t* serialized,
								 size_t count,
								 uint32_t* payload_crc32) {
	uint32_t crc = 0U;

	if (values == NULL || serialized == NULL || payload_crc32 == NULL) {
		return -EINVAL;
	}

	for (size_t index = 0U; index < count; ++index) {
		uint32_t bits;
		uint8_t* encoded = serialized + index * sizeof(float);

		memcpy(&bits, &values[index], sizeof(bits));
		sys_put_le32(bits, encoded);
		crc = protocol_crc32_update(crc, encoded, sizeof(float));
	}
	*payload_crc32 = crc;
	return 0;
}

static int float_array_deserialize(const uint8_t* serialized, size_t count, float* values) {
	if (serialized == NULL || values == NULL) {
		return -EINVAL;
	}

	for (size_t index = 0U; index < count; ++index) {
		uint32_t bits = sys_get_le32(serialized + index * sizeof(float));

		memcpy(&values[index], &bits, sizeof(bits));
	}
	return 0;
}

static bool transfer_matches(const struct pulse_protocol_transfer* first,
							 const struct pulse_protocol_transfer* second) {
	return first->model_version == second->model_version &&
		   first->session_id == second->session_id && first->head_version == second->head_version &&
		   first->total_size == second->total_size &&
		   first->payload_crc32 == second->payload_crc32 && first->dtype == second->dtype;
}

static bool transfer_identity_matches(const struct pulse_protocol_transfer* first,
									  const struct pulse_protocol_transfer* second) {
	return first->model_version == second->model_version &&
		   first->session_id == second->session_id && first->head_version == second->head_version &&
		   first->total_size == second->total_size && first->dtype == second->dtype;
}

static bool transfer_shape_valid(const struct pulse_protocol_transfer* transfer,
								 uint32_t model_version) {
	return transfer->model_version == model_version &&
		   transfer->dtype == PULSE_PROTOCOL_DTYPE_FLOAT32_LE &&
		   transfer->total_size == PULSE_PROTOCOL_TENSOR_BYTES;
}

static void encode_control_frame(uint8_t frame[PULSE_PROTOCOL_CONTROL_FRAME_SIZE],
								 enum pulse_protocol_frame_type frame_type,
								 const struct pulse_protocol_transfer* transfer,
								 enum pulse_protocol_remote_status status,
								 uint32_t received_size) {
	memset(frame, 0, PULSE_PROTOCOL_CONTROL_FRAME_SIZE);
	sys_put_le32(PULSE_PROTOCOL_MAGIC, frame + CONTROL_MAGIC_OFFSET);
	sys_put_le16(PULSE_PROTOCOL_VERSION, frame + CONTROL_PROTOCOL_VERSION_OFFSET);
	frame[CONTROL_FRAME_TYPE_OFFSET] = (uint8_t) frame_type;
	frame[CONTROL_DTYPE_OFFSET] = transfer->dtype;
	sys_put_le32(transfer->model_version, frame + CONTROL_MODEL_VERSION_OFFSET);
	sys_put_le32(transfer->session_id, frame + CONTROL_SESSION_ID_OFFSET);
	sys_put_le32(transfer->head_version, frame + CONTROL_HEAD_VERSION_OFFSET);
	sys_put_le32(transfer->total_size, frame + CONTROL_TOTAL_SIZE_OFFSET);
	sys_put_le32(transfer->payload_crc32, frame + CONTROL_PAYLOAD_CRC_OFFSET);
	sys_put_le16((uint16_t) status, frame + CONTROL_STATUS_OFFSET);
	sys_put_le16(0U, frame + CONTROL_FLAGS_OFFSET);
	sys_put_le32(received_size, frame + CONTROL_RECEIVED_SIZE_OFFSET);
	sys_put_le32(protocol_crc32(frame, CONTROL_FRAME_CRC_OFFSET), frame + CONTROL_FRAME_CRC_OFFSET);
}

static int decode_control_frame(const uint8_t frame[PULSE_PROTOCOL_CONTROL_FRAME_SIZE],
								struct decoded_control_frame* decoded) {
	uint32_t expected_crc;

	if (frame == NULL || decoded == NULL) {
		return -EINVAL;
	}

	if (sys_get_le32(frame + CONTROL_MAGIC_OFFSET) != PULSE_PROTOCOL_MAGIC ||
		sys_get_le16(frame + CONTROL_PROTOCOL_VERSION_OFFSET) != PULSE_PROTOCOL_VERSION ||
		sys_get_le16(frame + CONTROL_FLAGS_OFFSET) != 0U) {
		return -EPROTO;
	}

	expected_crc = protocol_crc32(frame, CONTROL_FRAME_CRC_OFFSET);
	if (sys_get_le32(frame + CONTROL_FRAME_CRC_OFFSET) != expected_crc) {
		return -EBADMSG;
	}

	decoded->frame_type = (enum pulse_protocol_frame_type) frame[CONTROL_FRAME_TYPE_OFFSET];
	decoded->transfer.dtype = frame[CONTROL_DTYPE_OFFSET];
	decoded->transfer.model_version = sys_get_le32(frame + CONTROL_MODEL_VERSION_OFFSET);
	decoded->transfer.session_id = sys_get_le32(frame + CONTROL_SESSION_ID_OFFSET);
	decoded->transfer.head_version = sys_get_le32(frame + CONTROL_HEAD_VERSION_OFFSET);
	decoded->transfer.total_size = sys_get_le32(frame + CONTROL_TOTAL_SIZE_OFFSET);
	decoded->transfer.payload_crc32 = sys_get_le32(frame + CONTROL_PAYLOAD_CRC_OFFSET);
	decoded->status =
		(enum pulse_protocol_remote_status) sys_get_le16(frame + CONTROL_STATUS_OFFSET);
	decoded->received_size = sys_get_le32(frame + CONTROL_RECEIVED_SIZE_OFFSET);
	return 0;
}

static void encode_chunk_header(uint8_t frame[PULSE_PROTOCOL_CHUNK_BUFFER_SIZE],
								const struct pulse_protocol_transfer* transfer,
								uint32_t payload_offset,
								uint16_t payload_size,
								uint32_t payload_crc32) {
	memset(frame, 0, PULSE_PROTOCOL_CHUNK_HEADER_SIZE);
	sys_put_le32(PULSE_PROTOCOL_MAGIC, frame + CHUNK_MAGIC_OFFSET);
	sys_put_le16(PULSE_PROTOCOL_VERSION, frame + CHUNK_PROTOCOL_VERSION_OFFSET);
	frame[CHUNK_FRAME_TYPE_OFFSET] = PULSE_PROTOCOL_FRAME_HEAD_CHUNK;
	frame[CHUNK_DTYPE_OFFSET] = transfer->dtype;
	sys_put_le32(transfer->model_version, frame + CHUNK_MODEL_VERSION_OFFSET);
	sys_put_le32(transfer->session_id, frame + CHUNK_SESSION_ID_OFFSET);
	sys_put_le32(transfer->head_version, frame + CHUNK_HEAD_VERSION_OFFSET);
	sys_put_le32(transfer->total_size, frame + CHUNK_TOTAL_SIZE_OFFSET);
	sys_put_le32(payload_offset, frame + CHUNK_PAYLOAD_OFFSET_OFFSET);
	sys_put_le16(payload_size, frame + CHUNK_PAYLOAD_SIZE_OFFSET);
	sys_put_le16(0U, frame + CHUNK_RESERVED_OFFSET);
	sys_put_le32(transfer->payload_crc32, frame + CHUNK_TOTAL_CRC_OFFSET);
	sys_put_le32(payload_crc32, frame + CHUNK_PAYLOAD_CRC_OFFSET);
	sys_put_le32(protocol_crc32(frame, CHUNK_HEADER_CRC_OFFSET), frame + CHUNK_HEADER_CRC_OFFSET);
}

static int decode_chunk_frame(const uint8_t* frame,
							  size_t length,
							  struct decoded_chunk_frame* decoded) {
	uint32_t expected_header_crc;

	if (frame == NULL || decoded == NULL || length < PULSE_PROTOCOL_CHUNK_HEADER_SIZE) {
		return -EINVAL;
	}

	if (sys_get_le32(frame + CHUNK_MAGIC_OFFSET) != PULSE_PROTOCOL_MAGIC ||
		sys_get_le16(frame + CHUNK_PROTOCOL_VERSION_OFFSET) != PULSE_PROTOCOL_VERSION ||
		frame[CHUNK_FRAME_TYPE_OFFSET] != PULSE_PROTOCOL_FRAME_HEAD_CHUNK ||
		sys_get_le16(frame + CHUNK_RESERVED_OFFSET) != 0U) {
		return -EPROTO;
	}

	expected_header_crc = protocol_crc32(frame, CHUNK_HEADER_CRC_OFFSET);
	if (sys_get_le32(frame + CHUNK_HEADER_CRC_OFFSET) != expected_header_crc) {
		return -EBADMSG;
	}

	decoded->transfer.dtype = frame[CHUNK_DTYPE_OFFSET];
	decoded->transfer.model_version = sys_get_le32(frame + CHUNK_MODEL_VERSION_OFFSET);
	decoded->transfer.session_id = sys_get_le32(frame + CHUNK_SESSION_ID_OFFSET);
	decoded->transfer.head_version = sys_get_le32(frame + CHUNK_HEAD_VERSION_OFFSET);
	decoded->transfer.total_size = sys_get_le32(frame + CHUNK_TOTAL_SIZE_OFFSET);
	decoded->transfer.payload_crc32 = sys_get_le32(frame + CHUNK_TOTAL_CRC_OFFSET);
	decoded->payload_offset = sys_get_le32(frame + CHUNK_PAYLOAD_OFFSET_OFFSET);
	decoded->payload_size = sys_get_le16(frame + CHUNK_PAYLOAD_SIZE_OFFSET);
	decoded->chunk_crc32 = sys_get_le32(frame + CHUNK_PAYLOAD_CRC_OFFSET);

	if (decoded->payload_size == 0U ||
		length != PULSE_PROTOCOL_CHUNK_HEADER_SIZE + decoded->payload_size) {
		return -EMSGSIZE;
	}

	return 0;
}

static void server_counter_rx_locked(size_t att_bytes, size_t application_bytes) {
	server_state.counters.att_value_rx_bytes =
		saturating_add_u32(server_state.counters.att_value_rx_bytes, att_bytes);
	server_state.counters.application_rx_bytes =
		saturating_add_u32(server_state.counters.application_rx_bytes, application_bytes);
	server_state.counters.rx_chunks = saturating_add_u32(server_state.counters.rx_chunks, 1U);
}

static void server_counter_tx_locked(size_t att_bytes, size_t application_bytes) {
	server_state.counters.att_value_tx_bytes =
		saturating_add_u32(server_state.counters.att_value_tx_bytes, att_bytes);
	server_state.counters.application_tx_bytes =
		saturating_add_u32(server_state.counters.application_tx_bytes, application_bytes);
	server_state.counters.tx_chunks = saturating_add_u32(server_state.counters.tx_chunks, 1U);
}

static void server_release_connection_locked(void) {
	if (server_state.conn != NULL) {
		bt_conn_unref(server_state.conn);
		server_state.conn = NULL;
	}
}

static void server_set_error_locked(enum pulse_protocol_remote_status status) {
	server_state.status = status;
	server_state.serialized_gradient = NULL;
}

static struct pulse_protocol_transfer server_visible_transfer_locked(void) {
	struct pulse_protocol_transfer transfer = server_state.head_transfer;

	if (server_state.status == PULSE_PROTOCOL_REMOTE_GRADIENT_READY ||
		server_state.status == PULSE_PROTOCOL_REMOTE_COMPLETE) {
		transfer = server_state.gradient_transfer;
	}

	if (transfer.total_size == 0U) {
		transfer.model_version = server_state.config.model_version;
		transfer.total_size = PULSE_PROTOCOL_TENSOR_BYTES;
		transfer.dtype = PULSE_PROTOCOL_DTYPE_FLOAT32_LE;
	}

	return transfer;
}

static ssize_t read_control(struct bt_conn* conn,
							const struct bt_gatt_attr* attr,
							void* buffer,
							uint16_t length,
							uint16_t offset) {
	uint8_t frame[PULSE_PROTOCOL_CONTROL_FRAME_SIZE];
	struct pulse_protocol_transfer transfer;
	ssize_t result;

	ARG_UNUSED(attr);
	k_mutex_lock(&server_lock, K_FOREVER);
	if (!server_state.initialized) {
		k_mutex_unlock(&server_lock);
		return BT_GATT_ERR(BT_ATT_ERR_UNLIKELY);
	}

	if (server_state.conn != NULL && conn != server_state.conn &&
		server_state.status != PULSE_PROTOCOL_REMOTE_COMPLETE &&
		server_state.status != PULSE_PROTOCOL_REMOTE_CANCELLED) {
		k_mutex_unlock(&server_lock);
		return BT_GATT_ERR(BT_ATT_ERR_AUTHORIZATION);
	}

	transfer = server_visible_transfer_locked();
	encode_control_frame(frame,
						 PULSE_PROTOCOL_FRAME_STATUS,
						 &transfer,
						 server_state.status,
						 server_state.received_size);
	k_mutex_unlock(&server_lock);

	result = bt_gatt_attr_read(conn, attr, buffer, length, offset, frame, sizeof(frame));
	if (result > 0) {
		k_mutex_lock(&server_lock, K_FOREVER);
		server_counter_tx_locked((size_t) result, 0U);
		k_mutex_unlock(&server_lock);
	}

	return result;
}

static ssize_t write_control(struct bt_conn* conn,
							 const struct bt_gatt_attr* attr,
							 const void* buffer,
							 uint16_t length,
							 uint16_t offset,
							 uint8_t flags) {
	struct decoded_control_frame decoded;
	struct pulse_protocol_transfer callback_transfer;
	void (*session_started)(const struct pulse_protocol_transfer*, void*) = NULL;
	void (*head_ready)(const struct pulse_protocol_transfer*, const uint8_t*, void*) = NULL;
	void (*gradient_received)(const struct pulse_protocol_transfer*, void*) = NULL;
	void (*session_complete)(const struct pulse_protocol_transfer*, void*) = NULL;
	void (*result_released)(const struct pulse_protocol_transfer*, void*) = NULL;
	void (*session_cancelled)(uint32_t, void*) = NULL;
	const uint8_t* callback_head = NULL;
	void* callback_user_data = NULL;
	uint32_t cancelled_session = 0U;
	int decode_result;
	uint8_t att_error = 0U;

	ARG_UNUSED(attr);
	ARG_UNUSED(flags);
	if (offset != 0U) {
		return BT_GATT_ERR(BT_ATT_ERR_INVALID_OFFSET);
	}
	if (length != PULSE_PROTOCOL_CONTROL_FRAME_SIZE) {
		return BT_GATT_ERR(BT_ATT_ERR_INVALID_ATTRIBUTE_LEN);
	}

	decode_result = decode_control_frame(buffer, &decoded);
	k_mutex_lock(&server_lock, K_FOREVER);
	if (!server_state.initialized) {
		k_mutex_unlock(&server_lock);
		return BT_GATT_ERR(BT_ATT_ERR_UNLIKELY);
	}
	/* RESULT_RELEASE is a gate-low instrumentation handshake, not measured protocol traffic. */
	if (decode_result != 0 || decoded.frame_type != PULSE_PROTOCOL_FRAME_RESULT_RELEASE) {
		server_counter_rx_locked(length, 0U);
	}
	if (server_state.conn != NULL && server_state.conn != conn &&
		(server_state.status == PULSE_PROTOCOL_REMOTE_RECEIVING_HEAD ||
		 server_state.status == PULSE_PROTOCOL_REMOTE_PROCESSING ||
		 server_state.status == PULSE_PROTOCOL_REMOTE_GRADIENT_READY)) {
		k_mutex_unlock(&server_lock);
		return BT_GATT_ERR(BT_ATT_ERR_AUTHORIZATION);
	}

	if (decode_result != 0) {
		if (server_state.status != PULSE_PROTOCOL_REMOTE_IDLE) {
			server_set_error_locked(decode_result == -EBADMSG
										? PULSE_PROTOCOL_REMOTE_ERROR_CRC
										: PULSE_PROTOCOL_REMOTE_ERROR_PROTOCOL);
		}
		k_mutex_unlock(&server_lock);
		return BT_GATT_ERR(BT_ATT_ERR_VALUE_NOT_ALLOWED);
	}

	if (!transfer_shape_valid(&decoded.transfer, server_state.config.model_version)) {
		if (server_state.status != PULSE_PROTOCOL_REMOTE_IDLE) {
			server_set_error_locked(decoded.transfer.model_version !=
											server_state.config.model_version
										? PULSE_PROTOCOL_REMOTE_ERROR_MODEL
										: PULSE_PROTOCOL_REMOTE_ERROR_SIZE);
		}
		k_mutex_unlock(&server_lock);
		return BT_GATT_ERR(BT_ATT_ERR_VALUE_NOT_ALLOWED);
	}

	switch (decoded.frame_type) {
	case PULSE_PROTOCOL_FRAME_HEAD_BEGIN:
		if (server_state.status != PULSE_PROTOCOL_REMOTE_IDLE) {
			att_error = BT_ATT_ERR_PROCEDURE_IN_PROGRESS;
			break;
		}
		if (server_state.config.callbacks.admit_session != NULL &&
			!server_state.config.callbacks.admit_session(&decoded.transfer,
												 server_state.config.user_data)) {
			att_error = BT_ATT_ERR_WRITE_REQ_REJECTED;
			break;
		}

		/* Exclude any rejected pre-session traffic while retaining this accepted BEGIN. */
		memset(&server_state.counters, 0, sizeof(server_state.counters));
		server_counter_rx_locked(length, 0U);
		server_release_connection_locked();
		server_state.conn = bt_conn_ref(conn);
		server_state.head_transfer = decoded.transfer;
		memset(&server_state.gradient_transfer, 0, sizeof(server_state.gradient_transfer));
		server_state.received_size = 0U;
		server_state.received_crc32 = 0U;
		server_state.serialized_gradient = NULL;
		server_state.status = PULSE_PROTOCOL_REMOTE_RECEIVING_HEAD;
		callback_transfer = server_state.head_transfer;
		session_started = server_state.config.callbacks.session_started;
		callback_user_data = server_state.config.user_data;
		break;

	case PULSE_PROTOCOL_FRAME_HEAD_COMMIT:
		if (server_state.conn != conn ||
			server_state.status != PULSE_PROTOCOL_REMOTE_RECEIVING_HEAD ||
			!transfer_matches(&decoded.transfer, &server_state.head_transfer)) {
			if (server_state.status != PULSE_PROTOCOL_REMOTE_IDLE) {
				server_set_error_locked(PULSE_PROTOCOL_REMOTE_ERROR_ORDER);
			}
			att_error = BT_ATT_ERR_WRITE_REQ_REJECTED;
			break;
		}
		if (server_state.received_size != PULSE_PROTOCOL_TENSOR_BYTES) {
			server_set_error_locked(PULSE_PROTOCOL_REMOTE_ERROR_SIZE);
			att_error = BT_ATT_ERR_INVALID_ATTRIBUTE_LEN;
			break;
		}

		if (server_state.received_crc32 != server_state.head_transfer.payload_crc32) {
			server_set_error_locked(PULSE_PROTOCOL_REMOTE_ERROR_CRC);
			att_error = BT_ATT_ERR_VALUE_NOT_ALLOWED;
			break;
		}

		server_state.status = PULSE_PROTOCOL_REMOTE_PROCESSING;
		callback_transfer = server_state.head_transfer;
		callback_head = server_state.config.serialized_head_receive_buffer;
		head_ready = server_state.config.callbacks.head_ready;
		callback_user_data = server_state.config.user_data;
		break;

	case PULSE_PROTOCOL_FRAME_GRADIENT_ACK:
		if (server_state.conn != conn ||
			server_state.status != PULSE_PROTOCOL_REMOTE_GRADIENT_READY ||
			!transfer_matches(&decoded.transfer, &server_state.gradient_transfer)) {
			if (server_state.status != PULSE_PROTOCOL_REMOTE_IDLE) {
				server_set_error_locked(PULSE_PROTOCOL_REMOTE_ERROR_ORDER);
			}
			att_error = BT_ATT_ERR_WRITE_REQ_REJECTED;
			break;
		}

		server_state.status = PULSE_PROTOCOL_REMOTE_COMPLETE;
		server_state.serialized_gradient = NULL;
		callback_transfer = server_state.gradient_transfer;
		gradient_received = server_state.config.callbacks.gradient_received;
		session_complete = server_state.config.callbacks.session_complete;
		callback_user_data = server_state.config.user_data;
		/* Retain the session until the gate-low RESULT_RELEASE transaction arrives. */
		break;

	case PULSE_PROTOCOL_FRAME_RESULT_RELEASE: {
		const struct pulse_protocol_transfer* active_transfer =
			server_state.status == PULSE_PROTOCOL_REMOTE_GRADIENT_READY ||
					server_state.status == PULSE_PROTOCOL_REMOTE_COMPLETE
				? &server_state.gradient_transfer
				: &server_state.head_transfer;

		if (server_state.conn != conn || server_state.cancel_pending ||
			(server_state.status != PULSE_PROTOCOL_REMOTE_COMPLETE &&
			 server_state.status != PULSE_PROTOCOL_REMOTE_CANCELLED) ||
			!transfer_identity_matches(&decoded.transfer, active_transfer)) {
			att_error = BT_ATT_ERR_WRITE_REQ_REJECTED;
			break;
		}

		callback_transfer = *active_transfer;
		callback_user_data = server_state.config.user_data;
		result_released = server_state.config.callbacks.result_released;
		server_state.serialized_gradient = NULL;
		server_release_connection_locked();
		break;
	}

	case PULSE_PROTOCOL_FRAME_CANCEL:
		if (server_state.conn != conn ||
			decoded.transfer.session_id != server_state.head_transfer.session_id ||
			server_state.status == PULSE_PROTOCOL_REMOTE_IDLE ||
			server_state.status == PULSE_PROTOCOL_REMOTE_COMPLETE ||
			server_state.status == PULSE_PROTOCOL_REMOTE_CANCELLED || server_state.cancel_pending) {
			att_error = BT_ATT_ERR_WRITE_REQ_REJECTED;
			break;
		}

		cancelled_session = server_state.head_transfer.session_id;
		/*
		 * STATUS stays PROCESSING until the application worker has stopped
		 * all responder work and lowered its event gate.
		 */
		server_state.status = PULSE_PROTOCOL_REMOTE_PROCESSING;
		server_state.cancel_pending = true;
		server_state.serialized_gradient = NULL;
		session_cancelled = server_state.config.callbacks.session_cancelled;
		callback_user_data = server_state.config.user_data;
		break;

	default:
		if (server_state.status != PULSE_PROTOCOL_REMOTE_IDLE) {
			server_set_error_locked(PULSE_PROTOCOL_REMOTE_ERROR_PROTOCOL);
		}
		att_error = BT_ATT_ERR_VALUE_NOT_ALLOWED;
		break;
	}

	k_mutex_unlock(&server_lock);
	if (att_error != 0U) {
		return BT_GATT_ERR(att_error);
	}

	if (session_started != NULL) {
		session_started(&callback_transfer, callback_user_data);
	}
	if (head_ready != NULL) {
		head_ready(&callback_transfer, callback_head, callback_user_data);
	}
	if (gradient_received != NULL) {
		gradient_received(&callback_transfer, callback_user_data);
	}
	if (session_complete != NULL) {
		session_complete(&callback_transfer, callback_user_data);
	}
	if (session_cancelled != NULL) {
		session_cancelled(cancelled_session, callback_user_data);
	}
	if (result_released != NULL) {
		result_released(&callback_transfer, callback_user_data);
	}

	return length;
}

static ssize_t write_head_chunk(struct bt_conn* conn,
								const struct bt_gatt_attr* attr,
								const void* buffer,
								uint16_t length,
								uint16_t offset,
								uint8_t flags) {
	struct decoded_chunk_frame decoded = {0};
	const uint8_t* bytes = buffer;
	const uint8_t* payload;
	uint32_t end_offset;
	int result;

	ARG_UNUSED(attr);
	ARG_UNUSED(flags);
	if (offset != 0U) {
		return BT_GATT_ERR(BT_ATT_ERR_INVALID_OFFSET);
	}

	result = decode_chunk_frame(bytes, length, &decoded);
	k_mutex_lock(&server_lock, K_FOREVER);
	if (!server_state.initialized) {
		k_mutex_unlock(&server_lock);
		return BT_GATT_ERR(BT_ATT_ERR_UNLIKELY);
	}
	server_counter_rx_locked(length, 0U);
	if (server_state.conn != NULL && server_state.conn != conn &&
		(server_state.status == PULSE_PROTOCOL_REMOTE_RECEIVING_HEAD ||
		 server_state.status == PULSE_PROTOCOL_REMOTE_PROCESSING ||
		 server_state.status == PULSE_PROTOCOL_REMOTE_GRADIENT_READY)) {
		k_mutex_unlock(&server_lock);
		return BT_GATT_ERR(BT_ATT_ERR_AUTHORIZATION);
	}

	if (result != 0) {
		if (server_state.status != PULSE_PROTOCOL_REMOTE_IDLE) {
			server_set_error_locked(result == -EBADMSG ? PULSE_PROTOCOL_REMOTE_ERROR_CRC
													   : PULSE_PROTOCOL_REMOTE_ERROR_PROTOCOL);
		}
		k_mutex_unlock(&server_lock);
		return BT_GATT_ERR(BT_ATT_ERR_VALUE_NOT_ALLOWED);
	}
	if (server_state.conn != conn || server_state.status != PULSE_PROTOCOL_REMOTE_RECEIVING_HEAD ||
		!transfer_matches(&decoded.transfer, &server_state.head_transfer)) {
		if (server_state.status != PULSE_PROTOCOL_REMOTE_IDLE) {
			server_set_error_locked(PULSE_PROTOCOL_REMOTE_ERROR_ORDER);
		}
		k_mutex_unlock(&server_lock);
		return BT_GATT_ERR(BT_ATT_ERR_WRITE_REQ_REJECTED);
	}
	if ((decoded.payload_offset % sizeof(float)) != 0U ||
		(decoded.payload_size % sizeof(float)) != 0U ||
		decoded.payload_offset != server_state.received_size ||
		decoded.payload_offset > PULSE_PROTOCOL_TENSOR_BYTES) {
		server_set_error_locked(PULSE_PROTOCOL_REMOTE_ERROR_ORDER);
		k_mutex_unlock(&server_lock);
		return BT_GATT_ERR(BT_ATT_ERR_INVALID_OFFSET);
	}

	end_offset = decoded.payload_offset + decoded.payload_size;
	if (end_offset < decoded.payload_offset || end_offset > PULSE_PROTOCOL_TENSOR_BYTES) {
		server_set_error_locked(PULSE_PROTOCOL_REMOTE_ERROR_SIZE);
		k_mutex_unlock(&server_lock);
		return BT_GATT_ERR(BT_ATT_ERR_INVALID_ATTRIBUTE_LEN);
	}

	payload = bytes + PULSE_PROTOCOL_CHUNK_HEADER_SIZE;
	if (protocol_crc32(payload, decoded.payload_size) != decoded.chunk_crc32) {
		server_set_error_locked(PULSE_PROTOCOL_REMOTE_ERROR_CRC);
		k_mutex_unlock(&server_lock);
		return BT_GATT_ERR(BT_ATT_ERR_VALUE_NOT_ALLOWED);
	}

	memcpy(server_state.config.serialized_head_receive_buffer + decoded.payload_offset,
		   payload,
		   decoded.payload_size);
	server_state.received_crc32 =
		protocol_crc32_update(server_state.received_crc32, payload, decoded.payload_size);
	server_state.received_size = end_offset;
	server_state.counters.application_rx_bytes =
		saturating_add_u32(server_state.counters.application_rx_bytes, decoded.payload_size);
	k_mutex_unlock(&server_lock);
	return length;
}

static ssize_t read_gradient(struct bt_conn* conn,
							 const struct bt_gatt_attr* attr,
							 void* buffer,
							 uint16_t length,
							 uint16_t offset) {
	uint8_t header[PULSE_PROTOCOL_CONTROL_FRAME_SIZE];
	uint8_t* output = buffer;
	size_t total_size = PULSE_PROTOCOL_GRADIENT_VALUE_SIZE;
	size_t returned_size;
	size_t copied = 0U;
	size_t application_bytes = 0U;

	ARG_UNUSED(attr);
	k_mutex_lock(&server_lock, K_FOREVER);
	if (!server_state.initialized || server_state.status != PULSE_PROTOCOL_REMOTE_GRADIENT_READY ||
		server_state.serialized_gradient == NULL) {
		k_mutex_unlock(&server_lock);
		return BT_GATT_ERR(BT_ATT_ERR_PROCEDURE_IN_PROGRESS);
	}
	if (server_state.conn != conn) {
		k_mutex_unlock(&server_lock);
		return BT_GATT_ERR(BT_ATT_ERR_AUTHORIZATION);
	}
	if (offset > total_size) {
		k_mutex_unlock(&server_lock);
		return BT_GATT_ERR(BT_ATT_ERR_INVALID_OFFSET);
	}

	returned_size = MIN((size_t) length, total_size - offset);
	encode_control_frame(header,
						 PULSE_PROTOCOL_FRAME_GRADIENT_DATA,
						 &server_state.gradient_transfer,
						 PULSE_PROTOCOL_REMOTE_GRADIENT_READY,
						 PULSE_PROTOCOL_TENSOR_BYTES);

	while (copied < returned_size) {
		size_t absolute_offset = (size_t) offset + copied;

		if (absolute_offset < PULSE_PROTOCOL_CONTROL_FRAME_SIZE) {
			size_t copy_size =
				MIN(PULSE_PROTOCOL_CONTROL_FRAME_SIZE - absolute_offset, returned_size - copied);

			memcpy(output + copied, header + absolute_offset, copy_size);
			copied += copy_size;
		} else {
			size_t tensor_offset = absolute_offset - PULSE_PROTOCOL_CONTROL_FRAME_SIZE;
			size_t copy_size = returned_size - copied;

			memcpy(output + copied, server_state.serialized_gradient + tensor_offset, copy_size);
			application_bytes += copy_size;
			copied += copy_size;
		}
	}

	if (returned_size > 0U) {
		server_counter_tx_locked(returned_size, application_bytes);
	}
	k_mutex_unlock(&server_lock);
	return (ssize_t) returned_size;
}

BT_GATT_SERVICE_DEFINE(pulse_protocol_service,
					   BT_GATT_PRIMARY_SERVICE(BT_UUID_PULSE_SERVICE),
					   BT_GATT_CHARACTERISTIC(BT_UUID_PULSE_CONTROL,
											  BT_GATT_CHRC_READ | BT_GATT_CHRC_WRITE,
											  BT_GATT_PERM_READ | BT_GATT_PERM_WRITE,
											  read_control,
											  write_control,
											  NULL),
					   BT_GATT_CHARACTERISTIC(BT_UUID_PULSE_HEAD_RX,
											  BT_GATT_CHRC_WRITE_WITHOUT_RESP,
											  BT_GATT_PERM_WRITE,
											  NULL,
											  write_head_chunk,
											  NULL),
					   BT_GATT_CHARACTERISTIC(BT_UUID_PULSE_GRADIENT_TX,
											  BT_GATT_CHRC_READ,
											  BT_GATT_PERM_READ,
											  read_gradient,
											  NULL,
											  NULL));

int pulse_protocol_server_init(const struct pulse_protocol_server_config* config) {
	if (config == NULL || config->serialized_head_receive_buffer == NULL ||
		config->serialized_head_receive_buffer_size < PULSE_PROTOCOL_TENSOR_BYTES ||
		config->callbacks.head_ready == NULL) {
		return -EINVAL;
	}

	k_mutex_lock(&server_lock, K_FOREVER);
	server_release_connection_locked();
	memset(&server_state, 0, sizeof(server_state));
	server_state.config = *config;
	server_state.status = PULSE_PROTOCOL_REMOTE_IDLE;
	server_state.initialized = true;
	k_mutex_unlock(&server_lock);
	return 0;
}

int pulse_protocol_server_reset(void) {
	k_mutex_lock(&server_lock, K_FOREVER);
	if (!server_state.initialized) {
		k_mutex_unlock(&server_lock);
		return -EACCES;
	}

	server_release_connection_locked();
	memset(&server_state.head_transfer, 0, sizeof(server_state.head_transfer));
	memset(&server_state.gradient_transfer, 0, sizeof(server_state.gradient_transfer));
	server_state.received_size = 0U;
	server_state.received_crc32 = 0U;
	server_state.serialized_gradient = NULL;
	server_state.cancel_pending = false;
	server_state.status = PULSE_PROTOCOL_REMOTE_IDLE;
	k_mutex_unlock(&server_lock);
	return 0;
}

void pulse_protocol_server_disconnected(struct bt_conn* conn) {
	void (*session_cancelled)(uint32_t, void*) = NULL;
	void* user_data = NULL;
	uint32_t session_id = 0U;

	if (conn == NULL) {
		return;
	}

	k_mutex_lock(&server_lock, K_FOREVER);
	if (server_state.initialized && server_state.conn == conn) {
		session_id = server_state.head_transfer.session_id;
		server_state.status = PULSE_PROTOCOL_REMOTE_CANCELLED;
		server_state.cancel_pending = false;
		server_state.serialized_gradient = NULL;
		session_cancelled = server_state.config.callbacks.session_cancelled;
		user_data = server_state.config.user_data;
		server_release_connection_locked();
	}
	k_mutex_unlock(&server_lock);

	if (session_cancelled != NULL) {
		session_cancelled(session_id, user_data);
	}
}

int pulse_protocol_server_publish_gradient(uint32_t session_id,
										   uint32_t head_version,
										   const uint8_t* serialized_gradient,
										   size_t serialized_size,
										   uint32_t payload_crc32) {
	if (serialized_gradient == NULL || serialized_size != PULSE_PROTOCOL_TENSOR_BYTES) {
		return -EINVAL;
	}

	k_mutex_lock(&server_lock, K_FOREVER);
	if (!server_state.initialized) {
		k_mutex_unlock(&server_lock);
		return -EACCES;
	}
	if (server_state.status != PULSE_PROTOCOL_REMOTE_PROCESSING || server_state.cancel_pending ||
		server_state.head_transfer.session_id != session_id ||
		server_state.head_transfer.head_version != head_version) {
		k_mutex_unlock(&server_lock);
		return -ESTALE;
	}

	server_state.gradient_transfer.model_version = server_state.config.model_version;
	server_state.gradient_transfer.session_id = session_id;
	server_state.gradient_transfer.head_version = head_version;
	server_state.gradient_transfer.total_size = PULSE_PROTOCOL_TENSOR_BYTES;
	server_state.gradient_transfer.payload_crc32 = payload_crc32;
	server_state.gradient_transfer.dtype = PULSE_PROTOCOL_DTYPE_FLOAT32_LE;
	server_state.serialized_gradient = serialized_gradient;
	server_state.status = PULSE_PROTOCOL_REMOTE_GRADIENT_READY;
	k_mutex_unlock(&server_lock);
	return 0;
}

int pulse_protocol_server_complete_cancel(uint32_t session_id) {
	k_mutex_lock(&server_lock, K_FOREVER);
	if (!server_state.initialized) {
		k_mutex_unlock(&server_lock);
		return -EACCES;
	}
	if (!server_state.cancel_pending || server_state.head_transfer.session_id != session_id) {
		k_mutex_unlock(&server_lock);
		return -EALREADY;
	}

	server_state.cancel_pending = false;
	server_state.status = PULSE_PROTOCOL_REMOTE_CANCELLED;
	server_state.serialized_gradient = NULL;
	k_mutex_unlock(&server_lock);
	return 0;
}

int pulse_protocol_server_status_get(struct pulse_protocol_server_snapshot* snapshot) {
	if (snapshot == NULL) {
		return -EINVAL;
	}

	k_mutex_lock(&server_lock, K_FOREVER);
	if (!server_state.initialized) {
		k_mutex_unlock(&server_lock);
		return -EACCES;
	}
	snapshot->status = server_state.status;
	snapshot->transfer = server_visible_transfer_locked();
	snapshot->received_size = server_state.received_size;
	k_mutex_unlock(&server_lock);
	return 0;
}

void pulse_protocol_server_counters_get(struct pulse_protocol_counters* counters) {
	if (counters == NULL) {
		return;
	}

	k_mutex_lock(&server_lock, K_FOREVER);
	*counters = server_state.counters;
	k_mutex_unlock(&server_lock);
}

void pulse_protocol_server_counters_reset(void) {
	k_mutex_lock(&server_lock, K_FOREVER);
	memset(&server_state.counters, 0, sizeof(server_state.counters));
	k_mutex_unlock(&server_lock);
}

uint32_t pulse_protocol_head_crc32(const pulse_head_t* head) {
	return head == NULL ? 0U : float_array_crc32(head->parameters, PULSE_HEAD_PARAMETER_COUNT);
}

uint32_t pulse_protocol_gradient_crc32(const pulse_gradient_t* gradient) {
	return gradient == NULL ? 0U
							: float_array_crc32(gradient->parameters, PULSE_HEAD_PARAMETER_COUNT);
}

int pulse_protocol_head_serialize(const pulse_head_t* head,
								  uint8_t* serialized,
								  size_t serialized_size,
								  uint32_t* payload_crc32) {
	if (head == NULL || serialized_size != PULSE_PROTOCOL_TENSOR_BYTES) {
		return -EINVAL;
	}
	return float_array_serialize(head->parameters,
								 serialized,
								 PULSE_HEAD_PARAMETER_COUNT,
								 payload_crc32);
}

int pulse_protocol_head_deserialize(const uint8_t* serialized,
									size_t serialized_size,
									pulse_head_t* head) {
	if (head == NULL || serialized_size != PULSE_PROTOCOL_TENSOR_BYTES) {
		return -EINVAL;
	}
	return float_array_deserialize(serialized, PULSE_HEAD_PARAMETER_COUNT, head->parameters);
}

int pulse_protocol_gradient_serialize(const pulse_gradient_t* gradient,
									  uint8_t* serialized,
									  size_t serialized_size,
									  uint32_t* payload_crc32) {
	if (gradient == NULL || serialized_size != PULSE_PROTOCOL_TENSOR_BYTES) {
		return -EINVAL;
	}
	return float_array_serialize(gradient->parameters,
								 serialized,
								 PULSE_HEAD_PARAMETER_COUNT,
								 payload_crc32);
}

int pulse_protocol_gradient_deserialize(const uint8_t* serialized,
										size_t serialized_size,
										pulse_gradient_t* gradient) {
	if (gradient == NULL || serialized_size != PULSE_PROTOCOL_TENSOR_BYTES) {
		return -EINVAL;
	}
	return float_array_deserialize(serialized, PULSE_HEAD_PARAMETER_COUNT, gradient->parameters);
}

struct client_notification {
	pulse_protocol_client_event_fn callback;
	enum pulse_protocol_client_event event;
	int error;
	enum pulse_protocol_remote_status remote_status;
	struct pulse_protocol_transfer transfer;
	void* user_data;
};

static void client_notification_prepare_locked(struct pulse_protocol_client* client,
											   enum pulse_protocol_client_event event,
											   int error,
											   struct client_notification* notification) {
	if (notification == NULL || client->config.event == NULL) {
		return;
	}

	notification->callback = client->config.event;
	notification->event = event;
	notification->error = error;
	notification->remote_status = client->remote_status;
	notification->transfer = client->transfer;
	notification->user_data = client->config.user_data;
}

static void client_notification_dispatch(struct pulse_protocol_client* client,
										 const struct client_notification* notification) {
	if (notification->callback != NULL) {
		notification->callback(client,
							   notification->event,
							   notification->error,
							   notification->remote_status,
							   &notification->transfer,
							   notification->user_data);
	}
}

#if defined(CONFIG_BT_GATT_CLIENT)

static bool client_has_discovered_handles(const struct pulse_protocol_client* client) {
	return client->control_handle != 0U && client->head_rx_handle != 0U &&
		   client->gradient_tx_handle != 0U;
}

static void client_fail_locked(struct pulse_protocol_client* client,
							   int error,
							   uint8_t att_error,
							   enum pulse_protocol_remote_status remote_status,
							   struct client_notification* notification) {
	client->last_error = error;
	client->last_att_error = att_error;
	client->remote_status = remote_status;
	client->state = PULSE_PROTOCOL_CLIENT_ERROR;
	client->serialized_head_tx = NULL;
	client->serialized_gradient_rx = NULL;
	client_notification_prepare_locked(client,
									   PULSE_PROTOCOL_CLIENT_EVENT_ERROR,
									   error,
									   notification);
}

static int remote_status_error(enum pulse_protocol_remote_status status) {
	switch (status) {
	case PULSE_PROTOCOL_REMOTE_ERROR_MODEL:
		return -EPROTONOSUPPORT;
	case PULSE_PROTOCOL_REMOTE_ERROR_SIZE:
		return -EMSGSIZE;
	case PULSE_PROTOCOL_REMOTE_ERROR_CRC:
		return -EBADMSG;
	case PULSE_PROTOCOL_REMOTE_ERROR_BUSY:
		return -EBUSY;
	case PULSE_PROTOCOL_REMOTE_ERROR_ORDER:
	case PULSE_PROTOCOL_REMOTE_ERROR_PROTOCOL:
		return -EPROTO;
	case PULSE_PROTOCOL_REMOTE_ERROR_INTERNAL:
		return -EIO;
	default:
		return 0;
	}
}
static int client_start_control_write(struct pulse_protocol_client* client,
									  enum pulse_protocol_frame_type frame_type);
static int client_queue_next_head_chunk(struct pulse_protocol_client* client);

static uint8_t client_discovery_callback(struct bt_conn* conn,
										 const struct bt_gatt_attr* attr,
										 struct bt_gatt_discover_params* params) {
	struct pulse_protocol_client* client =
		CONTAINER_OF(params, struct pulse_protocol_client, discover_params);
	struct client_notification notification = {0};
	uint8_t iteration = BT_GATT_ITER_STOP;
	int result;

	k_mutex_lock(&client->lock, K_FOREVER);
	if (client->conn != conn) {
		goto done;
	}

	if (client->state == PULSE_PROTOCOL_CLIENT_DISCOVERING_SERVICE) {
		const struct bt_gatt_service_val* service;

		if (attr == NULL || attr->user_data == NULL) {
			client_fail_locked(client, -ENOENT, 0U, PULSE_PROTOCOL_REMOTE_IDLE, &notification);
			goto done;
		}

		service = attr->user_data;
		client->service_start_handle = attr->handle + 1U;
		client->service_end_handle = service->end_handle;
		memset(params, 0, sizeof(*params));
		params->uuid = NULL;
		params->func = client_discovery_callback;
		params->start_handle = client->service_start_handle;
		params->end_handle = client->service_end_handle;
		params->type = BT_GATT_DISCOVER_CHARACTERISTIC;
		client->state = PULSE_PROTOCOL_CLIENT_DISCOVERING_CHARACTERISTICS;
		result = bt_gatt_discover(conn, params);
		if (result != 0) {
			client_fail_locked(client, result, 0U, PULSE_PROTOCOL_REMOTE_IDLE, &notification);
		}
		goto done;
	}

	if (client->state != PULSE_PROTOCOL_CLIENT_DISCOVERING_CHARACTERISTICS) {
		goto done;
	}

	if (attr == NULL) {
		if (!client_has_discovered_handles(client)) {
			client_fail_locked(client, -ENOENT, 0U, PULSE_PROTOCOL_REMOTE_IDLE, &notification);
		} else {
			client->state = PULSE_PROTOCOL_CLIENT_READY;
			client_notification_prepare_locked(client,
											   PULSE_PROTOCOL_CLIENT_EVENT_DISCOVERY_COMPLETE,
											   0,
											   &notification);
		}
		goto done;
	}

	if (attr->user_data != NULL) {
		const struct bt_gatt_chrc* characteristic = attr->user_data;

		if (bt_uuid_cmp(characteristic->uuid, &pulse_control_uuid.uuid) == 0) {
			client->control_handle = characteristic->value_handle;
		} else if (bt_uuid_cmp(characteristic->uuid, &pulse_head_rx_uuid.uuid) == 0) {
			client->head_rx_handle = characteristic->value_handle;
		} else if (bt_uuid_cmp(characteristic->uuid, &pulse_gradient_tx_uuid.uuid) == 0) {
			client->gradient_tx_handle = characteristic->value_handle;
		}
	}
	iteration = BT_GATT_ITER_CONTINUE;

done:
	k_mutex_unlock(&client->lock);
	client_notification_dispatch(client, &notification);
	return iteration;
}

static void client_head_chunk_sent(struct bt_conn* conn, void* user_data) {
	struct pulse_protocol_client* client = user_data;
	struct client_notification notification = {0};
	int result;

	if (client == NULL) {
		return;
	}

	k_mutex_lock(&client->lock, K_FOREVER);
	if (client->conn != conn || client->state != PULSE_PROTOCOL_CLIENT_SENDING_HEAD) {
		k_mutex_unlock(&client->lock);
		return;
	}

	client->transfer_offset += client->current_chunk_payload_size;
	client->current_chunk_payload_size = 0U;
	result = client_queue_next_head_chunk(client);
	if (result != 0) {
		client_fail_locked(client, result, 0U, PULSE_PROTOCOL_REMOTE_ERROR_INTERNAL, &notification);
	}
	k_mutex_unlock(&client->lock);
	client_notification_dispatch(client, &notification);
}

static void client_control_written(struct bt_conn* conn,
								   uint8_t error,
								   struct bt_gatt_write_params* params) {
	struct pulse_protocol_client* client =
		CONTAINER_OF(params, struct pulse_protocol_client, write_params);
	struct client_notification notification = {0};
	int result;

	k_mutex_lock(&client->lock, K_FOREVER);
	if (client->conn != conn) {
		goto done;
	}
	if (error != 0U) {
		client_fail_locked(client, -EIO, error, client->remote_status, &notification);
		goto done;
	}

	switch (client->state) {
	case PULSE_PROTOCOL_CLIENT_WRITING_BEGIN:
		client->remote_session_started = true;
		client->state = PULSE_PROTOCOL_CLIENT_SENDING_HEAD;
		client->transfer_offset = 0U;
		result = client_queue_next_head_chunk(client);
		if (result != 0) {
			client_fail_locked(client,
							   result,
							   0U,
							   PULSE_PROTOCOL_REMOTE_ERROR_INTERNAL,
							   &notification);
		}
		break;

	case PULSE_PROTOCOL_CLIENT_WRITING_COMMIT:
		client->serialized_head_tx = NULL;
		client->state = PULSE_PROTOCOL_CLIENT_HEAD_SENT;
		client->remote_status = PULSE_PROTOCOL_REMOTE_PROCESSING;
		client_notification_prepare_locked(client,
										   PULSE_PROTOCOL_CLIENT_EVENT_HEAD_SENT,
										   0,
										   &notification);
		break;

	case PULSE_PROTOCOL_CLIENT_WRITING_ACK:
		client->serialized_gradient_rx = NULL;
		client->state = PULSE_PROTOCOL_CLIENT_COMPLETE;
		client->remote_status = PULSE_PROTOCOL_REMOTE_COMPLETE;
		client_notification_prepare_locked(client,
										   PULSE_PROTOCOL_CLIENT_EVENT_COMPLETE,
										   0,
										   &notification);
		break;

	case PULSE_PROTOCOL_CLIENT_WRITING_RESULT_RELEASE:
		client->state = client->result_release_return_state == PULSE_PROTOCOL_CLIENT_COMPLETE
							? PULSE_PROTOCOL_CLIENT_COMPLETE
							: PULSE_PROTOCOL_CLIENT_READY;
		client_notification_prepare_locked(client,
										   PULSE_PROTOCOL_CLIENT_EVENT_RESULT_RELEASED,
										   0,
										   &notification);
		break;

	case PULSE_PROTOCOL_CLIENT_WRITING_CANCEL:
		client->serialized_head_tx = NULL;
		client->serialized_gradient_rx = NULL;
		client->state = PULSE_PROTOCOL_CLIENT_REMOTE_PROCESSING;
		client->remote_status = PULSE_PROTOCOL_REMOTE_PROCESSING;
		client_notification_prepare_locked(client,
										   PULSE_PROTOCOL_CLIENT_EVENT_REMOTE_PROCESSING,
										   0,
										   &notification);
		break;

	default:
		client_fail_locked(client, -EPROTO, 0U, PULSE_PROTOCOL_REMOTE_ERROR_ORDER, &notification);
		break;
	}

done:
	k_mutex_unlock(&client->lock);
	client_notification_dispatch(client, &notification);
}

static int client_start_control_write(struct pulse_protocol_client* client,
									  enum pulse_protocol_frame_type frame_type) {
	int result;

	encode_control_frame(client->control_buffer,
						 frame_type,
						 &client->transfer,
						 PULSE_PROTOCOL_REMOTE_IDLE,
						 0U);
	memset(&client->write_params, 0, sizeof(client->write_params));
	client->write_params.func = client_control_written;
	client->write_params.handle = client->control_handle;
	client->write_params.offset = 0U;
	client->write_params.data = client->control_buffer;
	client->write_params.length = PULSE_PROTOCOL_CONTROL_FRAME_SIZE;
	result = bt_gatt_write(client->conn, &client->write_params);
	if (result == 0 && frame_type != PULSE_PROTOCOL_FRAME_RESULT_RELEASE) {
		client->counters.att_value_tx_bytes =
			saturating_add_u32(client->counters.att_value_tx_bytes,
							   PULSE_PROTOCOL_CONTROL_FRAME_SIZE);
		client->counters.tx_chunks = saturating_add_u32(client->counters.tx_chunks, 1U);
	}

	return result;
}

static int client_queue_next_head_chunk(struct pulse_protocol_client* client) {
	size_t remaining;
	size_t att_value_capacity;
	size_t payload_capacity;
	size_t payload_size;
	uint16_t att_mtu;
	uint8_t* payload;
	uint32_t payload_crc;
	int result;

	if (client->transfer_offset >= PULSE_PROTOCOL_TENSOR_BYTES) {
		client->state = PULSE_PROTOCOL_CLIENT_WRITING_COMMIT;
		return client_start_control_write(client, PULSE_PROTOCOL_FRAME_HEAD_COMMIT);
	}

	att_mtu = bt_gatt_get_mtu(client->conn);
	if (att_mtu < PULSE_PROTOCOL_MIN_ATT_MTU) {
		return -EMSGSIZE;
	}
	att_value_capacity = MIN((size_t) att_mtu - 3U, (size_t) PULSE_PROTOCOL_CHUNK_BUFFER_SIZE);
	payload_capacity = (att_value_capacity - PULSE_PROTOCOL_CHUNK_HEADER_SIZE) &
					   ~(sizeof(float) - 1U);
	if (payload_capacity == 0U) {
		return -EMSGSIZE;
	}

	remaining = PULSE_PROTOCOL_TENSOR_BYTES - client->transfer_offset;
	payload_size = MIN(remaining, payload_capacity);
	payload = client->chunk_buffer + PULSE_PROTOCOL_CHUNK_HEADER_SIZE;
	memcpy(payload, client->serialized_head_tx + client->transfer_offset, payload_size);
	payload_crc = protocol_crc32(payload, payload_size);
	encode_chunk_header(client->chunk_buffer,
						&client->transfer,
						client->transfer_offset,
						(uint16_t) payload_size,
						payload_crc);
	client->current_chunk_payload_size = (uint16_t) payload_size;

	result = bt_gatt_write_without_response_cb(client->conn,
											   client->head_rx_handle,
											   client->chunk_buffer,
											   (uint16_t) (PULSE_PROTOCOL_CHUNK_HEADER_SIZE +
														   payload_size),
											   false,
											   client_head_chunk_sent,
											   client);
	if (result == 0) {
		client->counters.att_value_tx_bytes =
			saturating_add_u32(client->counters.att_value_tx_bytes,
							   PULSE_PROTOCOL_CHUNK_HEADER_SIZE + payload_size);
		client->counters.application_tx_bytes =
			saturating_add_u32(client->counters.application_tx_bytes, payload_size);
		client->counters.tx_chunks = saturating_add_u32(client->counters.tx_chunks, 1U);
	}

	return result;
}

static uint8_t client_status_read(struct bt_conn* conn,
								  uint8_t error,
								  struct bt_gatt_read_params* params,
								  const void* data,
								  uint16_t length) {
	struct pulse_protocol_client* client =
		CONTAINER_OF(params, struct pulse_protocol_client, read_params);
	struct client_notification notification = {0};
	struct decoded_control_frame decoded;
	uint8_t iteration = BT_GATT_ITER_STOP;
	int status_error;
	int result;

	k_mutex_lock(&client->lock, K_FOREVER);
	if (client->conn != conn || client->state != PULSE_PROTOCOL_CLIENT_READING_STATUS) {
		goto done;
	}
	if (error != 0U) {
		client_fail_locked(client, -EIO, error, client->remote_status, &notification);
		goto done;
	}

	if (data != NULL) {
		if (client->control_received + length > PULSE_PROTOCOL_CONTROL_FRAME_SIZE) {
			client_fail_locked(client,
							   -EMSGSIZE,
							   0U,
							   PULSE_PROTOCOL_REMOTE_ERROR_SIZE,
							   &notification);
			goto done;
		}

		memcpy(client->control_buffer + client->control_received, data, length);
		client->control_received += length;
		client->counters.att_value_rx_bytes =
			saturating_add_u32(client->counters.att_value_rx_bytes, length);
		client->counters.rx_chunks = saturating_add_u32(client->counters.rx_chunks, 1U);
		iteration = BT_GATT_ITER_CONTINUE;
		goto done;
	}

	if (client->control_received != PULSE_PROTOCOL_CONTROL_FRAME_SIZE) {
		client_fail_locked(client, -EMSGSIZE, 0U, PULSE_PROTOCOL_REMOTE_ERROR_SIZE, &notification);
		goto done;
	}
	result = decode_control_frame(client->control_buffer, &decoded);
	if (result != 0 || decoded.frame_type != PULSE_PROTOCOL_FRAME_STATUS) {
		client_fail_locked(client,
						   result != 0 ? result : -EPROTO,
						   0U,
						   PULSE_PROTOCOL_REMOTE_ERROR_PROTOCOL,
						   &notification);
		goto done;
	}

	client->remote_status = decoded.status;
	status_error = remote_status_error(decoded.status);
	if (status_error != 0) {
		client_fail_locked(client, status_error, 0U, decoded.status, &notification);
		goto done;
	}
	if (decoded.transfer.model_version != client->config.model_version ||
		decoded.transfer.session_id != client->transfer.session_id ||
		decoded.transfer.head_version != client->transfer.head_version) {
		client_fail_locked(client, -EPROTO, 0U, PULSE_PROTOCOL_REMOTE_ERROR_ORDER, &notification);
		goto done;
	}

	switch (decoded.status) {
	case PULSE_PROTOCOL_REMOTE_RECEIVING_HEAD:
	case PULSE_PROTOCOL_REMOTE_PROCESSING:
		client->state = PULSE_PROTOCOL_CLIENT_REMOTE_PROCESSING;
		client_notification_prepare_locked(client,
										   PULSE_PROTOCOL_CLIENT_EVENT_REMOTE_PROCESSING,
										   0,
										   &notification);
		break;

	case PULSE_PROTOCOL_REMOTE_GRADIENT_READY:
		if (!transfer_shape_valid(&decoded.transfer, client->config.model_version)) {
			client_fail_locked(client,
							   -EMSGSIZE,
							   0U,
							   PULSE_PROTOCOL_REMOTE_ERROR_SIZE,
							   &notification);
			break;
		}
		client->advertised_gradient_crc32 = decoded.transfer.payload_crc32;
		client->state = PULSE_PROTOCOL_CLIENT_GRADIENT_READY;
		client_notification_prepare_locked(client,
										   PULSE_PROTOCOL_CLIENT_EVENT_GRADIENT_READY,
										   0,
										   &notification);
		break;

	case PULSE_PROTOCOL_REMOTE_COMPLETE:
		client->state = PULSE_PROTOCOL_CLIENT_COMPLETE;
		client_notification_prepare_locked(client,
										   PULSE_PROTOCOL_CLIENT_EVENT_COMPLETE,
										   0,
										   &notification);
		break;

	case PULSE_PROTOCOL_REMOTE_CANCELLED:
		client->state = PULSE_PROTOCOL_CLIENT_READY;
		client_notification_prepare_locked(client,
										   PULSE_PROTOCOL_CLIENT_EVENT_CANCELLED,
										   0,
										   &notification);
		break;

	default:
		client_fail_locked(client,
						   -EPROTO,
						   0U,
						   PULSE_PROTOCOL_REMOTE_ERROR_PROTOCOL,
						   &notification);
		break;
	}

done:
	k_mutex_unlock(&client->lock);
	client_notification_dispatch(client, &notification);
	return iteration;
}

static uint8_t client_gradient_read(struct bt_conn* conn,
									uint8_t error,
									struct bt_gatt_read_params* params,
									const void* data,
									uint16_t length) {
	struct pulse_protocol_client* client =
		CONTAINER_OF(params, struct pulse_protocol_client, read_params);
	struct client_notification notification = {0};
	struct decoded_control_frame decoded;
	size_t stream_offset;
	size_t copied = 0U;
	uint8_t iteration = BT_GATT_ITER_STOP;
	int result;

	k_mutex_lock(&client->lock, K_FOREVER);
	if (client->conn != conn || client->state != PULSE_PROTOCOL_CLIENT_READING_GRADIENT) {
		goto done;
	}
	if (error != 0U) {
		client_fail_locked(client, -EIO, error, client->remote_status, &notification);
		goto done;
	}

	if (data != NULL) {
		if (client->gradient_stream_received + length > PULSE_PROTOCOL_GRADIENT_VALUE_SIZE) {
			client_fail_locked(client,
							   -EMSGSIZE,
							   0U,
							   PULSE_PROTOCOL_REMOTE_ERROR_SIZE,
							   &notification);
			goto done;
		}

		stream_offset = client->gradient_stream_received;
		while (copied < length) {
			if (stream_offset < PULSE_PROTOCOL_CONTROL_FRAME_SIZE) {
				size_t copy_size = MIN(PULSE_PROTOCOL_CONTROL_FRAME_SIZE - stream_offset,
									   (size_t) length - copied);

				memcpy(client->gradient_header + stream_offset,
					   (const uint8_t*) data + copied,
					   copy_size);
				stream_offset += copy_size;
				copied += copy_size;
			} else {
				size_t payload_offset = stream_offset - PULSE_PROTOCOL_CONTROL_FRAME_SIZE;
				size_t copy_size = (size_t) length - copied;

				memcpy(client->serialized_gradient_rx + payload_offset,
					   (const uint8_t*) data + copied,
					   copy_size);
				client->gradient_received_crc32 =
					protocol_crc32_update(client->gradient_received_crc32,
										  (const uint8_t*) data + copied,
										  copy_size);
				client->gradient_payload_received += copy_size;
				client->counters.application_rx_bytes =
					saturating_add_u32(client->counters.application_rx_bytes, copy_size);
				stream_offset += copy_size;
				copied += copy_size;
			}
		}

		client->gradient_stream_received += length;
		client->counters.att_value_rx_bytes =
			saturating_add_u32(client->counters.att_value_rx_bytes, length);
		client->counters.rx_chunks = saturating_add_u32(client->counters.rx_chunks, 1U);
		iteration = BT_GATT_ITER_CONTINUE;
		goto done;
	}

	if (client->gradient_stream_received != PULSE_PROTOCOL_GRADIENT_VALUE_SIZE ||
		client->gradient_payload_received != PULSE_PROTOCOL_TENSOR_BYTES) {
		client_fail_locked(client, -EMSGSIZE, 0U, PULSE_PROTOCOL_REMOTE_ERROR_SIZE, &notification);
		goto done;
	}
	result = decode_control_frame(client->gradient_header, &decoded);
	if (result != 0 || decoded.frame_type != PULSE_PROTOCOL_FRAME_GRADIENT_DATA ||
		decoded.status != PULSE_PROTOCOL_REMOTE_GRADIENT_READY ||
		!transfer_shape_valid(&decoded.transfer, client->config.model_version) ||
		decoded.transfer.session_id != client->transfer.session_id ||
		decoded.transfer.head_version != client->transfer.head_version ||
		decoded.transfer.payload_crc32 != client->advertised_gradient_crc32) {
		client_fail_locked(client,
						   result != 0 ? result : -EPROTO,
						   0U,
						   PULSE_PROTOCOL_REMOTE_ERROR_PROTOCOL,
						   &notification);
		goto done;
	}

	if (client->gradient_received_crc32 != decoded.transfer.payload_crc32) {
		client_fail_locked(client, -EBADMSG, 0U, PULSE_PROTOCOL_REMOTE_ERROR_CRC, &notification);
		goto done;
	}

	client->transfer.payload_crc32 = decoded.transfer.payload_crc32;
	client->serialized_gradient_rx = NULL;
	client->state = PULSE_PROTOCOL_CLIENT_GRADIENT_RECEIVED;
	client_notification_prepare_locked(client,
									   PULSE_PROTOCOL_CLIENT_EVENT_GRADIENT_RECEIVED,
									   0,
									   &notification);

done:
	k_mutex_unlock(&client->lock);
	client_notification_dispatch(client, &notification);
	return iteration;
}

#endif /* CONFIG_BT_GATT_CLIENT */

int pulse_protocol_client_init(struct pulse_protocol_client* client,
							   const struct pulse_protocol_client_config* config) {
	int result;

	if (client == NULL || config == NULL) {
		return -EINVAL;
	}

	memset(client, 0, sizeof(*client));
	result = k_mutex_init(&client->lock);
	if (result != 0) {
		return result;
	}
	client->config = *config;
	client->remote_status = PULSE_PROTOCOL_REMOTE_IDLE;
	/* Publish initialization only after the embedded lock is ready. */
	client->state = PULSE_PROTOCOL_CLIENT_IDLE;
	return 0;
}

int pulse_protocol_client_discover(struct pulse_protocol_client* client, struct bt_conn* conn) {
#if defined(CONFIG_BT_GATT_CLIENT)
	int result;

	if (client == NULL || conn == NULL || client->state == PULSE_PROTOCOL_CLIENT_UNINITIALIZED) {
		return -EINVAL;
	}
	k_mutex_lock(&client->lock, K_FOREVER);
	if (client->state != PULSE_PROTOCOL_CLIENT_IDLE || client->conn != NULL) {
		result = -EBUSY;
		goto done;
	}

	client->conn = bt_conn_ref(conn);
	client->remote_status = PULSE_PROTOCOL_REMOTE_IDLE;
	client->last_error = 0;
	client->last_att_error = 0U;
	client->service_start_handle = 0U;
	client->service_end_handle = 0U;
	client->control_handle = 0U;
	client->head_rx_handle = 0U;
	client->gradient_tx_handle = 0U;
	client->remote_session_started = false;
	memset(&client->discover_params, 0, sizeof(client->discover_params));
	client->discover_params.uuid = &pulse_service_uuid.uuid;
	client->discover_params.func = client_discovery_callback;
	client->discover_params.start_handle = BT_ATT_FIRST_ATTRIBUTE_HANDLE;
	client->discover_params.end_handle = BT_ATT_LAST_ATTRIBUTE_HANDLE;
	client->discover_params.type = BT_GATT_DISCOVER_PRIMARY;
	client->state = PULSE_PROTOCOL_CLIENT_DISCOVERING_SERVICE;
	result = bt_gatt_discover(conn, &client->discover_params);
	if (result != 0) {
		client->state = PULSE_PROTOCOL_CLIENT_IDLE;
		bt_conn_unref(client->conn);
		client->conn = NULL;
	}

done:
	k_mutex_unlock(&client->lock);
	return result;
#else
	ARG_UNUSED(client);
	ARG_UNUSED(conn);
	return -ENOTSUP;
#endif
}

int pulse_protocol_client_send_head(struct pulse_protocol_client* client,
									uint32_t session_id,
									uint32_t head_version,
									const uint8_t* serialized_head,
									size_t serialized_size,
									uint32_t payload_crc32) {
#if defined(CONFIG_BT_GATT_CLIENT)
	int result;

	if (client == NULL || serialized_head == NULL ||
		serialized_size != PULSE_PROTOCOL_TENSOR_BYTES) {
		return -EINVAL;
	}
	k_mutex_lock(&client->lock, K_FOREVER);
	if (client->conn == NULL || !client_has_discovered_handles(client)) {
		result = -ENOTCONN;
		goto done;
	}
	if (client->state != PULSE_PROTOCOL_CLIENT_READY &&
		client->state != PULSE_PROTOCOL_CLIENT_COMPLETE) {
		result = -EBUSY;
		goto done;
	}
	if (bt_gatt_get_mtu(client->conn) < PULSE_PROTOCOL_MIN_ATT_MTU) {
		result = -EMSGSIZE;
		goto done;
	}

	client->transfer.model_version = client->config.model_version;
	client->transfer.session_id = session_id;
	client->transfer.head_version = head_version;
	client->transfer.total_size = PULSE_PROTOCOL_TENSOR_BYTES;
	client->transfer.payload_crc32 = payload_crc32;
	client->transfer.dtype = PULSE_PROTOCOL_DTYPE_FLOAT32_LE;
	client->serialized_head_tx = serialized_head;
	client->serialized_gradient_rx = NULL;
	client->transfer_offset = 0U;
	client->remote_status = PULSE_PROTOCOL_REMOTE_IDLE;
	client->remote_session_started = false;
	client->state = PULSE_PROTOCOL_CLIENT_WRITING_BEGIN;
	result = client_start_control_write(client, PULSE_PROTOCOL_FRAME_HEAD_BEGIN);
	if (result != 0) {
		client->serialized_head_tx = NULL;
		client->state = PULSE_PROTOCOL_CLIENT_READY;
	}

done:
	k_mutex_unlock(&client->lock);
	return result;
#else
	ARG_UNUSED(client);
	ARG_UNUSED(session_id);
	ARG_UNUSED(head_version);
	ARG_UNUSED(serialized_head);
	ARG_UNUSED(serialized_size);
	ARG_UNUSED(payload_crc32);
	return -ENOTSUP;
#endif
}

int pulse_protocol_client_poll_status(struct pulse_protocol_client* client) {
#if defined(CONFIG_BT_GATT_CLIENT)
	enum pulse_protocol_client_state previous_state;
	int result;

	if (client == NULL) {
		return -EINVAL;
	}
	k_mutex_lock(&client->lock, K_FOREVER);
	if (client->conn == NULL) {
		result = -ENOTCONN;
		goto done;
	}
	if (client->state != PULSE_PROTOCOL_CLIENT_HEAD_SENT &&
		client->state != PULSE_PROTOCOL_CLIENT_REMOTE_PROCESSING) {
		result = -EBUSY;
		goto done;
	}
	if (client->control_handle == 0U) {
		result = -ENOENT;
		goto done;
	}

	previous_state = client->state;
	client->control_received = 0U;
	memset(&client->read_params, 0, sizeof(client->read_params));
	client->read_params.func = client_status_read;
	client->read_params.handle_count = 1U;
	client->read_params.single.handle = client->control_handle;
	client->read_params.single.offset = 0U;
	client->state = PULSE_PROTOCOL_CLIENT_READING_STATUS;
	result = bt_gatt_read(client->conn, &client->read_params);
	if (result != 0) {
		client->state = previous_state;
	}

done:
	k_mutex_unlock(&client->lock);
	return result;
#else
	ARG_UNUSED(client);
	return -ENOTSUP;
#endif
}

int pulse_protocol_client_receive_gradient(struct pulse_protocol_client* client,
										   uint8_t* serialized_destination,
										   size_t serialized_size) {
#if defined(CONFIG_BT_GATT_CLIENT)
	int result;

	if (client == NULL || serialized_destination == NULL ||
		serialized_size != PULSE_PROTOCOL_TENSOR_BYTES) {
		return -EINVAL;
	}
	k_mutex_lock(&client->lock, K_FOREVER);
	if (client->conn == NULL) {
		result = -ENOTCONN;
		goto done;
	}
	if (client->state != PULSE_PROTOCOL_CLIENT_GRADIENT_READY) {
		result = -EBUSY;
		goto done;
	}

	client->serialized_gradient_rx = serialized_destination;
	client->gradient_stream_received = 0U;
	client->gradient_payload_received = 0U;
	client->gradient_received_crc32 = 0U;
	memset(client->gradient_header, 0, sizeof(client->gradient_header));
	memset(&client->read_params, 0, sizeof(client->read_params));
	client->read_params.func = client_gradient_read;
	client->read_params.handle_count = 1U;
	client->read_params.single.handle = client->gradient_tx_handle;
	client->read_params.single.offset = 0U;
	client->state = PULSE_PROTOCOL_CLIENT_READING_GRADIENT;
	result = bt_gatt_read(client->conn, &client->read_params);
	if (result != 0) {
		client->serialized_gradient_rx = NULL;
		client->state = PULSE_PROTOCOL_CLIENT_GRADIENT_READY;
	}

done:
	k_mutex_unlock(&client->lock);
	return result;
#else
	ARG_UNUSED(client);
	ARG_UNUSED(serialized_destination);
	ARG_UNUSED(serialized_size);
	return -ENOTSUP;
#endif
}

int pulse_protocol_client_acknowledge(struct pulse_protocol_client* client) {
#if defined(CONFIG_BT_GATT_CLIENT)
	int result;

	if (client == NULL) {
		return -EINVAL;
	}
	k_mutex_lock(&client->lock, K_FOREVER);
	if (client->conn == NULL) {
		result = -ENOTCONN;
		goto done;
	}
	if (client->state != PULSE_PROTOCOL_CLIENT_GRADIENT_RECEIVED) {
		result = -EBUSY;
		goto done;
	}

	client->state = PULSE_PROTOCOL_CLIENT_WRITING_ACK;
	result = client_start_control_write(client, PULSE_PROTOCOL_FRAME_GRADIENT_ACK);
	if (result != 0) {
		client->state = PULSE_PROTOCOL_CLIENT_GRADIENT_RECEIVED;
	}

done:
	k_mutex_unlock(&client->lock);
	return result;
#else
	ARG_UNUSED(client);
	return -ENOTSUP;
#endif
}

int pulse_protocol_client_release_result(struct pulse_protocol_client* client) {
#if defined(CONFIG_BT_GATT_CLIENT)
	int result;

	if (client == NULL) {
		return -EINVAL;
	}
	k_mutex_lock(&client->lock, K_FOREVER);
	if (client->conn == NULL) {
		result = -ENOTCONN;
		goto done;
	}
	if (client->state != PULSE_PROTOCOL_CLIENT_HEAD_SENT &&
		client->state != PULSE_PROTOCOL_CLIENT_REMOTE_PROCESSING &&
		client->state != PULSE_PROTOCOL_CLIENT_GRADIENT_READY &&
		client->state != PULSE_PROTOCOL_CLIENT_GRADIENT_RECEIVED &&
		client->state != PULSE_PROTOCOL_CLIENT_COMPLETE &&
		client->state != PULSE_PROTOCOL_CLIENT_READY &&
		client->state != PULSE_PROTOCOL_CLIENT_ERROR) {
		result = -EBUSY;
		goto done;
	}
	if (client->control_handle == 0U || !client->remote_session_started) {
		result = -ENOENT;
		goto done;
	}

	client->result_release_return_state = client->state;
	client->state = PULSE_PROTOCOL_CLIENT_WRITING_RESULT_RELEASE;
	result = client_start_control_write(client, PULSE_PROTOCOL_FRAME_RESULT_RELEASE);
	if (result != 0) {
		client->state = client->result_release_return_state;
	}

done:
	k_mutex_unlock(&client->lock);
	return result;
#else
	ARG_UNUSED(client);
	return -ENOTSUP;
#endif
}

int pulse_protocol_client_cancel(struct pulse_protocol_client* client) {
#if defined(CONFIG_BT_GATT_CLIENT)
	enum pulse_protocol_client_state previous_state;
	int result;

	if (client == NULL) {
		return -EINVAL;
	}
	k_mutex_lock(&client->lock, K_FOREVER);
	if (client->conn == NULL) {
		result = -ENOTCONN;
		goto done;
	}
	if (client->state != PULSE_PROTOCOL_CLIENT_HEAD_SENT &&
		client->state != PULSE_PROTOCOL_CLIENT_REMOTE_PROCESSING &&
		client->state != PULSE_PROTOCOL_CLIENT_GRADIENT_READY &&
		client->state != PULSE_PROTOCOL_CLIENT_GRADIENT_RECEIVED &&
		client->state != PULSE_PROTOCOL_CLIENT_ERROR) {
		result = -EBUSY;
		goto done;
	}
	if (client->control_handle == 0U) {
		result = -ENOENT;
		goto done;
	}

	previous_state = client->state;
	client->state = PULSE_PROTOCOL_CLIENT_WRITING_CANCEL;
	result = client_start_control_write(client, PULSE_PROTOCOL_FRAME_CANCEL);
	if (result != 0) {
		client->state = previous_state;
	}

done:
	k_mutex_unlock(&client->lock);
	return result;
#else
	ARG_UNUSED(client);
	return -ENOTSUP;
#endif
}

void pulse_protocol_client_disconnected(struct pulse_protocol_client* client,
										struct bt_conn* conn) {
	struct client_notification notification = {0};

	if (client == NULL || conn == NULL || client->state == PULSE_PROTOCOL_CLIENT_UNINITIALIZED) {
		return;
	}

	k_mutex_lock(&client->lock, K_FOREVER);
	if (client->conn != conn) {
		k_mutex_unlock(&client->lock);
		return;
	}
	bt_conn_unref(client->conn);
	client->conn = NULL;
	client->serialized_head_tx = NULL;
	client->serialized_gradient_rx = NULL;
	client->control_handle = 0U;
	client->head_rx_handle = 0U;
	client->gradient_tx_handle = 0U;
	client->last_error = -ENOTCONN;
	client->state = PULSE_PROTOCOL_CLIENT_IDLE;
	client_notification_prepare_locked(client,
									   PULSE_PROTOCOL_CLIENT_EVENT_DISCONNECTED,
									   -ENOTCONN,
									   &notification);
	k_mutex_unlock(&client->lock);
	client_notification_dispatch(client, &notification);
}

int pulse_protocol_client_reset(struct pulse_protocol_client* client) {
	if (client == NULL || client->state == PULSE_PROTOCOL_CLIENT_UNINITIALIZED) {
		return -EINVAL;
	}
	k_mutex_lock(&client->lock, K_FOREVER);
	if (client->state != PULSE_PROTOCOL_CLIENT_IDLE &&
		client->state != PULSE_PROTOCOL_CLIENT_READY &&
		client->state != PULSE_PROTOCOL_CLIENT_COMPLETE &&
		client->state != PULSE_PROTOCOL_CLIENT_ERROR) {
		k_mutex_unlock(&client->lock);
		return -EBUSY;
	}

	if (client->conn != NULL) {
		bt_conn_unref(client->conn);
		client->conn = NULL;
	}
	client->serialized_head_tx = NULL;
	client->serialized_gradient_rx = NULL;
	client->control_handle = 0U;
	client->head_rx_handle = 0U;
	client->gradient_tx_handle = 0U;
	client->service_start_handle = 0U;
	client->service_end_handle = 0U;
	client->last_error = 0;
	client->last_att_error = 0U;
	client->remote_status = PULSE_PROTOCOL_REMOTE_IDLE;
	client->remote_session_started = false;
	client->state = PULSE_PROTOCOL_CLIENT_IDLE;
	k_mutex_unlock(&client->lock);
	return 0;
}

enum pulse_protocol_client_state pulse_protocol_client_state_get(
	struct pulse_protocol_client* client) {
	enum pulse_protocol_client_state state;

	if (client == NULL || client->state == PULSE_PROTOCOL_CLIENT_UNINITIALIZED) {
		return PULSE_PROTOCOL_CLIENT_UNINITIALIZED;
	}
	k_mutex_lock(&client->lock, K_FOREVER);
	state = client->state;
	k_mutex_unlock(&client->lock);
	return state;
}

void pulse_protocol_client_counters_get(struct pulse_protocol_client* client,
										struct pulse_protocol_counters* counters) {
	if (counters == NULL) {
		return;
	}
	if (client == NULL || client->state == PULSE_PROTOCOL_CLIENT_UNINITIALIZED) {
		memset(counters, 0, sizeof(*counters));
		return;
	}
	k_mutex_lock(&client->lock, K_FOREVER);
	*counters = client->counters;
	k_mutex_unlock(&client->lock);
}

void pulse_protocol_client_counters_reset(struct pulse_protocol_client* client) {
	if (client == NULL || client->state == PULSE_PROTOCOL_CLIENT_UNINITIALIZED) {
		return;
	}
	k_mutex_lock(&client->lock, K_FOREVER);
	memset(&client->counters, 0, sizeof(client->counters));
	client->remote_session_started = false;
	k_mutex_unlock(&client->lock);
}

bool pulse_protocol_client_remote_session_started_get(struct pulse_protocol_client* client) {
	bool started;

	if (client == NULL || client->state == PULSE_PROTOCOL_CLIENT_UNINITIALIZED) {
		return false;
	}
	k_mutex_lock(&client->lock, K_FOREVER);
	started = client->remote_session_started;
	k_mutex_unlock(&client->lock);
	return started;
}
