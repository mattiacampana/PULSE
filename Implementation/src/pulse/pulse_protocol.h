#ifndef SENSWEAR_PULSE_PROTOCOL_H_
#define SENSWEAR_PULSE_PROTOCOL_H_

#include "pulse_math.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <zephyr/bluetooth/conn.h>
#include <zephyr/bluetooth/gatt.h>
#include <zephyr/bluetooth/uuid.h>
#include <zephyr/kernel.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * PULSE benchmark service UUIDs. These values are a public wire interface; do not reuse them
 * for an incompatible layout.
 */
#define BT_UUID_PULSE_SERVICE_VAL \
	BT_UUID_128_ENCODE(0x7f3c1000, 0x5b2e, 0x4db0, 0xa6f1, 0x50554c534500)
#define BT_UUID_PULSE_CONTROL_VAL \
	BT_UUID_128_ENCODE(0x7f3c1001, 0x5b2e, 0x4db0, 0xa6f1, 0x50554c534500)
#define BT_UUID_PULSE_HEAD_RX_VAL \
	BT_UUID_128_ENCODE(0x7f3c1002, 0x5b2e, 0x4db0, 0xa6f1, 0x50554c534500)
#define BT_UUID_PULSE_GRADIENT_TX_VAL \
	BT_UUID_128_ENCODE(0x7f3c1003, 0x5b2e, 0x4db0, 0xa6f1, 0x50554c534500)

#define BT_UUID_PULSE_SERVICE BT_UUID_DECLARE_128(BT_UUID_PULSE_SERVICE_VAL)
#define BT_UUID_PULSE_CONTROL BT_UUID_DECLARE_128(BT_UUID_PULSE_CONTROL_VAL)
#define BT_UUID_PULSE_HEAD_RX BT_UUID_DECLARE_128(BT_UUID_PULSE_HEAD_RX_VAL)
#define BT_UUID_PULSE_GRADIENT_TX BT_UUID_DECLARE_128(BT_UUID_PULSE_GRADIENT_TX_VAL)

#define PULSE_PROTOCOL_MAGIC 0x534c5550U /* "PULS" as little-endian bytes. */
#define PULSE_PROTOCOL_VERSION 1U
#define PULSE_PROTOCOL_CONTROL_FRAME_SIZE 40U
#define PULSE_PROTOCOL_CHUNK_HEADER_SIZE 44U
#define PULSE_PROTOCOL_TENSOR_BYTES (PULSE_HEAD_PARAMETER_COUNT * sizeof(float))
#define PULSE_PROTOCOL_GRADIENT_VALUE_SIZE \
	(PULSE_PROTOCOL_CONTROL_FRAME_SIZE + PULSE_PROTOCOL_TENSOR_BYTES)

/* Scratch capacity for negotiated ATT values; the release configuration currently negotiates 247.
 */
#define PULSE_PROTOCOL_CHUNK_BUFFER_SIZE 512U
#define PULSE_PROTOCOL_MIN_ATT_MTU (3U + PULSE_PROTOCOL_CHUNK_HEADER_SIZE + sizeof(float))

/*
 * Every multi-byte field is little-endian. A control frame is exactly 40 bytes:
 *
 *  0 magic:u32          4 protocol_version:u16  6 frame_type:u8  7 dtype:u8
 *  8 model_version:u32 12 session_id:u32       16 head_version:u32
 * 20 total_size:u32    24 payload_crc32:u32    28 status:u16     30 flags:u16
 * 32 received_size:u32 36 frame_crc32:u32 (CRC over bytes [0, 36)).
 *
 * HEAD_BEGIN, HEAD_COMMIT, GRADIENT_ACK, RESULT_RELEASE, and CANCEL are written to Control.
 * Reading Control returns a STATUS frame. Reading Gradient TX returns a GRADIENT_DATA control
 * header followed by the serialized gradient. ATT Read Blob offsets provide the chunk offset for
 * that long read. RESULT_RELEASE is an out-of-measurement instrumentation handshake and is
 * deliberately excluded from protocol counters.
 *
 * Every Head RX write-without-response value starts with a 44-byte chunk header:
 *
 *  0 magic:u32          4 protocol_version:u16  6 frame_type:u8  7 dtype:u8
 *  8 model_version:u32 12 session_id:u32       16 head_version:u32
 * 20 total_size:u32    24 payload_offset:u32   28 payload_size:u16 30 reserved:u16
 * 32 total_crc32:u32   36 chunk_crc32:u32      40 header_crc32:u32
 * 44 payload bytes (CRC header covers bytes [0, 40)).
 */

enum pulse_protocol_dtype {
	PULSE_PROTOCOL_DTYPE_FLOAT32_LE = 1,
};

enum pulse_protocol_frame_type {
	PULSE_PROTOCOL_FRAME_HEAD_BEGIN = 1,
	PULSE_PROTOCOL_FRAME_HEAD_COMMIT = 2,
	PULSE_PROTOCOL_FRAME_GRADIENT_ACK = 3,
	PULSE_PROTOCOL_FRAME_CANCEL = 4,
	PULSE_PROTOCOL_FRAME_STATUS = 5,
	PULSE_PROTOCOL_FRAME_HEAD_CHUNK = 6,
	PULSE_PROTOCOL_FRAME_GRADIENT_DATA = 7,
	PULSE_PROTOCOL_FRAME_RESULT_RELEASE = 8,
};

enum pulse_protocol_remote_status {
	PULSE_PROTOCOL_REMOTE_IDLE = 0,
	PULSE_PROTOCOL_REMOTE_RECEIVING_HEAD = 1,
	PULSE_PROTOCOL_REMOTE_PROCESSING = 2,
	PULSE_PROTOCOL_REMOTE_GRADIENT_READY = 3,
	PULSE_PROTOCOL_REMOTE_COMPLETE = 4,
	PULSE_PROTOCOL_REMOTE_CANCELLED = 5,
	PULSE_PROTOCOL_REMOTE_ERROR_PROTOCOL = 0x100,
	PULSE_PROTOCOL_REMOTE_ERROR_MODEL = 0x101,
	PULSE_PROTOCOL_REMOTE_ERROR_SIZE = 0x102,
	PULSE_PROTOCOL_REMOTE_ERROR_CRC = 0x103,
	PULSE_PROTOCOL_REMOTE_ERROR_ORDER = 0x104,
	PULSE_PROTOCOL_REMOTE_ERROR_BUSY = 0x105,
	PULSE_PROTOCOL_REMOTE_ERROR_INTERNAL = 0x106,
};

struct pulse_protocol_transfer {
	uint32_t model_version;
	uint32_t session_id;
	uint32_t head_version;
	uint32_t total_size;
	uint32_t payload_crc32;
	uint8_t dtype;
};

/*
 * Counts application tensor bytes and ATT characteristic-value bytes, not link-layer bytes.
 * Client TX counters count values successfully submitted to the Bluetooth host; server RX
 * counters count values accepted by the GATT callbacks. They are not proof of over-air delivery.
 * Use the external sniffer for authoritative link traffic and retransmissions.
 */
struct pulse_protocol_counters {
	uint32_t application_tx_bytes;
	uint32_t application_rx_bytes;
	uint32_t att_value_tx_bytes;
	uint32_t att_value_rx_bytes;
	uint32_t tx_chunks;
	uint32_t rx_chunks;
};

struct pulse_protocol_server_callbacks {
	/*
	 * Callbacks run synchronously in Bluetooth/application callback context. admit_session runs
	 * while the server lock is held, before HEAD_BEGIN changes protocol state; it must not block,
	 * take application mutexes, or call a pulse_protocol_server_* API. Returning false rejects the
	 * request without accepting a remote session. session_started marks an accepted HEAD_BEGIN;
	 * head_ready marks fully received, CRC-validated serialized head bytes.
	 * The application performs FP32LE decoding outside callback context. Only record
	 * timestamps/GPIO and schedule work here: never block or perform model computation.
	 */
	bool (*admit_session)(const struct pulse_protocol_transfer* transfer, void* user_data);
	void (*session_started)(const struct pulse_protocol_transfer* transfer, void* user_data);
	void (*head_ready)(const struct pulse_protocol_transfer* transfer,
					   const uint8_t* serialized_head,
					   void* user_data);
	/*
	 * gradient_received marks receipt of the peer's CRC-validated GRADIENT_ACK. It therefore runs
	 * after the initiator has received the complete gradient, rather than when the final ATT read
	 * value is merely produced on the responder.
	 */
	void (*gradient_received)(const struct pulse_protocol_transfer* transfer, void* user_data);
	void (*session_complete)(const struct pulse_protocol_transfer* transfer, void* user_data);
	/*
	 * Runs only after the initiator has lowered its event gate and successfully written the
	 * out-of-band RESULT_RELEASE frame. A responder exporter must wait for this callback.
	 */
	void (*result_released)(const struct pulse_protocol_transfer* transfer, void* user_data);
	void (*session_cancelled)(uint32_t session_id, void* user_data);
};

struct pulse_protocol_server_config {
	uint32_t model_version;
	/* Caller-owned wire storage that must remain valid while the singleton server is initialized.
	 */
	uint8_t* serialized_head_receive_buffer;
	size_t serialized_head_receive_buffer_size;
	struct pulse_protocol_server_callbacks callbacks;
	void* user_data;
};

struct pulse_protocol_server_snapshot {
	enum pulse_protocol_remote_status status;
	struct pulse_protocol_transfer transfer;
	uint32_t received_size;
};

int pulse_protocol_server_init(const struct pulse_protocol_server_config* config);
int pulse_protocol_server_reset(void);
/* Call from the application's disconnect callback for the responder role. */
void pulse_protocol_server_disconnected(struct bt_conn* conn);
/*
 * Publishes a pre-serialized FP32LE gradient. The bytes must remain immutable until complete,
 * cancel, disconnect, or server reset; payload_crc32 must describe exactly those bytes.
 */
int pulse_protocol_server_publish_gradient(uint32_t session_id,
										   uint32_t head_version,
										   const uint8_t* serialized_gradient,
										   size_t serialized_size,
										   uint32_t payload_crc32);
/*
 * Completes an application-side cancel only after all measured responder work
 * and its event gate have stopped.  Until then STATUS remains PROCESSING, so a
 * client can prove the responder gate is low before sending RESULT_RELEASE.
 */
int pulse_protocol_server_complete_cancel(uint32_t session_id);
int pulse_protocol_server_status_get(struct pulse_protocol_server_snapshot* snapshot);
void pulse_protocol_server_counters_get(struct pulse_protocol_counters* counters);
void pulse_protocol_server_counters_reset(void);

enum pulse_protocol_client_state {
	PULSE_PROTOCOL_CLIENT_UNINITIALIZED = 0,
	PULSE_PROTOCOL_CLIENT_IDLE,
	PULSE_PROTOCOL_CLIENT_DISCOVERING_SERVICE,
	PULSE_PROTOCOL_CLIENT_DISCOVERING_CHARACTERISTICS,
	PULSE_PROTOCOL_CLIENT_READY,
	PULSE_PROTOCOL_CLIENT_WRITING_BEGIN,
	PULSE_PROTOCOL_CLIENT_SENDING_HEAD,
	PULSE_PROTOCOL_CLIENT_WRITING_COMMIT,
	PULSE_PROTOCOL_CLIENT_HEAD_SENT,
	PULSE_PROTOCOL_CLIENT_READING_STATUS,
	PULSE_PROTOCOL_CLIENT_REMOTE_PROCESSING,
	PULSE_PROTOCOL_CLIENT_GRADIENT_READY,
	PULSE_PROTOCOL_CLIENT_READING_GRADIENT,
	PULSE_PROTOCOL_CLIENT_GRADIENT_RECEIVED,
	PULSE_PROTOCOL_CLIENT_WRITING_ACK,
	PULSE_PROTOCOL_CLIENT_COMPLETE,
	PULSE_PROTOCOL_CLIENT_WRITING_RESULT_RELEASE,
	PULSE_PROTOCOL_CLIENT_WRITING_CANCEL,
	PULSE_PROTOCOL_CLIENT_ERROR,
};

enum pulse_protocol_client_event {
	PULSE_PROTOCOL_CLIENT_EVENT_DISCOVERY_COMPLETE = 0,
	/* Remote accepted the complete head. Start the initiator's local gradient here. */
	PULSE_PROTOCOL_CLIENT_EVENT_HEAD_SENT,
	PULSE_PROTOCOL_CLIENT_EVENT_REMOTE_PROCESSING,
	PULSE_PROTOCOL_CLIENT_EVENT_GRADIENT_READY,
	/* Destination bytes are CRC-validated; acknowledge receipt before application decoding. */
	PULSE_PROTOCOL_CLIENT_EVENT_GRADIENT_RECEIVED,
	PULSE_PROTOCOL_CLIENT_EVENT_COMPLETE,
	PULSE_PROTOCOL_CLIENT_EVENT_RESULT_RELEASED,
	PULSE_PROTOCOL_CLIENT_EVENT_CANCELLED,
	PULSE_PROTOCOL_CLIENT_EVENT_DISCONNECTED,
	PULSE_PROTOCOL_CLIENT_EVENT_ERROR,
};

struct pulse_protocol_client;

/* Client events are synchronous Bluetooth callbacks too; signal application work and return. */
typedef void (*pulse_protocol_client_event_fn)(struct pulse_protocol_client* client,
											   enum pulse_protocol_client_event event,
											   int error,
											   enum pulse_protocol_remote_status remote_status,
											   const struct pulse_protocol_transfer* transfer,
											   void* user_data);
struct pulse_protocol_client_config {
	uint32_t model_version;
	pulse_protocol_client_event_fn event;
	void* user_data;
};

/*
 * Caller-owned, zero-allocation client context. Keep it alive until disconnected/reset. Public
 * members are implementation storage and must only be initialized through the API. The API and
 * Bluetooth callbacks serialize state internally; application event callbacks run after the state
 * lock is released.
 */
struct pulse_protocol_client {
	struct pulse_protocol_client_config config;
	struct k_mutex lock;
	struct bt_conn* conn;
	enum pulse_protocol_client_state state;
	enum pulse_protocol_client_state result_release_return_state;
	enum pulse_protocol_remote_status remote_status;
	struct pulse_protocol_transfer transfer;
	struct pulse_protocol_counters counters;
	struct bt_gatt_discover_params discover_params;
	struct bt_gatt_write_params write_params;
	struct bt_gatt_read_params read_params;
	const uint8_t* serialized_head_tx;
	uint8_t* serialized_gradient_rx;
	uint16_t service_start_handle;
	uint16_t service_end_handle;
	uint16_t control_handle;
	uint16_t head_rx_handle;
	uint16_t gradient_tx_handle;
	uint16_t current_chunk_payload_size;
	uint8_t last_att_error;
	int last_error;
	uint32_t transfer_offset;
	uint32_t control_received;
	uint32_t gradient_stream_received;
	uint32_t gradient_payload_received;
	uint32_t gradient_received_crc32;
	uint32_t advertised_gradient_crc32;
	bool remote_session_started;
	uint8_t control_buffer[PULSE_PROTOCOL_CONTROL_FRAME_SIZE];
	uint8_t gradient_header[PULSE_PROTOCOL_CONTROL_FRAME_SIZE];
	uint8_t chunk_buffer[PULSE_PROTOCOL_CHUNK_BUFFER_SIZE];
};

int pulse_protocol_client_init(struct pulse_protocol_client* client,
							   const struct pulse_protocol_client_config* config);

/*
 * Starts GATT service/characteristic discovery on an already connected peer. Client operations
 * return -ENOTSUP when CONFIG_BT_GATT_CLIENT is disabled.
 */
int pulse_protocol_client_discover(struct pulse_protocol_client* client, struct bt_conn* conn);

/*
 * Asynchronously sends one immutable, pre-serialized 4,550-float FP32LE head. Keep the bytes alive
 * until HEAD_SENT or ERROR. payload_crc32 must describe exactly serialized_head. The remote begins
 * responder processing before HEAD_SENT is reported to the initiator. The negotiated ATT MTU must
 * be at least PULSE_PROTOCOL_MIN_ATT_MTU (51 bytes).
 */
int pulse_protocol_client_send_head(struct pulse_protocol_client* client,
									uint32_t session_id,
									uint32_t head_version,
									const uint8_t* serialized_head,
									size_t serialized_size,
									uint32_t payload_crc32);

/* Poll after local-gradient compute; GRADIENT_READY is reported through the event callback. */
int pulse_protocol_client_poll_status(struct pulse_protocol_client* client);

/*
 * Starts a GATT long read into caller-owned wire storage and reports CRC-validated serialized
 * gradient bytes. The application performs FP32LE decoding after GRADIENT_RECEIVED.
 */
int pulse_protocol_client_receive_gradient(struct pulse_protocol_client* client,
										   uint8_t* serialized_destination,
										   size_t serialized_size);

/* Sends GRADIENT_ACK immediately after CRC-validated receipt, before decoding/agreement/mixing. */
int pulse_protocol_client_acknowledge(struct pulse_protocol_client* client);

/*
 * Out-of-measurement exporter release. Call only after the initiator event gate is low. The
 * server accepts it for a matching completed or application-acknowledged-cancelled session,
 * allowing failure paths to release a responder record too. Its ATT bytes/chunk are excluded from
 * protocol counters on both peers.
 */
int pulse_protocol_client_release_result(struct pulse_protocol_client* client);

/*
 * Valid in stable post-head states. Completion means only that the cancel request was written; the
 * client must poll STATUS until CANCELLED before lowering its gate. It does not cancel a chunk
 * already queued in the controller.
 */
int pulse_protocol_client_cancel(struct pulse_protocol_client* client);

/* Call from the application's disconnect callback for the initiator role. */
void pulse_protocol_client_disconnected(struct pulse_protocol_client* client, struct bt_conn* conn);

/* Releases a retained connection only when no asynchronous operation is active. */
int pulse_protocol_client_reset(struct pulse_protocol_client* client);

enum pulse_protocol_client_state pulse_protocol_client_state_get(
	struct pulse_protocol_client* client);
void pulse_protocol_client_counters_get(struct pulse_protocol_client* client,
										struct pulse_protocol_counters* counters);
void pulse_protocol_client_counters_reset(struct pulse_protocol_client* client);
/* True only after the peer accepted this trial's HEAD_BEGIN write request. */
bool pulse_protocol_client_remote_session_started_get(struct pulse_protocol_client* client);

/* CRC over canonical little-endian FP32 bytes, suitable for fixture generation. */
uint32_t pulse_protocol_head_crc32(const pulse_head_t* head);
uint32_t pulse_protocol_gradient_crc32(const pulse_gradient_t* gradient);

/*
 * Full-tensor conversions used by the benchmark's explicit serialization/deserialization stages.
 * Serialize produces canonical FP32LE bytes and their CRC in one pass. Deserialize converts a
 * previously protocol-validated canonical buffer; source and destination may alias exactly.
 */
int pulse_protocol_head_serialize(const pulse_head_t* head,
								  uint8_t* serialized,
								  size_t serialized_size,
								  uint32_t* payload_crc32);
int pulse_protocol_head_deserialize(const uint8_t* serialized,
									size_t serialized_size,
									pulse_head_t* head);
int pulse_protocol_gradient_serialize(const pulse_gradient_t* gradient,
									  uint8_t* serialized,
									  size_t serialized_size,
									  uint32_t* payload_crc32);
int pulse_protocol_gradient_deserialize(const uint8_t* serialized,
										size_t serialized_size,
										pulse_gradient_t* gradient);

#ifdef __cplusplus
}
#endif

#endif /* SENSWEAR_PULSE_PROTOCOL_H_ */
