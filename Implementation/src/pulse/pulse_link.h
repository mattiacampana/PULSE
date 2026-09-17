#ifndef SENSWEAR_PULSE_LINK_H_
#define SENSWEAR_PULSE_LINK_H_

#include <stdbool.h>
#include <stdint.h>

#include <zephyr/bluetooth/addr.h>
#include <zephyr/bluetooth/conn.h>
#include <zephyr/bluetooth/gatt.h>
#include <zephyr/kernel.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PULSE_LINK_INTERVAL_US 7500U
#define PULSE_LINK_DATA_LENGTH 251U
#define PULSE_LINK_RSSI_UNAVAILABLE (-127)

enum pulse_link_role {
	PULSE_LINK_ROLE_UNRESOLVED = 0,
	PULSE_LINK_ROLE_INITIATOR = 1,
	PULSE_LINK_ROLE_RESPONDER = 2,
};

struct pulse_link;

/*
 * scan_found/connected/disconnected normally run in the Bluetooth receive context. A synchronous
 * setup failure may invoke connected from the pulse_link_start()/pulse_link_reconnect() caller.
 * Do not block, print, run model code, or start PULSE protocol procedures in these callbacks;
 * record timing/GPIO state and wake an application thread instead. Address pointers are only valid
 * for the duration of scan_found. Acquire a durable connection with pulse_link_connected_ref().
 */
struct pulse_link_callbacks {
	void (*scan_found)(const bt_addr_le_t* peer,
					   int8_t advertisement_rssi,
					   enum pulse_link_role role,
					   void* user_data);
	void (*connected)(enum pulse_link_role role, int error, void* user_data);
	void (*disconnected)(enum pulse_link_role role, uint8_t reason, void* user_data);
};

struct pulse_link_config {
	struct pulse_link_callbacks callbacks;
	void* user_data;
};

/*
 * Reconnect timing begins synchronously immediately before radio activity is requested. It ends in
 * Bluetooth callback context after connection parameter, PHY, data-length, and ATT-MTU negotiation
 * has resolved (or failed). Keep both hooks constant-time; they are intended for GPIO/timestamps.
 * The structure is copied by pulse_link_reconnect(), so stack construction is safe.
 */
struct pulse_link_timing_callbacks {
	void (*started)(enum pulse_link_role role, void* user_data);
	void (*finished)(enum pulse_link_role role, int error, void* user_data);
	void* user_data;
};

struct pulse_link_snapshot {
	bool connected;
	enum pulse_link_role role;
	bt_addr_le_t local_address;
	bt_addr_le_t peer_address;
	uint16_t att_mtu;
	uint32_t interval_us;
	uint8_t tx_phy;
	uint8_t rx_phy;
	uint16_t tx_max_len;
	uint16_t tx_max_time_us;
	uint16_t rx_max_len;
	uint16_t rx_max_time_us;
	int8_t rssi_dbm;
	int negotiation_error;
};

/*
 * Caller-owned singleton state. The implementation uses only this fixed storage and Bluetooth's
 * statically configured pools; it never calls a heap allocator. Treat every member as private.
 * Keep the object alive for the remainder of the boot after pulse_link_start().
 */
struct pulse_link {
	struct pulse_link_config config;
	struct pulse_link_timing_callbacks reconnect_timing;
	struct k_mutex lock;
	struct k_sem scan_found_sem;
	struct k_sem connected_sem;
	struct k_sem disconnected_sem;
	struct bt_gatt_exchange_params mtu_exchange;
	struct bt_conn* conn;
	bt_addr_le_t local_address;
	bt_addr_le_t peer_address;
	enum pulse_link_role role;
	int connection_result;
	int negotiation_error;
	uint32_t attempt_generation;
	int8_t advertisement_rssi;
	uint8_t negotiation_stage;
	uint8_t disconnect_reason;
	bool initialized;
	bool peer_valid;
	bool scanning;
	bool advertising;
	bool pool_ready;
	bool pending_create;
	bool pending_advertising;
	bool awaiting_connection;
	bool negotiation_finished;
	bool scan_reported;
	bool probe_only;
	bool reconnect_timing_active;
};

/*
 * Enables Bluetooth synchronously and starts the initial role activity. Kconfig override 1 selects
 * initiator, 2 selects responder, and 0 starts symmetric advertise+passive-scan discovery. In auto
 * mode the numerically/lexicographically lower canonical 48-bit address becomes initiator.
 */
int pulse_link_start(struct pulse_link* link, const struct pulse_link_config* config);

enum pulse_link_role pulse_link_role_get(struct pulse_link* link);

/* Waits for the next accepted PULSE service advertisement. Either output pointer may be NULL. */
int pulse_link_wait_scan_found(struct pulse_link* link,
							   bt_addr_le_t* peer,
							   int8_t* advertisement_rssi,
							   k_timeout_t timeout);

/*
 * Performs one passive scan for the already identified peer without creating
 * a connection. The regular scan_found callback runs before this function
 * returns, which lets the benchmark close an externally marked scan stage
 * without a cross-thread race. Valid only for a disconnected resolved
 * initiator between reconnect operations.
 */
int pulse_link_probe_peer(struct pulse_link* link,
						  bt_addr_le_t* peer,
						  int8_t* advertisement_rssi,
						  k_timeout_t timeout);

/* Returns a new connection reference only after link negotiation has succeeded. */
struct bt_conn* pulse_link_connected_ref(struct pulse_link* link);

/* Returns the connection/negotiation result. A timeout does not cancel background activity. */
int pulse_link_wait_connected(struct pulse_link* link, k_timeout_t timeout);

/*
 * Quiesces the current role activity.  Besides an established link, this also
 * cancels an in-progress scan/advertise/connect attempt so a timed-out measured
 * discovery can be followed by a clean reconnect.
 */
int pulse_link_disconnect(struct pulse_link* link);
int pulse_link_wait_disconnected(struct pulse_link* link, k_timeout_t timeout);

/* Starts role-stable scan/connect or advertising after a completed disconnect. */
int pulse_link_reconnect(struct pulse_link* link, const struct pulse_link_timing_callbacks* timing);

/*
 * Captures live connection values and performs a synchronous HCI Read RSSI using a fixed net_buf
 * pool. Call from an application thread, never a Bluetooth callback. RSSI is -127 when unavailable.
 */
int pulse_link_snapshot_get(struct pulse_link* link, struct pulse_link_snapshot* snapshot);

#ifdef __cplusplus
}
#endif

#endif /* SENSWEAR_PULSE_LINK_H_ */
