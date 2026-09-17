#include "pulse_link.h"

#include "pulse_protocol.h"

#include <errno.h>
#include <string.h>

#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/gap.h>
#include <zephyr/bluetooth/hci.h>
#include <zephyr/bluetooth/hci_types.h>
#include <zephyr/net_buf.h>
#include <zephyr/sys/byteorder.h>
#include <zephyr/sys/util.h>

BUILD_ASSERT(CONFIG_BT_MAX_CONN == 1, "PULSE link manager owns exactly one connection");

#define PULSE_LINK_INTERVAL_UNITS 6U
#define PULSE_LINK_SUPERVISION_TIMEOUT_UNITS 400U

enum pulse_link_negotiation_stage {
	PULSE_LINK_NEGOTIATION_IDLE = 0,
	PULSE_LINK_NEGOTIATION_PARAM,
	PULSE_LINK_NEGOTIATION_PHY,
	PULSE_LINK_NEGOTIATION_DATA_LENGTH,
	PULSE_LINK_NEGOTIATION_MTU,
	PULSE_LINK_NEGOTIATION_PEER_MTU,
	PULSE_LINK_NEGOTIATION_DONE,
};

struct pulse_link_uuid_search {
	bool found;
};

static struct pulse_link* active_link;

static const uint8_t pulse_service_uuid_le[16] = {BT_UUID_PULSE_SERVICE_VAL};
static const struct bt_data pulse_advertising_data[] = {
	BT_DATA_BYTES(BT_DATA_FLAGS, BT_LE_AD_GENERAL | BT_LE_AD_NO_BREDR),
	BT_DATA_BYTES(BT_DATA_UUID128_ALL, BT_UUID_PULSE_SERVICE_VAL),
};
static const struct bt_le_adv_param pulse_advertising_parameters =
	BT_LE_ADV_PARAM_INIT(BT_LE_ADV_OPT_CONN | BT_LE_ADV_OPT_USE_IDENTITY,
						 BT_GAP_ADV_FAST_INT_MIN_1,
						 BT_GAP_ADV_FAST_INT_MAX_1,
						 NULL);
static const struct bt_le_scan_param pulse_scan_parameters =
	BT_LE_SCAN_PARAM_INIT(BT_LE_SCAN_TYPE_PASSIVE,
						  BT_LE_SCAN_OPT_FILTER_DUPLICATE,
						  BT_GAP_SCAN_FAST_INTERVAL,
						  BT_GAP_SCAN_FAST_WINDOW);
static const struct bt_conn_le_create_param pulse_create_parameters =
	BT_CONN_LE_CREATE_PARAM_INIT(BT_CONN_LE_OPT_NONE,
								 BT_GAP_SCAN_FAST_INTERVAL,
								 BT_GAP_SCAN_FAST_INTERVAL);
static const struct bt_le_conn_param pulse_connection_parameters =
	BT_LE_CONN_PARAM_INIT(PULSE_LINK_INTERVAL_UNITS,
						  PULSE_LINK_INTERVAL_UNITS,
						  0U,
						  PULSE_LINK_SUPERVISION_TIMEOUT_UNITS);

static int pulse_link_start_advertising(struct pulse_link* link);
static int pulse_link_start_scanning(struct pulse_link* link);
static int pulse_link_create_connection(struct pulse_link* link, uint32_t attempt_generation);
static void pulse_link_continue_negotiation(struct pulse_link* link);
static void pulse_link_finish_negotiation(struct pulse_link* link);

static uint8_t pulse_link_configured_role(void) {
#if defined(CONFIG_SENSWEAR_PULSE_ROLE_OVERRIDE)
	return CONFIG_SENSWEAR_PULSE_ROLE_OVERRIDE;
#else
	return 0U;
#endif
}

/* Compare the canonical printed/network order; bt_addr_t::val itself is little-endian. */
static int pulse_link_address_compare(const bt_addr_le_t* first, const bt_addr_le_t* second) {
	int index;

	for (index = (int) sizeof(first->a.val) - 1; index >= 0; --index) {
		if (first->a.val[index] < second->a.val[index]) {
			return -1;
		}
		if (first->a.val[index] > second->a.val[index]) {
			return 1;
		}
	}

	return 0;
}

static bool pulse_link_uuid_parse(struct bt_data* data, void* user_data) {
	struct pulse_link_uuid_search* search = user_data;
	size_t offset;

	if (data->type != BT_DATA_UUID128_ALL && data->type != BT_DATA_UUID128_SOME) {
		return true;
	}

	for (offset = 0U; offset + sizeof(pulse_service_uuid_le) <= data->data_len;
		 offset += sizeof(pulse_service_uuid_le)) {
		if (memcmp(data->data + offset, pulse_service_uuid_le, sizeof(pulse_service_uuid_le)) ==
			0) {
			search->found = true;
			return false;
		}
	}

	return true;
}

static bool pulse_link_has_service(struct net_buf_simple* advertising_data) {
	struct pulse_link_uuid_search search = {0};
	struct net_buf_simple copy = *advertising_data;

	bt_data_parse(&copy, pulse_link_uuid_parse, &search);
	return search.found;
}

static void pulse_link_record_negotiation_error(struct pulse_link* link, int error) {
	if (error == 0) {
		return;
	}

	k_mutex_lock(&link->lock, K_FOREVER);
	if (link->negotiation_error == 0) {
		link->negotiation_error = error;
	}
	k_mutex_unlock(&link->lock);
}

static bool pulse_link_move_stage(struct pulse_link* link,
								  enum pulse_link_negotiation_stage expected,
								  enum pulse_link_negotiation_stage next) {
	bool moved = false;

	k_mutex_lock(&link->lock, K_FOREVER);
	if (link->negotiation_stage == expected && link->conn != NULL && !link->negotiation_finished) {
		link->negotiation_stage = next;
		moved = true;
	}
	k_mutex_unlock(&link->lock);
	return moved;
}

static void pulse_link_finish_connection(struct pulse_link* link, int error) {
	struct pulse_link_timing_callbacks timing = {0};
	void (*connected)(enum pulse_link_role, int, void*) = NULL;
	void* user_data = NULL;
	enum pulse_link_role role;

	k_mutex_lock(&link->lock, K_FOREVER);
	if (!link->awaiting_connection || link->negotiation_finished) {
		k_mutex_unlock(&link->lock);
		return;
	}

	link->awaiting_connection = false;
	link->negotiation_finished = true;
	link->negotiation_stage = PULSE_LINK_NEGOTIATION_DONE;
	link->connection_result = error;
	if (error != 0 && link->negotiation_error == 0) {
		link->negotiation_error = error;
	}
	role = link->role;
	connected = link->config.callbacks.connected;
	user_data = link->config.user_data;
	if (link->reconnect_timing_active) {
		timing = link->reconnect_timing;
		link->reconnect_timing_active = false;
	}
	k_sem_give(&link->connected_sem);
	k_mutex_unlock(&link->lock);

	/* Finish the measurement gate before any general application callback can export results. */
	if (timing.finished != NULL) {
		timing.finished(role, error, timing.user_data);
	}
	if (connected != NULL) {
		connected(role, error, user_data);
	}
}

static int pulse_link_basic_snapshot(struct bt_conn* conn, struct pulse_link_snapshot* snapshot) {
	struct bt_conn_info info;
	int result;

	result = bt_conn_get_info(conn, &info);
	if (result != 0) {
		return result;
	}
	if (info.type != BT_CONN_TYPE_LE) {
		return -EPROTOTYPE;
	}

	snapshot->connected = info.state == BT_CONN_STATE_CONNECTED;
	snapshot->interval_us = info.le.interval_us;
	snapshot->att_mtu = bt_gatt_get_mtu(conn);
#if defined(CONFIG_BT_USER_PHY_UPDATE)
	if (info.le.phy != NULL) {
		snapshot->tx_phy = info.le.phy->tx_phy;
		snapshot->rx_phy = info.le.phy->rx_phy;
	}
#endif
#if defined(CONFIG_BT_USER_DATA_LEN_UPDATE)
	if (info.le.data_len != NULL) {
		snapshot->tx_max_len = info.le.data_len->tx_max_len;
		snapshot->tx_max_time_us = info.le.data_len->tx_max_time;
		snapshot->rx_max_len = info.le.data_len->rx_max_len;
		snapshot->rx_max_time_us = info.le.data_len->rx_max_time;
	}
#endif
	return 0;
}

static int pulse_link_verify_negotiation(struct pulse_link* link, struct bt_conn* conn) {
	struct pulse_link_snapshot snapshot = {0};
	int result;

	k_mutex_lock(&link->lock, K_FOREVER);
	result = link->negotiation_error;
	k_mutex_unlock(&link->lock);

	if (pulse_link_basic_snapshot(conn, &snapshot) != 0) {
		return result != 0 ? result : -EIO;
	}
	if (result == 0 && snapshot.interval_us != PULSE_LINK_INTERVAL_US) {
		result = -ERANGE;
	}
	if (result == 0 &&
		(snapshot.tx_phy != BT_GAP_LE_PHY_2M || snapshot.rx_phy != BT_GAP_LE_PHY_2M)) {
		result = -ENOTSUP;
	}
	if (result == 0 && (snapshot.tx_max_len < PULSE_LINK_DATA_LENGTH ||
						snapshot.rx_max_len < PULSE_LINK_DATA_LENGTH)) {
		result = -EMSGSIZE;
	}
	if (result == 0 && snapshot.att_mtu < PULSE_PROTOCOL_MIN_ATT_MTU) {
		result = -EMSGSIZE;
	}

	return result;
}

static void pulse_link_finish_negotiation(struct pulse_link* link) {
	struct bt_conn* conn = NULL;
	int result;

	k_mutex_lock(&link->lock, K_FOREVER);
	if (link->conn != NULL && !link->negotiation_finished) {
		conn = bt_conn_ref(link->conn);
	}
	k_mutex_unlock(&link->lock);
	if (conn == NULL) {
		return;
	}

	result = pulse_link_verify_negotiation(link, conn);
	bt_conn_unref(conn);
	pulse_link_finish_connection(link, result);
}

static void pulse_link_mtu_exchanged(struct bt_conn* conn,
									 uint8_t error,
									 struct bt_gatt_exchange_params* params) {
	struct pulse_link* link = CONTAINER_OF(params, struct pulse_link, mtu_exchange);

	if (link != active_link) {
		return;
	}
	k_mutex_lock(&link->lock, K_FOREVER);
	bool matches = link->conn == conn && link->negotiation_stage == PULSE_LINK_NEGOTIATION_MTU;
	k_mutex_unlock(&link->lock);
	if (!matches) {
		return;
	}
	if (error != 0U) {
		pulse_link_record_negotiation_error(link, -EIO);
	}
	pulse_link_finish_negotiation(link);
}

static void pulse_link_continue_negotiation(struct pulse_link* link) {
	for (;;) {
		struct pulse_link_snapshot snapshot = {0};
		struct bt_conn* conn = NULL;
		enum pulse_link_negotiation_stage stage;
		int result;

		k_mutex_lock(&link->lock, K_FOREVER);
		if (link->conn != NULL && !link->negotiation_finished) {
			conn = bt_conn_ref(link->conn);
			stage = (enum pulse_link_negotiation_stage) link->negotiation_stage;
		} else {
			stage = PULSE_LINK_NEGOTIATION_IDLE;
		}
		k_mutex_unlock(&link->lock);
		if (conn == NULL) {
			return;
		}

		result = pulse_link_basic_snapshot(conn, &snapshot);
		if (result != 0) {
			bt_conn_unref(conn);
			pulse_link_record_negotiation_error(link, result);
			pulse_link_finish_negotiation(link);
			return;
		}

		switch (stage) {
		case PULSE_LINK_NEGOTIATION_PARAM:
			if (snapshot.interval_us == PULSE_LINK_INTERVAL_US) {
				(void) pulse_link_move_stage(link,
											 PULSE_LINK_NEGOTIATION_PARAM,
											 PULSE_LINK_NEGOTIATION_PHY);
				bt_conn_unref(conn);
				continue;
			}
			result = bt_conn_le_param_update(conn, &pulse_connection_parameters);
			bt_conn_unref(conn);
			if (result == 0) {
				return;
			}
			pulse_link_record_negotiation_error(link, result);
			(void) pulse_link_move_stage(link,
										 PULSE_LINK_NEGOTIATION_PARAM,
										 PULSE_LINK_NEGOTIATION_PHY);
			break;

		case PULSE_LINK_NEGOTIATION_PHY:
			if (snapshot.tx_phy == BT_GAP_LE_PHY_2M && snapshot.rx_phy == BT_GAP_LE_PHY_2M) {
				(void) pulse_link_move_stage(link,
											 PULSE_LINK_NEGOTIATION_PHY,
											 PULSE_LINK_NEGOTIATION_DATA_LENGTH);
				bt_conn_unref(conn);
				continue;
			}
#if defined(CONFIG_BT_USER_PHY_UPDATE)
#if defined(CONFIG_BT_AUTO_PHY_CENTRAL_2M)
			/* The host queued its 2 Mbit/s request before notifying connected callbacks. */
			bt_conn_unref(conn);
			return;
#else
			result = bt_conn_le_phy_update(conn, BT_CONN_LE_PHY_PARAM_2M);
#endif
#else
			result = -ENOTSUP;
#endif
			bt_conn_unref(conn);
			if (result == 0 || result == -EBUSY || result == -EALREADY) {
				return;
			}
			pulse_link_record_negotiation_error(link, result);
			(void) pulse_link_move_stage(link,
										 PULSE_LINK_NEGOTIATION_PHY,
										 PULSE_LINK_NEGOTIATION_DATA_LENGTH);
			break;

		case PULSE_LINK_NEGOTIATION_DATA_LENGTH:
			if (snapshot.tx_max_len >= PULSE_LINK_DATA_LENGTH &&
				snapshot.rx_max_len >= PULSE_LINK_DATA_LENGTH) {
				(void) pulse_link_move_stage(link,
											 PULSE_LINK_NEGOTIATION_DATA_LENGTH,
											 PULSE_LINK_NEGOTIATION_MTU);
				bt_conn_unref(conn);
				continue;
			}
#if defined(CONFIG_BT_USER_DATA_LEN_UPDATE)
			result = bt_conn_le_data_len_update(conn, BT_LE_DATA_LEN_PARAM_MAX);
#else
			result = -ENOTSUP;
#endif
			bt_conn_unref(conn);
			if (result == 0 || result == -EBUSY || result == -EALREADY) {
				return;
			}
			pulse_link_record_negotiation_error(link, result);
			(void) pulse_link_move_stage(link,
										 PULSE_LINK_NEGOTIATION_DATA_LENGTH,
										 PULSE_LINK_NEGOTIATION_MTU);
			break;

		case PULSE_LINK_NEGOTIATION_MTU:
			if (snapshot.att_mtu >= PULSE_PROTOCOL_MIN_ATT_MTU) {
				bt_conn_unref(conn);
				pulse_link_finish_negotiation(link);
				return;
			}
#if defined(CONFIG_BT_GATT_CLIENT)
			link->mtu_exchange.func = pulse_link_mtu_exchanged;
			result = bt_gatt_exchange_mtu(conn, &link->mtu_exchange);
#else
			result = -ENOTSUP;
#endif
			bt_conn_unref(conn);
			if (result == 0 || result == -EALREADY) {
				return;
			}
			pulse_link_record_negotiation_error(link, result);
			pulse_link_finish_negotiation(link);
			return;

		case PULSE_LINK_NEGOTIATION_PEER_MTU:
			bt_conn_unref(conn);
			return;

		default:
			bt_conn_unref(conn);
			pulse_link_record_negotiation_error(link, -EPROTO);
			pulse_link_finish_negotiation(link);
			return;
		}
	}
}

static void pulse_link_stop_scanning(struct pulse_link* link) {
	bool stop;

	k_mutex_lock(&link->lock, K_FOREVER);
	stop = link->scanning;
	link->scanning = false;
	k_mutex_unlock(&link->lock);
	if (stop) {
		(void) bt_le_scan_stop();
	}
}

static void pulse_link_stop_advertising(struct pulse_link* link) {
	bool stop;

	k_mutex_lock(&link->lock, K_FOREVER);
	stop = link->advertising;
	link->advertising = false;
	k_mutex_unlock(&link->lock);
	if (stop) {
		(void) bt_le_adv_stop();
	}
}

static int pulse_link_start_scanning(struct pulse_link* link) {
	int result;

	k_mutex_lock(&link->lock, K_FOREVER);
	if (link->scanning) {
		k_mutex_unlock(&link->lock);
		return 0;
	}
	link->scanning = true;
	k_mutex_unlock(&link->lock);

	result = bt_le_scan_start(&pulse_scan_parameters, NULL);
	if (result != 0) {
		k_mutex_lock(&link->lock, K_FOREVER);
		link->scanning = false;
		k_mutex_unlock(&link->lock);
	}
	return result;
}

static int pulse_link_start_advertising(struct pulse_link* link) {
	bool can_defer;
	int result;

	k_mutex_lock(&link->lock, K_FOREVER);
	if (link->advertising) {
		k_mutex_unlock(&link->lock);
		return 0;
	}
	can_defer = !link->pool_ready;
	link->pending_advertising = false;
	k_mutex_unlock(&link->lock);

	result = bt_le_adv_start(&pulse_advertising_parameters,
							 pulse_advertising_data,
							 ARRAY_SIZE(pulse_advertising_data),
							 NULL,
							 0U);
	k_mutex_lock(&link->lock, K_FOREVER);
	if (result == 0) {
		link->advertising = true;
		link->pool_ready = false;
	} else if (result == -ENOMEM && can_defer && link->awaiting_connection) {
		link->pending_advertising = true;
		result = 0;
	}
	k_mutex_unlock(&link->lock);
	return result;
}

static int pulse_link_create_connection(struct pulse_link* link, uint32_t attempt_generation) {
	struct bt_conn* conn = NULL;
	bt_addr_le_t peer;
	bool can_defer;
	bool cancelled = false;
	int result;

	k_mutex_lock(&link->lock, K_FOREVER);
	if (link->attempt_generation != attempt_generation || !link->awaiting_connection ||
		link->conn != NULL || !link->peer_valid) {
		k_mutex_unlock(&link->lock);
		return -ECANCELED;
	}
	peer = link->peer_address;
	can_defer = !link->pool_ready;
	link->pending_create = false;
	k_mutex_unlock(&link->lock);

	result =
		bt_conn_le_create(&peer, &pulse_create_parameters, &pulse_connection_parameters, &conn);
	k_mutex_lock(&link->lock, K_FOREVER);
	if (result == 0) {
		link->pool_ready = false;
		/*
		 * A deadline can cancel the attempt while bt_conn_le_create() is
		 * outside our lock.  Do not resurrect that attempt after cancellation;
		 * retire the controller connection below instead.
		 */
		if (link->attempt_generation != attempt_generation || !link->awaiting_connection ||
			link->negotiation_finished) {
			cancelled = true;
		} else {
			link->conn = conn;
		}
	} else if (link->attempt_generation != attempt_generation) {
		result = -ECANCELED;
	} else if (result == -ENOMEM && can_defer && link->awaiting_connection) {
		link->pending_create = true;
		result = 0;
	}
	k_mutex_unlock(&link->lock);
	if (cancelled) {
		(void) bt_conn_disconnect(conn, BT_HCI_ERR_REMOTE_USER_TERM_CONN);
		bt_conn_unref(conn);
		return -ECANCELED;
	}
	return result;
}

static int pulse_link_start_role_activity(struct pulse_link* link) {
	enum pulse_link_role role;
	int result;

	k_mutex_lock(&link->lock, K_FOREVER);
	role = link->role;
	k_mutex_unlock(&link->lock);

	if (role == PULSE_LINK_ROLE_INITIATOR) {
		return pulse_link_start_scanning(link);
	}
	if (role == PULSE_LINK_ROLE_RESPONDER) {
		return pulse_link_start_advertising(link);
	}

	/* Auto role discovery needs both identity-bearing advertising and passive scanning. */
	result = pulse_link_start_advertising(link);
	if (result != 0) {
		return result;
	}
	result = pulse_link_start_scanning(link);
	if (result != 0) {
		pulse_link_stop_advertising(link);
	}
	return result;
}

static void pulse_link_scan_received(const struct bt_le_scan_recv_info* info,
									 struct net_buf_simple* advertising_data) {
	struct pulse_link* link = active_link;
	void (*scan_found)(const bt_addr_le_t*, int8_t, enum pulse_link_role, void*) = NULL;
	void* user_data = NULL;
	enum pulse_link_role role;
	bt_addr_le_t peer;
	bool stop_advertising = false;
	bool accepted = false;
	bool probe_only = false;
	uint32_t attempt_generation;
	int comparison;
	int result;

	if (link == NULL || info == NULL || info->addr == NULL || advertising_data == NULL ||
		(info->adv_props & BT_GAP_ADV_PROP_CONNECTABLE) == 0U ||
		!pulse_link_has_service(advertising_data)) {
		return;
	}

	k_mutex_lock(&link->lock, K_FOREVER);
	if (!link->initialized || (!link->awaiting_connection && !link->probe_only) ||
		!link->scanning || bt_addr_le_eq(info->addr, &link->local_address) ||
		(link->role == PULSE_LINK_ROLE_INITIATOR && link->peer_valid &&
		 !bt_addr_le_eq(info->addr, &link->peer_address))) {
		k_mutex_unlock(&link->lock);
		return;
	}

	role = link->role;
	attempt_generation = link->attempt_generation;
	probe_only = link->probe_only;
	if (role == PULSE_LINK_ROLE_UNRESOLVED) {
		comparison = pulse_link_address_compare(&link->local_address, info->addr);
		if (comparison == 0) {
			k_mutex_unlock(&link->lock);
			return;
		}
		role = comparison < 0 ? PULSE_LINK_ROLE_INITIATOR : PULSE_LINK_ROLE_RESPONDER;
		link->role = role;
	}

	bt_addr_le_copy(&link->peer_address, info->addr);
	link->peer_valid = true;
	link->advertisement_rssi = info->rssi;
	peer = link->peer_address;
	if (!link->scan_reported) {
		link->scan_reported = true;
		link->probe_only = false;
		accepted = true;
		scan_found = link->config.callbacks.scan_found;
		user_data = link->config.user_data;
	}
	stop_advertising = role == PULSE_LINK_ROLE_INITIATOR && link->advertising;
	k_mutex_unlock(&link->lock);

	if (!accepted) {
		return;
	}
	if (scan_found != NULL) {
		scan_found(&peer, info->rssi, role, user_data);
	}
	pulse_link_stop_scanning(link);
	k_sem_give(&link->scan_found_sem);
	if (probe_only) {
		return;
	}
	if (role == PULSE_LINK_ROLE_RESPONDER) {
		return;
	}
	if (stop_advertising) {
		pulse_link_stop_advertising(link);
	}
	result = pulse_link_create_connection(link, attempt_generation);
	if (result != 0 && result != -ECANCELED) {
		pulse_link_finish_connection(link, result);
	}
}

static void pulse_link_connected(struct bt_conn* conn, uint8_t error) {
	struct pulse_link* link = active_link;
	struct bt_conn_info info;
	struct bt_conn* release = NULL;
	bool stop_scan = false;
	int result;

	if (link == NULL) {
		return;
	}

	if (error != 0U) {
		k_mutex_lock(&link->lock, K_FOREVER);
		if (link->conn != conn) {
			/* Ignore a callback from an attempt retired at an earlier deadline. */
			k_mutex_unlock(&link->lock);
			return;
		}
		release = link->conn;
		link->conn = NULL;
		link->pool_ready = false;
		link->advertising = false;
		link->scanning = false;
		link->pending_create = false;
		link->pending_advertising = false;
		k_mutex_unlock(&link->lock);
		bt_conn_unref(release);
		pulse_link_finish_connection(link, -ECONNREFUSED);
		/* A disconnect waiter may immediately reconnect, so publish the finished attempt first. */
		k_sem_give(&link->disconnected_sem);
		return;
	}

	result = bt_conn_get_info(conn, &info);
	if (result != 0 || info.type != BT_CONN_TYPE_LE) {
		(void) bt_conn_disconnect(conn, BT_HCI_ERR_REMOTE_USER_TERM_CONN);
		pulse_link_finish_connection(link, result != 0 ? result : -EPROTOTYPE);
		return;
	}

	k_mutex_lock(&link->lock, K_FOREVER);
	if (!link->initialized || !link->awaiting_connection ||
		(link->conn != NULL && link->conn != conn)) {
		k_mutex_unlock(&link->lock);
		(void) bt_conn_disconnect(conn, BT_HCI_ERR_REMOTE_USER_TERM_CONN);
		return;
	}
	if (link->conn == NULL) {
		link->conn = bt_conn_ref(conn);
	}
	if (info.le.dst != NULL) {
		bt_addr_le_copy(&link->peer_address, info.le.dst);
		link->peer_valid = true;
	}
	link->role = info.role == BT_CONN_ROLE_CENTRAL ? PULSE_LINK_ROLE_INITIATOR
												   : PULSE_LINK_ROLE_RESPONDER;
	link->advertising = false;
	stop_scan = link->scanning;
	link->scanning = false;
	link->pool_ready = false;
	link->negotiation_error = 0;
	link->negotiation_stage = link->role == PULSE_LINK_ROLE_INITIATOR
								  ? PULSE_LINK_NEGOTIATION_PARAM
								  : PULSE_LINK_NEGOTIATION_PEER_MTU;
	k_mutex_unlock(&link->lock);

	if (stop_scan) {
		(void) bt_le_scan_stop();
	}
	if (info.role == BT_CONN_ROLE_CENTRAL) {
		pulse_link_continue_negotiation(link);
	}
}

static void pulse_link_disconnected(struct bt_conn* conn, uint8_t reason) {
	struct pulse_link* link = active_link;
	struct bt_conn* release = NULL;
	void (*disconnected)(enum pulse_link_role, uint8_t, void*) = NULL;
	void* user_data = NULL;
	enum pulse_link_role role;
	bool unfinished;

	if (link == NULL) {
		return;
	}

	k_mutex_lock(&link->lock, K_FOREVER);
	if (link->conn != conn) {
		k_mutex_unlock(&link->lock);
		return;
	}
	release = link->conn;
	link->conn = NULL;
	link->disconnect_reason = reason;
	link->pool_ready = false;
	link->advertising = false;
	link->scanning = false;
	link->pending_create = false;
	link->pending_advertising = false;
	unfinished = link->awaiting_connection && !link->negotiation_finished;
	if (!unfinished) {
		link->connection_result = -ENOTCONN;
	}
	role = link->role;
	disconnected = link->config.callbacks.disconnected;
	user_data = link->config.user_data;
	k_mutex_unlock(&link->lock);

	bt_conn_unref(release);
	if (unfinished) {
		pulse_link_finish_connection(link, -ECONNRESET);
	}
	k_sem_give(&link->disconnected_sem);
	if (disconnected != NULL) {
		disconnected(role, reason, user_data);
	}
}

static void pulse_link_recycled(void) {
	struct pulse_link* link = active_link;
	bool create;
	bool advertise;
	uint32_t attempt_generation;
	int result = 0;

	if (link == NULL) {
		return;
	}

	k_mutex_lock(&link->lock, K_FOREVER);
	link->pool_ready = true;
	create = link->pending_create && link->awaiting_connection;
	advertise = !create && link->pending_advertising && link->awaiting_connection;
	attempt_generation = link->attempt_generation;
	link->pending_create = false;
	link->pending_advertising = false;
	k_mutex_unlock(&link->lock);

	if (create) {
		result = pulse_link_create_connection(link, attempt_generation);
	} else if (advertise) {
		result = pulse_link_start_advertising(link);
	}
	if (result != 0 && result != -ECANCELED) {
		pulse_link_finish_connection(link, result);
	}
}

static bool pulse_link_param_request(struct bt_conn* conn, struct bt_le_conn_param* parameters) {
	struct pulse_link* link = active_link;
	bool matches;

	if (link == NULL || parameters == NULL) {
		return true;
	}
	k_mutex_lock(&link->lock, K_FOREVER);
	matches = link->conn == conn;
	k_mutex_unlock(&link->lock);
	if (!matches) {
		return true;
	}

	*parameters = pulse_connection_parameters;
	return true;
}

static void pulse_link_param_updated(struct bt_conn* conn,
									 uint16_t interval,
									 uint16_t latency,
									 uint16_t timeout) {
	struct pulse_link* link = active_link;

	ARG_UNUSED(interval);
	ARG_UNUSED(latency);
	ARG_UNUSED(timeout);
	if (link == NULL) {
		return;
	}
	k_mutex_lock(&link->lock, K_FOREVER);
	bool matches = link->conn == conn;
	k_mutex_unlock(&link->lock);
	if (matches &&
		pulse_link_move_stage(link, PULSE_LINK_NEGOTIATION_PARAM, PULSE_LINK_NEGOTIATION_PHY)) {
		pulse_link_continue_negotiation(link);
	}
}

#if defined(CONFIG_BT_USER_PHY_UPDATE)
static void pulse_link_phy_updated(struct bt_conn* conn, struct bt_conn_le_phy_info* info) {
	struct pulse_link* link = active_link;

	ARG_UNUSED(info);
	if (link == NULL) {
		return;
	}
	k_mutex_lock(&link->lock, K_FOREVER);
	bool matches = link->conn == conn;
	k_mutex_unlock(&link->lock);
	if (matches && pulse_link_move_stage(link,
										 PULSE_LINK_NEGOTIATION_PHY,
										 PULSE_LINK_NEGOTIATION_DATA_LENGTH)) {
		pulse_link_continue_negotiation(link);
	}
}
#endif

#if defined(CONFIG_BT_USER_DATA_LEN_UPDATE)
static void pulse_link_data_length_updated(struct bt_conn* conn,
										   struct bt_conn_le_data_len_info* info) {
	struct pulse_link* link = active_link;

	ARG_UNUSED(info);
	if (link == NULL) {
		return;
	}
	k_mutex_lock(&link->lock, K_FOREVER);
	bool matches = link->conn == conn;
	k_mutex_unlock(&link->lock);
	if (matches && pulse_link_move_stage(link,
										 PULSE_LINK_NEGOTIATION_DATA_LENGTH,
										 PULSE_LINK_NEGOTIATION_MTU)) {
		pulse_link_continue_negotiation(link);
	}
}
#endif

static void pulse_link_att_mtu_updated(struct bt_conn* conn, uint16_t tx, uint16_t rx) {
	struct pulse_link* link = active_link;
	bool matches;

	ARG_UNUSED(tx);
	ARG_UNUSED(rx);
	if (link == NULL) {
		return;
	}
	k_mutex_lock(&link->lock, K_FOREVER);
	matches = link->conn == conn && (link->negotiation_stage == PULSE_LINK_NEGOTIATION_MTU ||
									 link->negotiation_stage == PULSE_LINK_NEGOTIATION_PEER_MTU);
	k_mutex_unlock(&link->lock);
	if (matches) {
		pulse_link_finish_negotiation(link);
	}
}

static struct bt_le_scan_cb pulse_link_scan_callbacks = {
	.recv = pulse_link_scan_received,
};

static struct bt_conn_cb pulse_link_connection_callbacks = {
	.connected = pulse_link_connected,
	.disconnected = pulse_link_disconnected,
	.recycled = pulse_link_recycled,
	.le_param_req = pulse_link_param_request,
	.le_param_updated = pulse_link_param_updated,
#if defined(CONFIG_BT_USER_PHY_UPDATE)
	.le_phy_updated = pulse_link_phy_updated,
#endif
#if defined(CONFIG_BT_USER_DATA_LEN_UPDATE)
	.le_data_len_updated = pulse_link_data_length_updated,
#endif
};

static struct bt_gatt_cb pulse_link_gatt_callbacks = {
	.att_mtu_updated = pulse_link_att_mtu_updated,
};

int pulse_link_start(struct pulse_link* link, const struct pulse_link_config* config) {
	bt_addr_le_t identities[1];
	size_t identity_count = ARRAY_SIZE(identities);
	uint8_t configured_role;
	int result;

	if (link == NULL || active_link != NULL) {
		return link == NULL ? -EINVAL : -EALREADY;
	}

	memset(link, 0, sizeof(*link));
	k_mutex_init(&link->lock);
	k_sem_init(&link->scan_found_sem, 0U, 1U);
	k_sem_init(&link->connected_sem, 0U, 1U);
	k_sem_init(&link->disconnected_sem, 0U, 1U);
	if (config != NULL) {
		link->config = *config;
	}
	link->advertisement_rssi = PULSE_LINK_RSSI_UNAVAILABLE;
	link->connection_result = -EINPROGRESS;
	link->pool_ready = true;
	link->attempt_generation = 1U;
	configured_role = pulse_link_configured_role();
	if (configured_role > PULSE_LINK_ROLE_RESPONDER) {
		return -EINVAL;
	}
	link->role = (enum pulse_link_role) configured_role;

	result = bt_enable(NULL);
	if (result != 0 && result != -EALREADY) {
		return result;
	}
	bt_id_get(identities, &identity_count);
	if (identity_count == 0U || bt_addr_le_eq(&identities[0], BT_ADDR_LE_ANY)) {
		return -ENOENT;
	}
	link->local_address = identities[0];
	link->initialized = true;
	link->awaiting_connection = true;
	active_link = link;

	result = bt_conn_cb_register(&pulse_link_connection_callbacks);
	if (result != 0) {
		goto fail_active;
	}
	result = bt_le_scan_cb_register(&pulse_link_scan_callbacks);
	if (result != 0) {
		(void) bt_conn_cb_unregister(&pulse_link_connection_callbacks);
		goto fail_active;
	}
	bt_gatt_cb_register(&pulse_link_gatt_callbacks);

	result = pulse_link_start_role_activity(link);
	if (result == 0) {
		return 0;
	}

	pulse_link_stop_scanning(link);
	pulse_link_stop_advertising(link);
	(void) bt_gatt_cb_unregister(&pulse_link_gatt_callbacks);
	bt_le_scan_cb_unregister(&pulse_link_scan_callbacks);
	(void) bt_conn_cb_unregister(&pulse_link_connection_callbacks);

fail_active:
	active_link = NULL;
	link->initialized = false;
	link->awaiting_connection = false;
	return result;
}

enum pulse_link_role pulse_link_role_get(struct pulse_link* link) {
	enum pulse_link_role role;

	if (link == NULL) {
		return PULSE_LINK_ROLE_UNRESOLVED;
	}
	k_mutex_lock(&link->lock, K_FOREVER);
	role = link->role;
	k_mutex_unlock(&link->lock);
	return role;
}

int pulse_link_wait_scan_found(struct pulse_link* link,
							   bt_addr_le_t* peer,
							   int8_t* advertisement_rssi,
							   k_timeout_t timeout) {
	int result;

	if (link == NULL || !link->initialized) {
		return -EINVAL;
	}
	result = k_sem_take(&link->scan_found_sem, timeout);
	if (result != 0) {
		return result;
	}

	k_mutex_lock(&link->lock, K_FOREVER);
	if (peer != NULL) {
		*peer = link->peer_address;
	}
	if (advertisement_rssi != NULL) {
		*advertisement_rssi = link->advertisement_rssi;
	}
	k_mutex_unlock(&link->lock);
	return 0;
}

int pulse_link_probe_peer(struct pulse_link* link,
						  bt_addr_le_t* peer,
						  int8_t* advertisement_rssi,
						  k_timeout_t timeout) {
	bool accepted;
	int result;

	if (link == NULL || !link->initialized) {
		return -EINVAL;
	}

	k_mutex_lock(&link->lock, K_FOREVER);
	if (link->role != PULSE_LINK_ROLE_INITIATOR) {
		result = -EACCES;
		goto done;
	}
	if (!link->peer_valid) {
		result = -ENOENT;
		goto done;
	}
	if (link->conn != NULL || link->awaiting_connection || link->scanning || link->advertising ||
		link->probe_only) {
		result = -EBUSY;
		goto done;
	}

	k_sem_reset(&link->scan_found_sem);
	link->scan_reported = false;
	++link->attempt_generation;
	link->probe_only = true;
	k_mutex_unlock(&link->lock);

	result = pulse_link_start_scanning(link);
	if (result == 0) {
		result = k_sem_take(&link->scan_found_sem, timeout);
	}
	if (result != 0) {
		pulse_link_stop_scanning(link);
		k_mutex_lock(&link->lock, K_FOREVER);
		/* Resolve the deadline race in favour of a report that the RX callback
		 * already accepted. scan_reported is set before that callback releases
		 * the lock; after stop_scanning() clears scanning no later report can be
		 * accepted. Wait for an in-flight callback before the caller closes or
		 * copies its measurement record. */
		accepted = link->scan_reported;
		link->probe_only = false;
		k_mutex_unlock(&link->lock);
		if (accepted && k_sem_take(&link->scan_found_sem, K_FOREVER) == 0) {
			result = 0;
		}
	}
	if (result != 0) {
		return result;
	}

	k_mutex_lock(&link->lock, K_FOREVER);
	if (peer != NULL) {
		*peer = link->peer_address;
	}
	if (advertisement_rssi != NULL) {
		*advertisement_rssi = link->advertisement_rssi;
	}
	k_mutex_unlock(&link->lock);
	return 0;

done:
	k_mutex_unlock(&link->lock);
	return result;
}

struct bt_conn* pulse_link_connected_ref(struct pulse_link* link) {
	struct bt_conn* conn = NULL;

	if (link == NULL || !link->initialized) {
		return NULL;
	}
	k_mutex_lock(&link->lock, K_FOREVER);
	if (link->negotiation_finished && link->connection_result == 0 && link->conn != NULL) {
		conn = bt_conn_ref(link->conn);
	}
	k_mutex_unlock(&link->lock);
	return conn;
}

int pulse_link_wait_connected(struct pulse_link* link, k_timeout_t timeout) {
	bool finished;
	int result;

	if (link == NULL || !link->initialized) {
		return -EINVAL;
	}
	k_mutex_lock(&link->lock, K_FOREVER);
	finished = link->negotiation_finished;
	result = link->connection_result;
	k_mutex_unlock(&link->lock);
	if (finished) {
		return result;
	}

	result = k_sem_take(&link->connected_sem, timeout);
	if (result != 0) {
		return result;
	}
	k_mutex_lock(&link->lock, K_FOREVER);
	result = link->connection_result;
	k_mutex_unlock(&link->lock);
	return result;
}

int pulse_link_disconnect(struct pulse_link* link) {
	struct bt_conn* conn = NULL;
	struct pulse_link_timing_callbacks timing = {0};
	void (*connected)(enum pulse_link_role, int, void*) = NULL;
	void* user_data = NULL;
	enum pulse_link_role role = PULSE_LINK_ROLE_UNRESOLVED;
	bool stop_advertising = false;
	bool stop_scanning = false;
	bool cancelled_pending = false;
	int result;

	if (link == NULL || !link->initialized) {
		return -EINVAL;
	}
	k_mutex_lock(&link->lock, K_FOREVER);
	if (link->conn != NULL) {
		conn = bt_conn_ref(link->conn);
	} else if (link->awaiting_connection || link->scanning || link->advertising ||
			   link->pending_create || link->pending_advertising) {
		/*
		 * Publish cancellation while holding the state lock.  An in-flight
		 * scan callback will then observe awaiting_connection=false before it
		 * can create a connection.
		 */
		stop_scanning = link->scanning;
		stop_advertising = link->advertising;
		link->scanning = false;
		link->advertising = false;
		link->pending_create = false;
		link->pending_advertising = false;
		link->awaiting_connection = false;
		link->negotiation_finished = true;
		link->negotiation_stage = PULSE_LINK_NEGOTIATION_DONE;
		link->connection_result = -ECANCELED;
		if (link->negotiation_error == 0) {
			link->negotiation_error = -ECANCELED;
		}
		role = link->role;
		connected = link->config.callbacks.connected;
		user_data = link->config.user_data;
		if (link->reconnect_timing_active) {
			timing = link->reconnect_timing;
			link->reconnect_timing_active = false;
		}
		cancelled_pending = true;
		k_sem_give(&link->connected_sem);
	}
	k_mutex_unlock(&link->lock);
	if (cancelled_pending) {
		if (stop_scanning) {
			(void) bt_le_scan_stop();
		}
		if (stop_advertising) {
			(void) bt_le_adv_stop();
		}
		if (timing.finished != NULL) {
			timing.finished(role, -ECANCELED, timing.user_data);
		}
		if (connected != NULL) {
			connected(role, -ECANCELED, user_data);
		}
		return 0;
	}
	if (conn == NULL) {
		return -ENOTCONN;
	}

	result = bt_conn_disconnect(conn, BT_HCI_ERR_REMOTE_USER_TERM_CONN);
	bt_conn_unref(conn);
	return result;
}

int pulse_link_wait_disconnected(struct pulse_link* link, k_timeout_t timeout) {
	bool disconnected;
	int result;

	if (link == NULL || !link->initialized) {
		return -EINVAL;
	}
	k_mutex_lock(&link->lock, K_FOREVER);
	disconnected = link->conn == NULL;
	k_mutex_unlock(&link->lock);
	if (disconnected) {
		return 0;
	}
	result = k_sem_take(&link->disconnected_sem, timeout);
	return result;
}

int pulse_link_reconnect(struct pulse_link* link,
						 const struct pulse_link_timing_callbacks* timing) {
	struct pulse_link_timing_callbacks timing_copy = {0};
	void (*started)(enum pulse_link_role, void*) = NULL;
	void* timing_user_data = NULL;
	enum pulse_link_role role;
	int result;

	if (link == NULL || !link->initialized) {
		return -EINVAL;
	}
	if (timing != NULL) {
		timing_copy = *timing;
	}

	k_mutex_lock(&link->lock, K_FOREVER);
	if (link->conn != NULL || link->awaiting_connection || link->scanning || link->advertising ||
		link->probe_only) {
		k_mutex_unlock(&link->lock);
		return -EBUSY;
	}
	if (link->role == PULSE_LINK_ROLE_INITIATOR && !link->peer_valid) {
		k_mutex_unlock(&link->lock);
		return -ENOENT;
	}

	k_sem_reset(&link->scan_found_sem);
	k_sem_reset(&link->connected_sem);
	k_sem_reset(&link->disconnected_sem);
	link->scan_reported = false;
	++link->attempt_generation;
	link->connection_result = -EINPROGRESS;
	link->negotiation_error = 0;
	link->negotiation_finished = false;
	link->negotiation_stage = PULSE_LINK_NEGOTIATION_IDLE;
	link->awaiting_connection = true;
	link->reconnect_timing = timing_copy;
	link->reconnect_timing_active = timing != NULL;
	role = link->role;
	started = timing_copy.started;
	timing_user_data = timing_copy.user_data;
	k_mutex_unlock(&link->lock);

	if (started != NULL) {
		started(role, timing_user_data);
	}
	result = pulse_link_start_role_activity(link);
	if (result != 0) {
		pulse_link_finish_connection(link, result);
	}
	return result;
}

static int8_t pulse_link_read_rssi(struct bt_conn* conn) {
	struct bt_hci_cp_read_rssi* command;
	struct bt_hci_rp_read_rssi* response_data;
	struct net_buf* command_buffer;
	struct net_buf* response = NULL;
	uint16_t handle;
	int8_t rssi = PULSE_LINK_RSSI_UNAVAILABLE;
	int result;

	result = bt_hci_get_conn_handle(conn, &handle);
	if (result != 0) {
		return rssi;
	}
	command_buffer = bt_hci_cmd_alloc(K_NO_WAIT);
	if (command_buffer == NULL) {
		return rssi;
	}
	command = net_buf_add(command_buffer, sizeof(*command));
	command->handle = sys_cpu_to_le16(handle);
	result = bt_hci_cmd_send_sync(BT_HCI_OP_READ_RSSI, command_buffer, &response);
	if (result != 0 || response == NULL || response->len < sizeof(*response_data)) {
		if (response != NULL) {
			net_buf_unref(response);
		}
		return rssi;
	}

	response_data = (struct bt_hci_rp_read_rssi*) response->data;
	if (response_data->status == 0U && response_data->rssi != BT_HCI_LE_RSSI_NOT_AVAILABLE) {
		rssi = response_data->rssi;
	}
	net_buf_unref(response);
	return rssi;
}

int pulse_link_snapshot_get(struct pulse_link* link, struct pulse_link_snapshot* snapshot) {
	struct bt_conn* conn = NULL;
	int result = 0;

	if (link == NULL || snapshot == NULL || !link->initialized) {
		return -EINVAL;
	}
	memset(snapshot, 0, sizeof(*snapshot));
	snapshot->rssi_dbm = PULSE_LINK_RSSI_UNAVAILABLE;

	k_mutex_lock(&link->lock, K_FOREVER);
	snapshot->role = link->role;
	snapshot->local_address = link->local_address;
	if (link->peer_valid) {
		snapshot->peer_address = link->peer_address;
	}
	snapshot->negotiation_error = link->negotiation_error;
	if (link->conn != NULL) {
		conn = bt_conn_ref(link->conn);
	}
	k_mutex_unlock(&link->lock);

	if (conn != NULL) {
		result = pulse_link_basic_snapshot(conn, snapshot);
		if (result == 0 && snapshot->connected) {
			snapshot->rssi_dbm = pulse_link_read_rssi(conn);
		}
		bt_conn_unref(conn);
	}
	return result;
}
