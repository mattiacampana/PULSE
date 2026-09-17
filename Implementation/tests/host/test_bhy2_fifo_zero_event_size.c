/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * Regression test for the Bosch FIFO parser's zero-length event guard.
 * Compile with a host C compiler:
 * gcc -std=c99 -O2 tests/host/test_bhy2_fifo_zero_event_size.c \
 *   libs/BHY2-Sensor-API/bhy2_hif.c -o test_bhy2_fifo.exe
 */

#include <assert.h>
#include <stdint.h>

/* Include the implementation to exercise its otherwise-private FIFO parser. */
#include "../../libs/BHY2-Sensor-API/bhy2.c"

static void test_unknown_event_rejected(void) {
	struct bhy2_dev dev = {0};
	uint8_t bytes[] = {0x5d, 0x12};
	struct bhy2_fifo_buffer fifo = {0};

	fifo.buffer = bytes;
	fifo.buffer_size = sizeof(bytes);
	fifo.read_length = sizeof(bytes);

	assert(parse_fifo(BHY2_FIFO_TYPE_NON_WAKEUP, &fifo, &dev) == BHY2_E_INVALID_EVENT_SIZE);
	assert(fifo.read_pos == 0);
}

static void test_known_event_advances(void) {
	struct bhy2_dev dev = {0};
	uint8_t bytes[] = {0x5d, 0x12};
	struct bhy2_fifo_buffer fifo = {0};

	dev.event_size[0x5d] = sizeof(bytes);
	fifo.buffer = bytes;
	fifo.buffer_size = sizeof(bytes);
	fifo.read_length = sizeof(bytes);

	assert(parse_fifo(BHY2_FIFO_TYPE_NON_WAKEUP, &fifo, &dev) == BHY2_OK);
	assert(fifo.read_pos == sizeof(bytes));
}

int main(void) {
	test_unknown_event_rejected();
	test_known_event_advances();
	return 0;
}
