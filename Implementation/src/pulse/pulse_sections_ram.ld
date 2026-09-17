/* SPDX-License-Identifier: Apache-2.0 */

SECTION_DATA_PROLOGUE(pulse_state,(NOLOAD),SUBALIGN(4))
{
	__pulse_state_start = .;
	KEEP(*(.pulse.state))
	__pulse_state_end = .;
} GROUP_NOLOAD_LINK_IN(RAMABLE_REGION, ROMABLE_REGION)

SECTION_DATA_PROLOGUE(pulse_activations,(NOLOAD),SUBALIGN(4))
{
	__pulse_activations_start = .;
	KEEP(*(.pulse.activations))
	__pulse_activations_end = .;
} GROUP_NOLOAD_LINK_IN(RAMABLE_REGION, ROMABLE_REGION)

SECTION_DATA_PROLOGUE(pulse_gradients,(NOLOAD),SUBALIGN(4))
{
	__pulse_gradients_start = .;
	KEEP(*(.pulse.gradient.*))
	__pulse_gradients_end = .;
} GROUP_NOLOAD_LINK_IN(RAMABLE_REGION, ROMABLE_REGION)

SECTION_DATA_PROLOGUE(pulse_protocol,(NOLOAD),SUBALIGN(4))
{
	__pulse_protocol_start = .;
	KEEP(*(.pulse.protocol))
	__pulse_protocol_end = .;
} GROUP_NOLOAD_LINK_IN(RAMABLE_REGION, ROMABLE_REGION)

SECTION_DATA_PROLOGUE(pulse_metrics,(NOLOAD),SUBALIGN(4))
{
	__pulse_metrics_start = .;
	KEEP(*(.pulse.metrics))
	__pulse_metrics_end = .;
} GROUP_NOLOAD_LINK_IN(RAMABLE_REGION, ROMABLE_REGION)
