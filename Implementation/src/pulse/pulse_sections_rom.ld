/* SPDX-License-Identifier: Apache-2.0 */

SECTION_PROLOGUE(pulse_encoder,,SUBALIGN(4))
{
	__pulse_encoder_start = .;
	KEEP(*(.pulse.encoder))
	__pulse_encoder_end = .;
} GROUP_LINK_IN(ROMABLE_REGION)

SECTION_PROLOGUE(pulse_head_initial,,SUBALIGN(4))
{
	__pulse_head_initial_start = .;
	KEEP(*(.pulse.head))
	__pulse_head_initial_end = .;
} GROUP_LINK_IN(ROMABLE_REGION)

SECTION_PROLOGUE(pulse_fixture,,SUBALIGN(4))
{
	__pulse_fixture_start = .;
	KEEP(*(.pulse.fixture))
	__pulse_fixture_end = .;
} GROUP_LINK_IN(ROMABLE_REGION)

SECTION_PROLOGUE(pulse_golden,,SUBALIGN(4))
{
	__pulse_golden_start = .;
	KEEP(*(.pulse.golden))
	__pulse_golden_end = .;
} GROUP_LINK_IN(ROMABLE_REGION)
