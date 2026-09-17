# SPDX-License-Identifier: Apache-2.0
#
# This script is invoked by an always-run build target for final PULSE capture
# images.  Configure-time checks alone are insufficient: a source file can be
# edited after configuration and then compiled without CMake being rerun.

if(NOT DEFINED PULSE_SOURCE_DIR OR PULSE_SOURCE_DIR STREQUAL "")
    message(FATAL_ERROR "PULSE final-capture build guard is missing PULSE_SOURCE_DIR")
endif()
if(NOT DEFINED PULSE_EXPECTED_GIT_COMMIT OR PULSE_EXPECTED_GIT_COMMIT STREQUAL "")
    message(FATAL_ERROR "PULSE final-capture build guard is missing the expected Git commit")
endif()

execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY "${PULSE_SOURCE_DIR}"
    RESULT_VARIABLE _pulse_guard_revision_result
    OUTPUT_VARIABLE _pulse_guard_revision
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE _pulse_guard_revision_error)
if(NOT _pulse_guard_revision_result EQUAL 0)
    message(FATAL_ERROR
        "PULSE final-capture build guard could not read the Git revision: "
        "${_pulse_guard_revision_error}")
endif()
if(NOT _pulse_guard_revision STREQUAL PULSE_EXPECTED_GIT_COMMIT)
    message(FATAL_ERROR
        "PULSE final-capture source revision changed after configuration: expected "
        "${PULSE_EXPECTED_GIT_COMMIT}, got ${_pulse_guard_revision}. Reconfigure from a clean tree.")
endif()

execute_process(
    COMMAND git status --porcelain --untracked-files=normal
    WORKING_DIRECTORY "${PULSE_SOURCE_DIR}"
    RESULT_VARIABLE _pulse_guard_status_result
    OUTPUT_VARIABLE _pulse_guard_status
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE _pulse_guard_status_error)
if(NOT _pulse_guard_status_result EQUAL 0)
    message(FATAL_ERROR
        "PULSE final-capture build guard could not inspect the worktree: "
        "${_pulse_guard_status_error}")
endif()
if(NOT _pulse_guard_status STREQUAL "")
    message(FATAL_ERROR
        "PULSE final-capture build requires a clean worktree with no untracked files. "
        "Reconfigure after resolving:\n${_pulse_guard_status}")
endif()

if(DEFINED PULSE_ARTIFACT_PATH AND NOT PULSE_ARTIFACT_PATH STREQUAL "")
    if(NOT EXISTS "${PULSE_ARTIFACT_PATH}")
        message(FATAL_ERROR
            "PULSE final-capture artifact disappeared after configuration: ${PULSE_ARTIFACT_PATH}")
    endif()
    if(NOT DEFINED PULSE_EXPECTED_ARTIFACT_SHA256 OR
       PULSE_EXPECTED_ARTIFACT_SHA256 STREQUAL "")
        message(FATAL_ERROR "PULSE final-capture build guard is missing the artifact SHA-256")
    endif()
    file(SHA256 "${PULSE_ARTIFACT_PATH}" _pulse_guard_artifact_sha256)
    if(NOT _pulse_guard_artifact_sha256 STREQUAL PULSE_EXPECTED_ARTIFACT_SHA256)
        message(FATAL_ERROR
            "PULSE final-capture artifact changed after configuration: expected "
            "${PULSE_EXPECTED_ARTIFACT_SHA256}, got ${_pulse_guard_artifact_sha256}. Reconfigure.")
    endif()
endif()
