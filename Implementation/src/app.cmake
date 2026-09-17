# SPDX-License-Identifier: Apache-2.0
#
# Application sources and include directories, collected into the `app_src`
# INTERFACE library (mirrors the board_drivers.cmake pattern). The top-level
# CMakeLists links `app_src` into Zephyr's `app` target, so these sources are
# compiled as part of the application and the include directories / compile
# definitions propagate to it.
#
# Must be included after find_package(Zephyr) so CONFIG_* are available.

set(APP_SRC_DIR ${CMAKE_CURRENT_LIST_DIR})

add_library(app_src INTERFACE)

target_include_directories(app_src INTERFACE
    ${APP_SRC_DIR}
)

if(CONFIG_SENSWEAR_PULSE_BENCHMARK)
    set(PULSE_FINAL_CAPTURE OFF CACHE BOOL
        "Enforce real-artifact, hardware-revision, and clean-source requirements")
    string(TOLOWER "${CONFIG_SENSWEAR_PULSE_HARDWARE_REVISION}"
        _pulse_hardware_revision_lower)
    string(STRIP "${_pulse_hardware_revision_lower}"
        _pulse_hardware_revision_lower)
    if(PULSE_FINAL_CAPTURE AND
       (_pulse_hardware_revision_lower STREQUAL "" OR
        _pulse_hardware_revision_lower MATCHES
            "^(operator|unknown|unset|tbd|todo|n/?a)([-_ ]|$)"))
        message(FATAL_ERROR
            "PULSE_FINAL_CAPTURE requires an exact CONFIG_SENSWEAR_PULSE_HARDWARE_REVISION")
    endif()

    # Section V uses an isolated application so ordinary GATT services, visual
    # behavior, and monitor logging cannot perturb the measurements. The
    # sensing/device-manager drivers are still linked through the top level and
    # are identical in the PULSE and no-PULSE builds.
    target_sources(app_src INTERFACE
        ${APP_SRC_DIR}/app/pulse_benchmark_main.c
        ${APP_SRC_DIR}/pulse/pulse_link.c
        ${APP_SRC_DIR}/pulse/pulse_metrics.c
    )
    if(CONFIG_SENSWEAR_PULSE_NGMO2_AUTONOMOUS_CAPTURE)
        target_sources(app_src INTERFACE
            ${APP_SRC_DIR}/pulse/pulse_capture_log.c)
    endif()
    target_include_directories(app_src INTERFACE ${APP_SRC_DIR}/pulse)
    zephyr_linker_sources(SECTIONS ${APP_SRC_DIR}/pulse/pulse_sections_rom.ld)
    zephyr_linker_sources(DATA_SECTIONS ${APP_SRC_DIR}/pulse/pulse_sections_ram.ld)

    if(CONFIG_SENSWEAR_PULSE_ALGORITHM)
        find_package(Python3 REQUIRED COMPONENTS Interpreter)
        set(PULSE_FIXTURE_PYTHON ${Python3_EXECUTABLE} CACHE FILEPATH
            "Python with NumPy (and PyTorch for strict correctness vectors)")
        set(PULSE_ARTIFACT_NPZ "" CACHE FILEPATH
            "Optional versioned NPZ from tools/export_pulse_artifact.py")
        if(PULSE_FINAL_CAPTURE AND NOT PULSE_ARTIFACT_NPZ)
            message(FATAL_ERROR
                "PULSE_FINAL_CAPTURE requires PULSE_ARTIFACT_NPZ from export_pulse_artifact.py")
        endif()
        set(_pulse_generated_dir ${CMAKE_CURRENT_BINARY_DIR}/generated/pulse)
        set(_pulse_fixture_c ${_pulse_generated_dir}/pulse_fixture_generated.c)
        set(_pulse_manifest ${_pulse_generated_dir}/pulse_fixture_manifest.json)
        set(_pulse_golden_c ${_pulse_generated_dir}/pulse_golden_generated.c)
        file(MAKE_DIRECTORY ${_pulse_generated_dir})

        set(_pulse_generator_args
            --fixture-c ${_pulse_fixture_c}
            --manifest ${_pulse_manifest})
        if(PULSE_ARTIFACT_NPZ)
            get_filename_component(_pulse_artifact_abs "${PULSE_ARTIFACT_NPZ}" ABSOLUTE)
            if(NOT EXISTS "${_pulse_artifact_abs}")
                message(FATAL_ERROR "PULSE_ARTIFACT_NPZ does not exist: ${_pulse_artifact_abs}")
            endif()
            if(PULSE_FINAL_CAPTURE)
                file(SHA256 "${_pulse_artifact_abs}" _pulse_final_artifact_sha256)
                set(_pulse_final_artifact_path "${_pulse_artifact_abs}")
            endif()
            list(APPEND _pulse_generator_args --artifact-npz ${_pulse_artifact_abs})
            # Development builds regenerate when an artifact changes. A final-capture build must
            # instead keep the configure-time hash immutable so the always-run build guard rejects
            # any post-configuration artifact mutation rather than silently accepting it through an
            # automatic CMake reconfigure.
            if(NOT PULSE_FINAL_CAPTURE)
                set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
                    ${_pulse_artifact_abs})
            endif()
        endif()
        if(CONFIG_SENSWEAR_PULSE_CORRECTNESS)
            list(APPEND _pulse_generator_args
                --golden-c ${_pulse_golden_c}
                --require-pytorch)
        endif()

        execute_process(
            COMMAND ${PULSE_FIXTURE_PYTHON}
                ${CMAKE_SOURCE_DIR}/tools/generate_pulse_fixture.py
                ${_pulse_generator_args}
            RESULT_VARIABLE _pulse_generator_result
            OUTPUT_VARIABLE _pulse_generator_stdout
            ERROR_VARIABLE _pulse_generator_stderr)
        if(NOT _pulse_generator_result EQUAL 0)
            message(FATAL_ERROR
                "PULSE fixture generation failed (${_pulse_generator_result}):\n"
                "${_pulse_generator_stdout}${_pulse_generator_stderr}")
        endif()
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
            ${CMAKE_SOURCE_DIR}/tools/generate_pulse_fixture.py)

        target_sources(app_src INTERFACE
            ${APP_SRC_DIR}/pulse/pulse_math.c
            ${APP_SRC_DIR}/pulse/pulse_peers.c
            ${APP_SRC_DIR}/pulse/pulse_benchmark.c
            ${APP_SRC_DIR}/pulse/pulse_protocol.c
            ${_pulse_fixture_c}
        )
        target_compile_definitions(app_src INTERFACE
            PULSE_MAX_PEERS=${CONFIG_SENSWEAR_PULSE_PEER_LIMIT})

        # Keep the reference contract explicit; contraction changes reduction
        # rounding and makes correctness depend on compiler heuristics.
        set_source_files_properties(${APP_SRC_DIR}/pulse/pulse_math.c
            PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")

        if(CONFIG_SENSWEAR_PULSE_CORRECTNESS)
            target_sources(app_src INTERFACE
                ${APP_SRC_DIR}/pulse/pulse_correctness.c
                ${_pulse_golden_c})
        endif()
    endif()

    execute_process(
        COMMAND git describe --always --dirty --abbrev=12
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        RESULT_VARIABLE _pulse_git_result
        OUTPUT_VARIABLE _pulse_git_revision
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(NOT _pulse_git_result EQUAL 0)
        set(_pulse_git_revision unknown)
    endif()
    execute_process(
        COMMAND git rev-parse HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        RESULT_VARIABLE _pulse_git_commit_result
        OUTPUT_VARIABLE _pulse_git_commit
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(NOT _pulse_git_commit_result EQUAL 0)
        set(_pulse_git_commit unknown)
    endif()
    execute_process(
        COMMAND git status --porcelain --untracked-files=normal
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        RESULT_VARIABLE _pulse_git_status_result
        OUTPUT_VARIABLE _pulse_git_status
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(PULSE_FINAL_CAPTURE AND
       (_pulse_git_revision STREQUAL "unknown" OR
        _pulse_git_commit STREQUAL "unknown" OR
        _pulse_git_revision MATCHES "-dirty$" OR
        NOT _pulse_git_status_result EQUAL 0 OR
        NOT _pulse_git_status STREQUAL ""))
        message(FATAL_ERROR
            "PULSE_FINAL_CAPTURE requires a clean Git revision with no untracked files; "
            "got revision ${_pulse_git_revision} and status:\n${_pulse_git_status}")
    endif()
    if(PULSE_FINAL_CAPTURE)
        set(_pulse_build_guard final_capture_requirements_enforced)
        # Paper-facing provenance is a content-stable full Git object ID.  A
        # `git describe --abbrev=12` string is useful to developers but is not
        # strong enough for the analysis-side ELF/UART binding.
        set(_pulse_firmware_revision ${_pulse_git_commit})
        add_custom_target(pulse_final_capture_guard ALL
            COMMAND ${CMAKE_COMMAND}
                "-DPULSE_SOURCE_DIR:PATH=${CMAKE_SOURCE_DIR}"
                "-DPULSE_EXPECTED_GIT_COMMIT:STRING=${_pulse_git_commit}"
                "-DPULSE_ARTIFACT_PATH:FILEPATH=${_pulse_final_artifact_path}"
                "-DPULSE_EXPECTED_ARTIFACT_SHA256:STRING=${_pulse_final_artifact_sha256}"
                -P ${CMAKE_SOURCE_DIR}/config/cmake/pulse_final_capture_guard.cmake
            COMMENT "Verifying immutable PULSE final-capture inputs"
            VERBATIM)
        add_dependencies(app pulse_final_capture_guard)
    else()
        set(_pulse_build_guard development_validation_only)
        # Retain the human-friendly dirty marker for non-paper validation
        # images so they cannot accidentally satisfy the strict hash schema.
        set(_pulse_firmware_revision ${_pulse_git_revision})
    endif()
    target_compile_definitions(app_src INTERFACE
        SENSWEAR_FIRMWARE_REVISION="${_pulse_firmware_revision}"
        SENSWEAR_PULSE_BUILD_GUARD="${_pulse_build_guard}")
else()
    # Normal out-of-box application and its Bluetooth GATT services.
    include(${APP_SRC_DIR}/bluetooth/services/services.cmake)
    target_sources(app_src INTERFACE ${APP_SRC_DIR}/app/oob_main.c)

    # ble_services carries GATT sources, service include directories, and the
    # common device-manager contract.
    target_link_libraries(app_src INTERFACE ble_services)
endif()
