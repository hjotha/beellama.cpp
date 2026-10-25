cmake_minimum_required(VERSION 3.22)
# Built from the BeeLlama root; all frontends share its existing llama target.
if(NOT TARGET llama)
    message(FATAL_ERROR "Configure the BeeLlama root with -DLLAMA_BUILD_XBOX_HOST=ON")
endif()
set(XLLAMA_CORE_DIR "${PROJECT_SOURCE_DIR}")

# ------------------------------------------------------------------
# Options
# ------------------------------------------------------------------
set(XLLAMA_TARGET "linux" CACHE STRING "Build target: linux or uwp")
set_property(CACHE XLLAMA_TARGET PROPERTY STRINGS linux uwp)

option(XLLAMA_BUILD_TESTS    "Build unit tests" OFF)
option(XLLAMA_ENABLE_ASAN    "Enable AddressSanitizer" OFF)
option(XLLAMA_ENABLE_UBSAN   "Enable UndefinedBehaviourSanitizer" OFF)
option(XLLAMA_BUILD_PROBES   "Build benchmarking probes (membw, diskbw, gpubw, gpugemv, gpustep, ramceil)" ON)

# ------------------------------------------------------------------
# Target selection
# ------------------------------------------------------------------
if(XLLAMA_TARGET STREQUAL "uwp")
    message(FATAL_ERROR
        "XLLAMA_TARGET=uwp is not supported via CMake directly.\n"
        "UWP packaging requires MSVC and the Windows SDK.\n"
        "On Windows, run: .\\scripts\\build-uwp.ps1\n"
        "See docs/uwp-constraints.md for details.")
endif()

if(NOT XLLAMA_TARGET STREQUAL "linux")
    message(FATAL_ERROR "Unknown XLLAMA_TARGET '${XLLAMA_TARGET}'. Valid values: linux, uwp.")
endif()

message(STATUS "xllama: building for target=linux")

# ------------------------------------------------------------------
# xllama library (shared between CLI and UWP)
# ------------------------------------------------------------------
add_library(xllama STATIC
    src/bridge/inference.cpp
    src/bridge/session.cpp
    src/bridge/mtp_draft.cpp
    src/bridge/embedding.cpp
    src/bridge/kv_store.cpp
    src/bridge/bench.cpp
    src/bridge/platform.cpp
    src/bridge/path_utils.cpp
    src/bridge/utf8_utils.cpp
    src/bridge/chat_prompt.cpp
    src/bridge/prompt_budget.cpp
    src/bridge/cli.cpp
    src/bridge/training.cpp
    src/bridge/device_train.cpp
    src/bridge/preference_capture.cpp
    src/bridge/personalize.cpp
    src/bridge/autopilot.cpp
    src/bridge/capture_probe.cpp
    src/bridge/d3d12_compute.cpp
    src/bridge/ggml_d3d12.cpp
)

# Optional benchmarking probes (membw, diskbw, gpubw, gpugemv, gpustep, ramceil)
if(XLLAMA_BUILD_PROBES)
    target_sources(xllama PRIVATE
        src/bridge/membw.cpp
        src/bridge/diskbw.cpp
        src/bridge/gpubw.cpp
        src/bridge/gpugemv.cpp
        src/bridge/gpustep.cpp
        src/bridge/tttarget.cpp
        src/bridge/ramceil.cpp
        src/bridge/replay.cpp
        src/bridge/diverge.cpp
        src/bridge/termgate.cpp
    )
endif()

target_compile_features(xllama PUBLIC cxx_std_17)

target_include_directories(xllama PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${XLLAMA_CORE_DIR}/include
    ${XLLAMA_CORE_DIR}/common
)
# The d3d12 ggml backend implements ggml-backend-impl.h, which ggml keeps internal.
if(WIN32)
    target_include_directories(xllama PRIVATE ${XLLAMA_CORE_DIR}/ggml/src)
endif()
# AOT DXIL blobs for Phase 15 gpubw (#211) / gpugemv (#228); host builds do not link D3D12.
target_include_directories(xllama PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/generated
)

target_link_libraries(xllama PUBLIC llama)

target_compile_definitions(xllama PUBLIC
    XLLAMA_LINUX=1
    XLLAMA_USE_LLAMA=1
    XLLAMA_DEVICE_TRAIN=1
)

# Sanitizers (Linux / Clang/GCC only)
if(XLLAMA_ENABLE_ASAN)
    target_compile_options(xllama PUBLIC -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(xllama PUBLIC -fsanitize=address)
endif()
if(XLLAMA_ENABLE_UBSAN)
    target_compile_options(xllama PUBLIC -fsanitize=undefined)
    target_link_options(xllama PUBLIC -fsanitize=undefined)
endif()

# ------------------------------------------------------------------
# xllama-cli
# ------------------------------------------------------------------
add_executable(xllama-cli src/main.cpp)

target_link_libraries(xllama-cli PRIVATE xllama)

target_compile_options(xllama-cli PRIVATE
    -fno-omit-frame-pointer
)

set_target_properties(xllama-cli PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ------------------------------------------------------------------
# Install
# ------------------------------------------------------------------
install(TARGETS xllama-cli xllama
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------
if(XLLAMA_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    enable_testing()
    add_subdirectory(tests)
endif()

message(STATUS "xllama: xllama-cli target configured — build with: cmake --build . -j")
