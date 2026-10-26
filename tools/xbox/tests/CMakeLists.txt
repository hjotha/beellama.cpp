include(FetchContent)
# doctest 2.4.11 supports these policies but declares an older CMake floor.
# Scope the CMake 4 compatibility override to this test subdirectory.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 4.0 AND NOT DEFINED CMAKE_POLICY_VERSION_MINIMUM)
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
endif()

FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG v2.4.11
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(doctest)

add_executable(xllama-tests
    test_cli.cpp
    test_utf8.cpp
    test_path.cpp
    test_platform.cpp
    test_bench.cpp
    test_session.cpp
    test_kv_store.cpp
    test_chat_history.cpp
    test_diffusion.cpp
    test_routing_policy.cpp
    test_cancel_policy.cpp
    test_sampling.cpp
    test_decode_loop.cpp
    test_profile_policy.cpp
    test_chat_prompt.cpp
    test_prompt_budget.cpp
    test_manifest_merge.cpp
    test_llama_ini.cpp
    test_membw.cpp
    test_diskbw.cpp
    test_gpubw.cpp
    test_gpupeak.cpp
    test_gpugemv.cpp
    test_gpustep.cpp
    test_ggml_d3d12.cpp
    test_replay.cpp
    test_diverge.cpp
    test_termgate.cpp
    test_ramceil.cpp
    test_model_provision.cpp
    test_model_write.cpp
    test_logit_parity.cpp
    test_training.cpp
    test_device_train.cpp
    test_json_utils.cpp
    test_preference_capture.cpp
    test_personalize.cpp
    test_api_config.cpp
    test_api_pull.cpp
    test_embedding.cpp
    test_autopilot.cpp
    test_speculative.cpp
    test_capture_probe.cpp
)

target_link_libraries(xllama-tests PRIVATE xllama doctest::doctest)

# Diffusion tests use the vendored nlohmann/json to load golden vectors, and are
# pointed at the in-repo golden file + CLIP tokenizer assets via compile defs.
target_include_directories(xllama-tests PRIVATE
    ${XLLAMA_CORE_DIR}/vendor)
target_compile_definitions(xllama-tests PRIVATE
    GOLDEN_JSON_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../diffusion/golden_vectors.json"
    CLIP_TOKENIZER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../diffusion/clip_tokenizer"
    PARITY_GOLDEN_BIN="${CMAKE_CURRENT_SOURCE_DIR}/golden/logits-smol-short.bin")

add_test(NAME xllama-tests COMMAND xllama-tests)

# Pure-Python roofline model for the GPU ceiling suite. No hardware, no numpy:
# it pins the pass1 FLOP/byte accounting and the probe-JSON contract.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_test(NAME xllama-ceiling-roofline
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_rkva_ceiling_roofline.py)
    add_test(NAME xllama-gpupeak-stream-model
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_gpupeak_stream_model.py)
    add_test(NAME xllama-gpupeak-dxil-fresh
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/compile-gpupeak-shaders.py --check)
endif()
