cmake_minimum_required(VERSION 3.16)

# Buildable both standalone (cmake -S tools/rkva-native-key-inspect) and as
# part of the repository tools tree (add_subdirectory).
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(rkva_native_key_inspect LANGUAGES CXX)
    include(CTest)
    set(RKVA_KEY_INSPECT_STANDALONE ON)
    find_package(OpenSSL REQUIRED)
else()
    set(RKVA_KEY_INSPECT_STANDALONE OFF)
    # Optional dependency: the enclosing build keeps working without OpenSSL.
    find_package(OpenSSL QUIET)
    if(NOT OpenSSL_FOUND)
        message(STATUS "rkva-native-key-inspect: OpenSSL not found, tool disabled")
        return()
    endif()
endif()

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(RKVA_KEY_INSPECT_STANDALONE)
        message(FATAL_ERROR "This read-only file-mapping inspector is Linux-only")
    endif()
    return()
endif()

add_executable(rkva-native-key-inspect main.cpp native-key-inspect.cpp ../../src/llama-kv-mixed-state.cpp)
target_compile_features(rkva-native-key-inspect PRIVATE cxx_std_17)
target_include_directories(rkva-native-key-inspect PRIVATE ../../src ../../include ../../ggml/include)
target_link_libraries(rkva-native-key-inspect PRIVATE OpenSSL::Crypto)

if(RKVA_KEY_INSPECT_STANDALONE)
    set(RKVA_KEY_INSPECT_TESTS ${BUILD_TESTING})
else()
    set(RKVA_KEY_INSPECT_TESTS ${LLAMA_BUILD_TESTS})
endif()

if(RKVA_KEY_INSPECT_TESTS)
    # The regression binary takes an externally qualified snapshot path, so it
    # is built but not registered with CTest: there is no fixture in the tree.
    add_executable(test-rkva-native-key-inspect ../../tests/test-rkva-native-key-inspect.cpp ../../src/llama-kv-mixed-state.cpp)
    target_compile_features(test-rkva-native-key-inspect PRIVATE cxx_std_17)
    target_compile_definitions(test-rkva-native-key-inspect PRIVATE RKVA_NATIVE_KEY_INSPECT_TESTING)
    target_include_directories(test-rkva-native-key-inspect PRIVATE ../../src ../../include ../../ggml/include)
    target_link_libraries(test-rkva-native-key-inspect PRIVATE OpenSSL::Crypto)
endif()
