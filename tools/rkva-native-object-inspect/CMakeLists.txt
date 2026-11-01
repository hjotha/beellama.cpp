cmake_minimum_required(VERSION 3.16)

# Buildable both standalone (cmake -S tools/rkva-native-object-inspect) and as
# part of the repository tools tree (add_subdirectory).
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(rkva_native_object_inspect LANGUAGES CXX)
    include(CTest)
    set(RKVA_OBJECT_INSPECT_STANDALONE ON)
else()
    set(RKVA_OBJECT_INSPECT_STANDALONE OFF)
endif()

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(RKVA_OBJECT_INSPECT_STANDALONE)
        message(FATAL_ERROR "This read-only object inspector is Linux-only")
    endif()
    return()
endif()

add_executable(rkva-native-object-inspect main.cpp native-object-inspect.cpp)
target_compile_features(rkva-native-object-inspect PRIVATE cxx_std_17)
target_include_directories(rkva-native-object-inspect PRIVATE ../../ggml/src/ggml-remote-attn)

if(RKVA_OBJECT_INSPECT_STANDALONE)
    set(RKVA_OBJECT_INSPECT_TESTS ${BUILD_TESTING})
else()
    set(RKVA_OBJECT_INSPECT_TESTS ${LLAMA_BUILD_TESTS})
endif()

if(RKVA_OBJECT_INSPECT_TESTS)
    add_executable(test-rkva-native-object-inspect ../../tests/test-rkva-native-object-inspect.cpp native-object-inspect.cpp)
    target_compile_features(test-rkva-native-object-inspect PRIVATE cxx_std_17)
    target_compile_definitions(test-rkva-native-object-inspect PRIVATE RKVA_NATIVE_OBJECT_INSPECT_TESTING)
    target_include_directories(test-rkva-native-object-inspect PRIVATE ../../ggml/src/ggml-remote-attn)
    add_test(NAME test-rkva-native-object-inspect COMMAND test-rkva-native-object-inspect $<TARGET_FILE:rkva-native-object-inspect>)
    set_property(TEST test-rkva-native-object-inspect PROPERTY LABELS main)
endif()
