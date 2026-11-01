cmake_minimum_required(VERSION 3.16)
project(rkva_managed_store_authority LANGUAGES CXX)
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "Managed store authority fixture is Linux-only")
endif()
find_package(SQLite3 REQUIRED)
find_package(Threads REQUIRED)
include(CTest)
# Concrete lease + SQLite + fail-closed v3 object composition. This is the
# Phase 4 copy: the earlier PREPARE-SQLite variant targeted a newer
# anchored-sqlite revision (pending_request.native_header) that Phase 3 did not
# publish, so it is intentionally not built here.
add_executable(test-rkva-managed-store-authority ../../tests/test-rkva-managed-store-authority.cpp
    ../../ggml/src/ggml-remote-attn/rkva-managed-store-authority.cpp
    ../../ggml/src/ggml-remote-attn/rkva-anchored-sqlite.cpp
    ../xbox/uwp/rkva/rkva-managed-object-v3.cpp)
target_compile_features(test-rkva-managed-store-authority PRIVATE cxx_std_17)
target_include_directories(test-rkva-managed-store-authority PRIVATE ../../ggml/include ../../ggml/src/ggml-remote-attn)
target_link_libraries(test-rkva-managed-store-authority PRIVATE SQLite::SQLite3 Threads::Threads)
add_test(NAME test-rkva-managed-store-authority COMMAND test-rkva-managed-store-authority)
