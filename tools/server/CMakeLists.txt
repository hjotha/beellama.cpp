include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# server-context containing the core server logic, used by llama-server and CLI

# F4 TEST-ONLY full-logit capture hook (f4-test-logit-capture.*). OFF by default
# and enabled ONLY for the build-f4 test tree, so it is never compiled into
# build-optimized / production binaries. It adds no configuration surface.
option(LLAMA_F4_TEST_LOGIT_CAPTURE "F4 test-only full-logit capture hook (never in production builds)" OFF)
set(F4_TEST_LOGIT_CAPTURE_SOURCES)
if (LLAMA_F4_TEST_LOGIT_CAPTURE)
    list(APPEND F4_TEST_LOGIT_CAPTURE_SOURCES f4-test-logit-capture.cpp f4-test-logit-capture.h)
endif()

set(TARGET server-context)

add_library(${TARGET} STATIC
    server-chat.cpp
    server-chat.h
    server-task.cpp
    server-task.h
    server-queue.cpp
    server-queue.h
    server-common.cpp
    server-common.h
    server-model-identity.cpp
    server-model-identity.h
    server-route-state.cpp
    server-route-state.h
    server-decision.cpp
    server-decision.h
    server-context.cpp
    server-context.h
    server-mixed-kv-handoff.cpp
    server-mixed-kv-handoff.h
server-loop-guard.cpp
    server-loop-guard.h
    server-gpu-power.cpp
    server-gpu-power.h
    server-stream.cpp
    server-stream.h
    server-tools.cpp
    server-tools.h
    server-mcp.cpp
    server-mcp.h
    server-schema.cpp
    server-schema.h
    ${F4_TEST_LOGIT_CAPTURE_SOURCES}
)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PRIVATE ../mtmd)

if (LLAMA_F4_TEST_LOGIT_CAPTURE)
    target_compile_definitions(${TARGET} PRIVATE LLAMA_F4_TEST_LOGIT_CAPTURE=1)
endif()

# F4 TEST-ONLY backend copy/dispatch counters (ggml-f4-counters.h). OFF by
# default; enabled only in build-f4, so nothing is added to build-optimized.
if (LLAMA_F4_TEST_COUNTERS)
    target_compile_definitions(${TARGET} PRIVATE LLAMA_F4_TEST_COUNTERS=1)
endif()
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common mtmd vendor::hash ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})
target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common mtmd ${CMAKE_THREAD_LIBS_INIT})

# llama-server-impl: server logic, reusable by app

set(TARGET llama-server-impl)

add_library(${TARGET}
    server.cpp
    server-http.cpp
    server-http.h
    server-models.cpp
    server-models.h
)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${TARGET} PRIVATE ../mtmd ${PROJECT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC server-context llama-ui cpp-httplib ${CMAKE_THREAD_LIBS_INIT})

add_dependencies(${TARGET} llama-ui-assets)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# llama-server executable

set(TARGET llama-server)

add_executable(${TARGET} main.cpp)
install(TARGETS ${TARGET} RUNTIME)

target_link_libraries(${TARGET} PRIVATE llama-server-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
