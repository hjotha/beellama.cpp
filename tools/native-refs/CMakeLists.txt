set(TARGET llama-native-refs)

add_executable(${TARGET} native-refs.cpp)
target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
target_link_libraries(${TARGET} PRIVATE server-context)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
