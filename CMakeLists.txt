cmake_minimum_required(VERSION 3.20)
project(loopcd CXX)
if(WIN32)
    enable_language(RC)   # UTF-8 argv via src/loopcd.rc + src/loopcd.manifest
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

include(FetchContent)

# Pin ggml to a known commit. Bump deliberately, never track a branch.
#
# M7 adds two ops (KITTY_QK / KITTY_SV) that are not upstream at this pin, so the change is
# carried as a patch applied right after checkout: pinned SHA + fixed patch is deterministic,
# needs no second repository, and reads as a reviewable diff. tools/ggml-kitty-ops.patch is
# the only delta from upstream ggml.
#
# Offline/speed: -DFETCHCONTENT_SOURCE_DIR_GGML=<a CLEAN checkout of the pin>. The patch is
# applied to whatever source directory is used, so that directory must not already contain
# the ops (pointing it at an already-patched tree will fail the patch step).
FetchContent_Declare(
    ggml
    GIT_REPOSITORY https://github.com/ggml-org/ggml.git
    GIT_TAG        ffa4e8b80930029a35991f94e7c8a93cd67730ab
    PATCH_COMMAND  ${CMAKE_COMMAND} -E chdir <SOURCE_DIR> git apply --whitespace=nowarn
                   ${CMAKE_CURRENT_SOURCE_DIR}/tools/ggml-kitty-ops.patch
)
# M2: Vulkan is the shipping backend. -DLOOPCD_VULKAN=OFF builds a CPU-only binary.
option(LOOPCD_VULKAN "loopcd: build the Vulkan backend (requires the Vulkan SDK)" ON)
if(LOOPCD_VULKAN)
    set(GGML_VULKAN ON  CACHE BOOL "" FORCE)
else()
    set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
endif()
set(GGML_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TOOLS     OFF CACHE BOOL "" FORCE)
set(GGML_CCACHE          OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(ggml)

add_executable(loopcd
    src/main.cpp
    src/model.cpp
    src/graph.cpp
    src/kv_cache.cpp
    src/kitty_kv.cpp
    src/loopcd.cpp
    src/unicode.cpp
    src/unicode-data.cpp
    src/tokenizer.cpp
    src/chat_template.cpp
    src/generate.cpp
    src/sampling.cpp
    src/server.cpp
    third_party/cpp-httplib/httplib.cpp
)
if(WIN32)
    target_sources(loopcd PRIVATE src/loopcd.rc)
endif()
target_include_directories(loopcd PRIVATE src)
target_link_libraries(loopcd PRIVATE ggml::ggml)

target_include_directories(loopcd PRIVATE third_party third_party/cpp-httplib)
if(WIN32)
    # cp-httplib 0.58.0 requires Windows 10+, and needs the Winsock import library.
    add_compile_definitions(_WIN32_WINNT=0x0A00)
    target_link_libraries(loopcd PRIVATE ws2_32)

    # Toolchain quirk (winlibs GCC 16.2.0): the driver auto-appends a CRT
    # `default-manifest.o` to every executable, which collides with the UTF-8
    # activeCodePage manifest embedded via src/loopcd.rc ("multiple non-default
    # manifests"). Patch the compiler's own specs at configure time to drop only
    # that auto-append, keeping our manifest (and hence UTF-8 argv).
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpspecs
                        OUTPUT_VARIABLE _gcc_specs RESULT_VARIABLE _specs_rc)
        if(_specs_rc EQUAL 0)
            string(REPLACE "%{!shared:%:if-exists(default-manifest.o%s)}" "" _gcc_specs "${_gcc_specs}")
            set(_spec_file "${CMAKE_CURRENT_BINARY_DIR}/no-default-manifest.spec")
            file(WRITE "${_spec_file}" "${_gcc_specs}")
            target_link_options(loopcd PRIVATE "-specs=${_spec_file}")
        endif()
    endif()
endif()

enable_testing()
add_executable(test_ops tests/unit/test_ops.cpp)
target_link_libraries(test_ops PRIVATE ggml::ggml)
add_test(NAME ops COMMAND test_ops)

add_executable(test_kv tests/unit/test_kv.cpp src/kv_cache.cpp)
target_include_directories(test_kv PRIVATE src)
target_link_libraries(test_kv PRIVATE ggml::ggml)
add_test(NAME kv COMMAND test_kv)

add_executable(test_loopcd tests/unit/test_loopcd.cpp src/loopcd.cpp)
target_include_directories(test_loopcd PRIVATE src)
target_link_libraries(test_loopcd PRIVATE ggml::ggml)
add_test(NAME loopcd COMMAND test_loopcd)

add_executable(test_sampling tests/unit/test_sampling.cpp src/sampling.cpp)
target_include_directories(test_sampling PRIVATE src)
target_link_libraries(test_sampling PRIVATE ggml::ggml)
add_test(NAME sampling COMMAND test_sampling)

add_executable(test_kitty tests/unit/test_kitty.cpp src/kv_cache.cpp src/kitty_kv.cpp)
target_include_directories(test_kitty PRIVATE src)
target_link_libraries(test_kitty PRIVATE ggml::ggml)
add_test(NAME kitty COMMAND test_kitty)
set_tests_properties(kitty PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

add_executable(test_cache_alloc tests/unit/test_cache_alloc.cpp src/kv_cache.cpp)
target_include_directories(test_cache_alloc PRIVATE src)
target_link_libraries(test_cache_alloc PRIVATE ggml::ggml)
add_test(NAME cache_alloc COMMAND test_cache_alloc)

add_executable(test_kitty_pack tests/unit/test_kitty_pack.cpp src/kv_cache.cpp src/kitty_kv.cpp)
target_include_directories(test_kitty_pack PRIVATE src)
target_link_libraries(test_kitty_pack PRIVATE ggml::ggml)
add_test(NAME kitty_pack COMMAND test_kitty_pack)
set_tests_properties(kitty_pack PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

add_executable(test_kitty_ops tests/unit/test_kitty_ops.cpp src/kv_cache.cpp src/kitty_kv.cpp)
target_include_directories(test_kitty_ops PRIVATE src)
target_link_libraries(test_kitty_ops PRIVATE ggml::ggml)
add_test(NAME kitty_ops COMMAND test_kitty_ops)

add_executable(test_kitty_equiv tests/unit/test_kitty_equiv.cpp src/kv_cache.cpp src/kitty_kv.cpp)
target_include_directories(test_kitty_equiv PRIVATE src)
target_link_libraries(test_kitty_equiv PRIVATE ggml::ggml)
add_test(NAME kitty_equiv COMMAND test_kitty_equiv)

add_executable(test_tokenizer tests/unit/test_tokenizer.cpp src/tokenizer.cpp src/unicode.cpp src/unicode-data.cpp)
target_include_directories(test_tokenizer PRIVATE src)
target_link_libraries(test_tokenizer PRIVATE ggml::ggml)
add_test(NAME tokenizer COMMAND test_tokenizer ${CMAKE_SOURCE_DIR}/models/ouro/ouro.f16.gguf)
