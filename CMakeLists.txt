cmake_minimum_required(VERSION 3.20)
project(loopcd CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

include(FetchContent)

# Pin ggml to a known commit. Bump deliberately, never track a branch.
# Offline/speed: -DFETCHCONTENT_SOURCE_DIR_GGML=C:/Users/tnmh/projects/ggml
FetchContent_Declare(
    ggml
    GIT_REPOSITORY https://github.com/ggml-org/ggml.git
    GIT_TAG        ffa4e8b80930029a35991f94e7c8a93cd67730ab
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
    src/loopcd.cpp
)
target_include_directories(loopcd PRIVATE src)
target_link_libraries(loopcd PRIVATE ggml::ggml)

target_include_directories(loopcd PRIVATE third_party third_party/cpp-httplib)
if(WIN32)
    # cp-httplib 0.58.0 requires Windows 10+, and needs the Winsock import library.
    add_compile_definitions(_WIN32_WINNT=0x0A00)
    target_link_libraries(loopcd PRIVATE ws2_32)
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
