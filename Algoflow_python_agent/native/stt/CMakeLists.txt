cmake_minimum_required(VERSION 3.22)
project(local_stt_runtime LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.12.0
)

FetchContent_Declare(
  whispercpp
  GIT_REPOSITORY https://github.com/ggml-org/whisper.cpp.git
  GIT_TAG v1.7.6
)

set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(pybind11 whispercpp)

pybind11_add_module(_local_stt whisper_runtime.cpp)
target_link_libraries(_local_stt PRIVATE whisper)
target_compile_definitions(_local_stt PRIVATE PYBIND11_DETAILED_ERROR_MESSAGES)
