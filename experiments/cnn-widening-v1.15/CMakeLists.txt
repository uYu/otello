cmake_minimum_required(VERSION 3.16)
project(otello_cnn_widening LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(OTELLO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
if(APPLE)
  execute_process(COMMAND xcrun --show-sdk-path OUTPUT_VARIABLE OTELLO_MACOS_SDK OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(EXISTS "${OTELLO_MACOS_SDK}/usr/include/c++/v1")
    include_directories(SYSTEM "${OTELLO_MACOS_SDK}/usr/include/c++/v1")
  endif()
endif()
add_library(othello SHARED
  ${OTELLO_ROOT}/cpp/search.cpp ${OTELLO_ROOT}/cpp/move_policy.cpp
  ${OTELLO_ROOT}/cpp/nnue.cpp ${OTELLO_ROOT}/cpp/cnn.cpp
  ${OTELLO_ROOT}/cpp/resnet.cpp ${CMAKE_CURRENT_SOURCE_DIR}/mcts.cpp
  ${OTELLO_ROOT}/cpp/default_engine.cpp ${OTELLO_ROOT}/cpp/api.cpp)
target_include_directories(othello PUBLIC ${OTELLO_ROOT}/cpp)
add_executable(otello_tests ${OTELLO_ROOT}/tests/native.cpp)
target_link_libraries(otello_tests PRIVATE othello)
enable_testing()
add_test(NAME rules_and_search COMMAND otello_tests)
