# Automatically create a test executable for each test_*.cpp file
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test)

file(GLOB TEST_SOURCES test_*.cpp)

foreach(test_src ${TEST_SOURCES})
  get_filename_component(test_name ${test_src} NAME_WE)
  add_executable(${test_name} ${test_src})
  target_link_libraries(${test_name} gtest_main cuda)
  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
