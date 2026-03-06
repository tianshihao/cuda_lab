set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test)

find_package(GTest REQUIRED)
find_package(CUDAToolkit REQUIRED)

file(GLOB TEST_SOURCES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

foreach(test_src ${TEST_SOURCES})
	get_filename_component(test_name ${test_src} NAME_WE)
	add_executable(${test_name} ${test_src})
	target_include_directories(${test_name} PRIVATE ${CUDAToolkit_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
	target_link_libraries(${test_name} gtest_main GTest::gtest GTest::gtest_main CUDA::cudart libcuda_lab_device_fill libcuda_lab_add_one)
	add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
