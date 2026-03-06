# Encapsulate per-operator CUDA shared library creation
function(auto_cuda_operators_library OPERATORS_DIR)
  file(GLOB OPERATORS_CU_FILES "${OPERATORS_DIR}/*.cu")

  foreach(CU_FILE ${OPERATORS_CU_FILES})
    get_filename_component(OP_NAME ${CU_FILE} NAME_WE)
    set(LIB_NAME "libcuda_lab_${OP_NAME}")
    add_library(${LIB_NAME} SHARED ${CU_FILE})
    set_target_properties(${LIB_NAME} PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      POSITION_INDEPENDENT_CODE ON
      OUTPUT_NAME "cuda_lab_${OP_NAME}"
      PREFIX "lib"
    )
    target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(${LIB_NAME} PUBLIC CUDA::cudart)
  endforeach()
endfunction()
