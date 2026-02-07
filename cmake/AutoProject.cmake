function(auto_project)
    get_filename_component(_proj_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)
    project(${_proj_name} LANGUAGES CXX CUDA)
    message(STATUS "[auto_project] Project name: ${_proj_name}")

    file(GLOB _srcs "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

    if(NOT _srcs)
        set(_main_cpp "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
        file(WRITE ${_main_cpp} "int main() { return 0; }\n")
        list(APPEND _srcs ${_main_cpp})
        message(STATUS "[auto_project] No source found, generated main.cpp")
    endif()

    add_executable(${_proj_name} ${_srcs})
    target_include_directories(${_proj_name} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(${_proj_name} PRIVATE CUDA::cudart)
    message(STATUS "[auto_project] Executable: ${_proj_name}")
endfunction()
