function(auto_cuda_project)
    get_filename_component(_proj_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)
    project(${_proj_name} LANGUAGES CXX CUDA)
    message(STATUS "[auto_cuda_project] Project name: ${_proj_name}")

    file(GLOB _srcs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
    list(TRANSFORM _srcs PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

    if(NOT _srcs)
        set(_main_cpp "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
        file(WRITE ${_main_cpp} "int main() { return 0; }\n")
        list(APPEND _srcs ${_main_cpp})
        message(STATUS "[auto_cuda_project] No source found, generated main.cpp")
    endif()

    add_executable(${_proj_name} ${_srcs})
    target_include_directories(${_proj_name} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(${_proj_name} PRIVATE CUDA::cudart)
    message(STATUS "[auto_cuda_project] Executable: ${_proj_name}")
endfunction()

function(auto_subdirs parent_dir)
    file(GLOB children RELATIVE ${parent_dir} ${parent_dir}/*)

    foreach(child ${children})
        if(IS_DIRECTORY ${parent_dir}/${child})
            if(NOT EXISTS ${parent_dir}/${child}/CMakeLists.txt)
                file(WRITE ${parent_dir}/${child}/CMakeLists.txt
                    "include(\${CMAKE_SOURCE_DIR}/cmake/AutoProject.cmake)\nauto_cuda_project()\n")
            endif()

            add_subdirectory(${child})
        endif()
    endforeach()
endfunction()
