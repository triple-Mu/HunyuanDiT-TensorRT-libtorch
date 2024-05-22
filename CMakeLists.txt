cmake_minimum_required(VERSION 3.18)
project(py_hunyuan_dit VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# CUDA
find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    get_filename_component(CUDA_LIBS_DIRS ${CUDA_LIBRARIES} DIRECTORY)
endif ()

# Python
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "from __future__ import print_function;import torch;print(torch.utils.cmake_prefix_path, end='')"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE TORCH_CMAKE
)

get_filename_component(TMP_DIR "${TORCH_CMAKE}" DIRECTORY)
get_filename_component(TORCH_ROOT "${TMP_DIR}" DIRECTORY)
set(Torch_LIBS_DIRS "${TORCH_ROOT}/lib")

set(Torch_DIR "${TORCH_CMAKE}/Torch")
find_package(Torch REQUIRED)
if (Torch_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
endif ()

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
find_package(TensorRT REQUIRED)

list(APPEND ALL_INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${TORCH_INCLUDE_DIRS}
        ${TensorRT_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
)

list(APPEND ALL_DIRS
        ${TensorRT_LIBS_DIRS}
        ${Torch_LIBS_DIRS}
)

list(APPEND ALL_LIBS
        ${Python3_LIBRARIES}
        ${TensorRT_LIBRARIES}
        torch
        torch_python
)

add_library(${PROJECT_NAME} SHARED csrc/hunyuan.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX ""
        SUFFIX ".cpython-${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}-${CMAKE_SYSTEM_PROCESSOR}-linux-gnu.so")
target_include_directories(${PROJECT_NAME} PRIVATE ${ALL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} "${ALL_LIBS}")
target_link_directories(${PROJECT_NAME} PRIVATE ${ALL_DIRS})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

