cmake_minimum_required(VERSION 3.10)

project(cuImageProcessor LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

include_directories(${PROJECT_SOURCE_DIR}/include/cuImageProcessor)

set(CUDA_SOURCES
    resize.cu
    grayscale.cu
    crop.cu
    rotate.cu
    flip.cu
)

add_library(${PROJECT_NAME} SHARED ${CUDA_SOURCES})

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
