cmake_minimum_required(VERSION 3.10)

project(cuImageProcessor)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include/cuImageProcessor)

set(CUDA_SOURCES
    grayscale.cu
    resize.cu
)

set(CPP_SOURCES
    stb_image_wrapper.cpp
)

add_library(${PROJECT_NAME} SHARED ${CUDA_SOURCES} ${CPP_SOURCES})

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
