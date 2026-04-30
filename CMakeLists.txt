cmake_minimum_required(VERSION 3.18)
project(trt_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TRT_DIR "C:/nvidia/TensorRT-10.15.1.29")
set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9")

find_package(CUDAToolkit REQUIRED)

add_executable(trt_engine
    app/main.cpp
    lib/src/inference_engine.cpp
    lib/src/logger.cpp
)

target_include_directories(trt_engine PRIVATE
    lib/include
    ${TRT_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(trt_engine PRIVATE
    CUDA::cudart
    ${TRT_DIR}/lib/nvinfer_10.lib
    ${TRT_DIR}/lib/nvinfer_plugin_10.lib
)
