cmake_minimum_required(VERSION 3.18)
project(trt_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TRT_DIR "/opt/TensorRT-10.16.1.11")
set(CUDAToolkit_ROOT "/usr/local/cuda-12.9")

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core highgui imgcodecs imgproc)

add_executable(trt_engine
    app/main.cpp
    app/viewer.cpp
    lib/src/inference_engine.cpp
    lib/src/logger.cpp
)

target_include_directories(trt_engine PRIVATE
    app
    lib/include
    ${TRT_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(trt_engine PRIVATE
    CUDA::cudart
    ${OpenCV_LIBS}
    ${TRT_DIR}/lib/libnvinfer.so
    ${TRT_DIR}/lib/libnvinfer_plugin.so
)
