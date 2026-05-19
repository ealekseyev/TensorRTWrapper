cmake_minimum_required(VERSION 3.18)
project(trt_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Platform-specific defaults (can still be overridden via -DTRT_DIR=... -DCUDAToolkit_ROOT=...)
if(WIN32)
    set(TRT_DIR "C:/TensorRT-10.16.1.11" CACHE PATH "TensorRT root directory")
    set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9" CACHE PATH "CUDA Toolkit root")
elseif(UNIX)
    set(TRT_DIR "/opt/TensorRT-10.16.1.11" CACHE PATH "TensorRT root directory")
    set(CUDAToolkit_ROOT "/usr/local/cuda-12.9" CACHE PATH "CUDA Toolkit root")
else()
    message(FATAL_ERROR "Unsupported platform")
endif()

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core highgui imgcodecs imgproc)

# TensorRT library discovery (cross-platform)
if(WIN32)
    set(TRT_LIB_HINTS "${TRT_DIR}/lib" "${TRT_DIR}/lib/x64")
else()
    set(TRT_LIB_HINTS "${TRT_DIR}/lib" "${TRT_DIR}/lib64")
endif()

find_library(TRT_NVINFER_LIB
    NAMES nvinfer
    HINTS ${TRT_LIB_HINTS}
    REQUIRED
)

find_library(TRT_NVINFER_PLUGIN_LIB
    NAMES nvinfer_plugin
    HINTS ${TRT_LIB_HINTS}
    REQUIRED
)

add_executable(trt_engine
    app/main.cpp
    app/image_utils.cpp
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
    ${TRT_NVINFER_LIB}
    ${TRT_NVINFER_PLUGIN_LIB}
)