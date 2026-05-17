#!/usr/bin/env python3

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

import tensorrt as trt
from cuda import cudart


TARGET_H = 640
TARGET_W = 640


def check_cuda(status, message):
    if status != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{message}: {status!r}")


def letterbox_bgr(image: np.ndarray, target_width: int = TARGET_W, target_height: int = TARGET_H) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(target_width / width, target_height / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))

    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    pad_x = (target_width - resized_width) // 2
    pad_y = (target_height - resized_height) // 2
    canvas[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = resized
    return canvas


def preprocess_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to open image: {image_path}")

    letterboxed = letterbox_bgr(image)
    rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return np.ascontiguousarray(chw[np.newaxis, ...])


def volume(shape) -> int:
    size = 1
    for dim in shape:
        size *= int(dim)
    return size


def resolve_shape(engine: trt.ICudaEngine, context: trt.IExecutionContext, tensor_name: str):
    shape = tuple(engine.get_tensor_shape(tensor_name))
    if any(dim < 0 for dim in shape):
        min_shape, opt_shape, max_shape = engine.get_tensor_profile_shape(tensor_name, 0)
        context.set_input_shape(tensor_name, opt_shape)
        shape = tuple(context.get_tensor_shape(tensor_name))
    return tuple(int(dim) for dim in shape)


def benchmark(engine_path: Path, image_path: Path, iterations: int, warmup: int) -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    engine_bytes = engine_path.read_bytes()
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create TensorRT execution context")

    stream_status, stream = cudart.cudaStreamCreate()
    check_cuda(stream_status, "cudaStreamCreate failed")

    input_tensor = preprocess_image(image_path)

    tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    input_names = [name for name in tensor_names if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT]
    output_names = [name for name in tensor_names if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT]
    if len(input_names) != 1:
        raise RuntimeError(f"Expected exactly one input tensor, got {len(input_names)}")

    input_name = input_names[0]
    input_shape = resolve_shape(engine, context, input_name)
    if tuple(input_tensor.shape) != input_shape:
        raise RuntimeError(f"Input tensor shape mismatch: preprocessed image is {input_tensor.shape}, engine expects {input_shape}")

    device_buffers = {}
    host_outputs = {}

    try:
        for name in tensor_names:
            shape = resolve_shape(engine, context, name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            nbytes = volume(shape) * np.dtype(dtype).itemsize

            status, device_ptr = cudart.cudaMalloc(nbytes)
            check_cuda(status, f"cudaMalloc failed for tensor {name}")
            device_buffers[name] = (device_ptr, nbytes)
            context.set_tensor_address(name, int(device_ptr))

            if name in output_names:
                host_outputs[name] = np.empty(shape, dtype=dtype)

        input_ptr, input_nbytes = device_buffers[input_name]

        def run_once():
            status = cudart.cudaMemcpyAsync(
                input_ptr,
                input_tensor.ctypes.data,
                input_nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                stream,
            )[0]
            check_cuda(status, "cudaMemcpyAsync H2D failed")

            if not context.execute_async_v3(stream):
                raise RuntimeError("TensorRT execute_async_v3 failed")

            for output_name in output_names:
                output_ptr, output_nbytes = device_buffers[output_name]
                host_output = host_outputs[output_name]
                status = cudart.cudaMemcpyAsync(
                    host_output.ctypes.data,
                    output_ptr,
                    output_nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    stream,
                )[0]
                check_cuda(status, f"cudaMemcpyAsync D2H failed for tensor {output_name}")

            check_cuda(cudart.cudaStreamSynchronize(stream)[0], "cudaStreamSynchronize failed")

        for _ in range(warmup):
            run_once()

        start = time.perf_counter()
        for _ in range(iterations):
            run_once()
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / iterations) * 1000.0
        fps = iterations / elapsed
        print(f"Engine: {engine_path}")
        print(f"Image:  {image_path}")
        print(f"Warmup: {warmup}")
        print(f"Runs:   {iterations}")
        print(f"Total:  {elapsed:.4f} s")
        print(f"Avg:    {avg_ms:.3f} ms")
        print(f"FPS:    {fps:.2f}")
    finally:
        for device_ptr, _ in device_buffers.values():
            cudart.cudaFree(device_ptr)
        cudart.cudaStreamDestroy(stream)


def main():
    parser = argparse.ArgumentParser(description="Benchmark a TensorRT .trt engine on one in-memory image tensor.")
    parser.add_argument("image", type=Path, help="Input image path")
    parser.add_argument("--engine", type=Path, default=Path("models/trt/model.trt"), help="TensorRT engine path")
    parser.add_argument("--iterations", type=int, default=500, help="Measured inference iterations")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    args = parser.parse_args()

    benchmark(args.engine, args.image, args.iterations, args.warmup)


if __name__ == "__main__":
    main()
