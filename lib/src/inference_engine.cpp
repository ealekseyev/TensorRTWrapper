#include "inference_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

std::size_t volume(const nvinfer1::Dims& dims) {
    std::size_t result = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) {
            throw std::runtime_error("Tensor shape contains an unresolved dimension");
        }
        result *= static_cast<std::size_t>(dims.d[i]);
    }
    return result;
}

std::size_t element_size(nvinfer1::DataType type) {
    switch (type) {
        case nvinfer1::DataType::kFLOAT: return sizeof(float);
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT32: return sizeof(std::int32_t);
        case nvinfer1::DataType::kINT8: return sizeof(std::int8_t);
        case nvinfer1::DataType::kBOOL: return sizeof(bool);
        default: throw std::runtime_error("Unsupported TensorRT tensor data type");
    }
}

void throw_if_cuda_failed(cudaError_t status, const std::string& action) {
    if (status != cudaSuccess) {
        throw std::runtime_error(action + ": " + cudaGetErrorString(status));
    }
}

TensorSpec make_tensor_spec(nvinfer1::ICudaEngine& engine,
                            const nvinfer1::Dims& dims,
                            const char* name,
                            std::size_t batch_size) {
    TensorSpec spec;
    spec.name = name;
    spec.dims = dims;
    spec.element_bytes = element_size(engine.getTensorDataType(name));
    spec.total_elements = volume(dims);
    spec.max_batch_size = std::max<std::size_t>(batch_size, 1);
    if (spec.total_elements % spec.max_batch_size != 0) {
        throw std::runtime_error("Tensor shape is not divisible by batch size for tensor " + spec.name);
    }
    spec.sample_elements = spec.total_elements / spec.max_batch_size;
    spec.total_bytes = spec.total_elements * spec.element_bytes;
    spec.sample_bytes = spec.sample_elements * spec.element_bytes;
    return spec;
}

}  // namespace

InferenceEngine::InferenceEngine(Logger::Severity log_level, std::size_t queue_batch_size)
    : _logger(log_level), _queue_batch_size(std::max<std::size_t>(queue_batch_size, 1)) {}

InferenceEngine::~InferenceEngine() {
    if (_inference_thread.joinable()) {
        {
            std::lock_guard<std::mutex> lock(_queue_mutex);
            _stop_requested = true;
        }
        _work_available_cv.notify_one();
        _inference_thread.join();
    }

    for (auto& tensor : _input_buffers) {
        if (tensor.device_ptr != nullptr) {
            cudaFree(tensor.device_ptr);
        }
    }
    for (auto& tensor : _output_buffers) {
        if (tensor.device_ptr != nullptr) {
            cudaFree(tensor.device_ptr);
        }
    }
    if (_stream != nullptr) {
        cudaStreamDestroy(_stream);
    }

    delete _context;
    delete _engine;
    delete _runtime;
}

void InferenceEngine::load_model(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Cannot open engine: " + path);
    }

    const std::size_t file_size = static_cast<std::size_t>(file.tellg());
    file.seekg(0);
    std::vector<char> engine_data(file_size);
    file.read(engine_data.data(), static_cast<std::streamsize>(engine_data.size()));
    if (!file) {
        throw std::runtime_error("Failed to read engine: " + path);
    }

    _runtime = nvinfer1::createInferRuntime(_logger);
    if (_runtime == nullptr) {
        throw std::runtime_error("createInferRuntime failed");
    }

    _engine = _runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    if (_engine == nullptr) {
        throw std::runtime_error("deserializeCudaEngine failed");
    }

    _context = _engine->createExecutionContext();
    if (_context == nullptr) {
        throw std::runtime_error("createExecutionContext failed");
    }

    throw_if_cuda_failed(cudaStreamCreate(&_stream), "cudaStreamCreate failed");

    _input_specs.clear();
    _output_specs.clear();
    _input_buffers.clear();
    _output_buffers.clear();

    const int io_count = _engine->getNbIOTensors();

    for (int i = 0; i < io_count; ++i) {
        const char* name = _engine->getIOTensorName(i);
        if (_engine->getTensorIOMode(name) != nvinfer1::TensorIOMode::kINPUT) {
            continue;
        }

        nvinfer1::Dims dims = _engine->getTensorShape(name);
        bool has_dynamic_dimension = false;
        for (int d = 0; d < dims.nbDims; ++d) {
            if (dims.d[d] < 0) {
                has_dynamic_dimension = true;
                break;
            }
        }

        if (has_dynamic_dimension) {
            dims = _engine->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kOPT);
            if (!_context->setInputShape(name, dims)) {
                throw std::runtime_error("setInputShape failed for input tensor: " + std::string(name));
            }
        }

        if (_input_specs.empty() && dims.nbDims > 0) {
            if (dims.d[0] <= 0) {
                throw std::runtime_error("Input tensor has an invalid batch dimension");
            }
            _engine_batch_size = std::max<std::size_t>(1, static_cast<std::size_t>(dims.d[0]));
        }

        TensorSpec spec = make_tensor_spec(*_engine, dims, name, _engine_batch_size);
        if (spec.element_bytes != sizeof(float)) {
            throw std::runtime_error("Only float input tensors are currently supported: " + spec.name);
        }
        void* device_ptr = nullptr;
        throw_if_cuda_failed(cudaMalloc(&device_ptr, spec.total_bytes),
                             "cudaMalloc failed for input tensor " + spec.name);
        if (!_context->setTensorAddress(name, device_ptr)) {
            throw std::runtime_error("setTensorAddress failed for input tensor: " + spec.name);
        }

        _input_specs.push_back(spec);
        _input_buffers.push_back({spec, device_ptr});
    }

    for (int i = 0; i < io_count; ++i) {
        const char* name = _engine->getIOTensorName(i);
        if (_engine->getTensorIOMode(name) != nvinfer1::TensorIOMode::kOUTPUT) {
            continue;
        }

        const nvinfer1::Dims dims = _context->getTensorShape(name);
        TensorSpec spec = make_tensor_spec(*_engine, dims, name, _engine_batch_size);
        if (spec.element_bytes != sizeof(float)) {
            throw std::runtime_error("Only float output tensors are currently supported: " + spec.name);
        }
        void* device_ptr = nullptr;
        throw_if_cuda_failed(cudaMalloc(&device_ptr, spec.total_bytes),
                             "cudaMalloc failed for output tensor " + spec.name);
        if (!_context->setTensorAddress(name, device_ptr)) {
            throw std::runtime_error("setTensorAddress failed for output tensor: " + spec.name);
        }

        _output_specs.push_back(spec);
        _output_buffers.push_back({spec, device_ptr});
    }

    _model_loaded = true;
}

void InferenceEngine::start() {
    _ensure_worker_started();
}

InferenceFuture InferenceEngine::submit(TensorData input) {
    return submit(TensorList{std::move(input)});
}

InferenceFuture InferenceEngine::submit(TensorList inputs) {
    if (!_model_loaded) {
        throw std::runtime_error("submit called before load_model");
    }

    _validate_inputs(inputs);
    _ensure_worker_started();

    PendingRequest request;
    request.inputs = std::move(inputs);
    InferenceFuture future = request.promise.get_future();

    {
        std::lock_guard<std::mutex> lock(_queue_mutex);
        _work_queue.push(std::move(request));
    }
    _work_available_cv.notify_one();

    return future;
}

TensorList InferenceEngine::infer(TensorData input) {
    return infer(TensorList{std::move(input)});
}

TensorList InferenceEngine::infer(TensorList inputs) {
    InferenceFuture future = submit(std::move(inputs));
    return future.get();
}

void InferenceEngine::_ensure_worker_started() {
    if (_worker_started) {
        return;
    }

    std::lock_guard<std::mutex> lock(_queue_mutex);
    if (_worker_started) {
        return;
    }

    _stop_requested = false;
    _inference_thread = std::thread(&InferenceEngine::_inference_worker, this);
    _worker_started = true;
}

void InferenceEngine::_validate_inputs(const TensorList& inputs) const {
    if (inputs.size() != _input_specs.size()) {
        throw std::runtime_error("Expected " + std::to_string(_input_specs.size()) +
                                 " input tensors, got " + std::to_string(inputs.size()));
    }

    for (std::size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].size() != _input_specs[i].sample_elements) {
            throw std::runtime_error("Input tensor " + _input_specs[i].name +
                                     " expects " + std::to_string(_input_specs[i].sample_elements) +
                                     " float values, got " + std::to_string(inputs[i].size()));
        }
    }
}

TensorList InferenceEngine::_run_inference_batch(const std::vector<PendingRequest>& batch) {
    const std::size_t actual_batch_size = batch.size();
    if (actual_batch_size == 0) {
        return {};
    }

    for (std::size_t tensor_index = 0; tensor_index < _input_buffers.size(); ++tensor_index) {
        const DeviceTensor& tensor = _input_buffers[tensor_index];
        throw_if_cuda_failed(cudaMemsetAsync(tensor.device_ptr, 0, tensor.spec.total_bytes, _stream),
                             "cudaMemsetAsync failed for input tensor " + tensor.spec.name);

        for (std::size_t batch_index = 0; batch_index < actual_batch_size; ++batch_index) {
            const TensorData& host_input = batch[batch_index].inputs[tensor_index];
            const auto* source = reinterpret_cast<const std::byte*>(host_input.data());
            auto* destination = static_cast<std::byte*>(tensor.device_ptr) +
                                (batch_index * tensor.spec.sample_bytes);

            throw_if_cuda_failed(cudaMemcpyAsync(destination,
                                                 source,
                                                 tensor.spec.sample_bytes,
                                                 cudaMemcpyHostToDevice,
                                                 _stream),
                                 "cudaMemcpyAsync failed for input tensor " + tensor.spec.name);
        }
    }

    if (!_context->enqueueV3(_stream)) {
        throw std::runtime_error("enqueueV3 failed");
    }

    TensorList flat_outputs(_output_buffers.size());
    for (std::size_t tensor_index = 0; tensor_index < _output_buffers.size(); ++tensor_index) {
        const DeviceTensor& tensor = _output_buffers[tensor_index];
        flat_outputs[tensor_index].resize(tensor.spec.sample_elements * actual_batch_size);

        throw_if_cuda_failed(cudaMemcpyAsync(flat_outputs[tensor_index].data(),
                                             tensor.device_ptr,
                                             tensor.spec.sample_bytes * actual_batch_size,
                                             cudaMemcpyDeviceToHost,
                                             _stream),
                             "cudaMemcpyAsync failed for output tensor " + tensor.spec.name);
    }

    throw_if_cuda_failed(cudaStreamSynchronize(_stream), "cudaStreamSynchronize failed");
    return flat_outputs;
}

void InferenceEngine::_inference_worker() {
    constexpr auto kBatchCollectWindow = std::chrono::milliseconds(2);

    while (true) {
        std::vector<PendingRequest> batch;

        {
            std::unique_lock<std::mutex> lock(_queue_mutex);
            _work_available_cv.wait(lock, [this]() {
                return _stop_requested || !_work_queue.empty();
            });

            if (_stop_requested && _work_queue.empty()) {
                break;
            }

            const std::size_t target_batch_size = std::min(_queue_batch_size, _engine_batch_size);
            if (!_stop_requested && _work_queue.size() < target_batch_size) {
                _work_available_cv.wait_for(lock,
                                            kBatchCollectWindow,
                                            [this, target_batch_size]() {
                                                return _stop_requested ||
                                                       _work_queue.size() >= target_batch_size;
                                            });
            }

            const std::size_t available_batch_size = std::min(target_batch_size, _work_queue.size());
            batch.reserve(available_batch_size);

            for (std::size_t i = 0; i < available_batch_size; ++i) {
                batch.push_back(std::move(_work_queue.front()));
                _work_queue.pop();
            }
        }

        try {
            TensorList flat_outputs = _run_inference_batch(batch);

            for (std::size_t batch_index = 0; batch_index < batch.size(); ++batch_index) {
                TensorList request_outputs;
                request_outputs.reserve(_output_specs.size());

                for (std::size_t tensor_index = 0; tensor_index < _output_specs.size(); ++tensor_index) {
                    const TensorSpec& spec = _output_specs[tensor_index];
                    const auto begin = flat_outputs[tensor_index].begin() +
                                       static_cast<std::ptrdiff_t>(batch_index * spec.sample_elements);
                    const auto end = begin + static_cast<std::ptrdiff_t>(spec.sample_elements);
                    request_outputs.emplace_back(begin, end);
                }

                batch[batch_index].promise.set_value(std::move(request_outputs));
            }
        } catch (...) {
            for (auto& request : batch) {
                request.promise.set_exception(std::current_exception());
            }
        }
    }
}
