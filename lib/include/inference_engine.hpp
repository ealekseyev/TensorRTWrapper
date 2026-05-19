#pragma once

#include <condition_variable>
#include <cstddef>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "NvInfer.h"
#include "logger.hpp"

using TensorData = std::vector<float>;
using TensorList = std::vector<TensorData>;

struct TensorSpec {
    std::string name;
    nvinfer1::Dims dims;
    std::size_t element_bytes = 0;
    std::size_t total_elements = 0;
    std::size_t sample_elements = 0;
    std::size_t total_bytes = 0;
    std::size_t sample_bytes = 0;
    std::size_t max_batch_size = 1;
};

using InferenceFuture = std::future<TensorList>;

class InferenceEngine {
public:
    explicit InferenceEngine(Logger::Severity log_level = Logger::Severity::kWARNING,
                             std::size_t queue_batch_size = 4);
    ~InferenceEngine();

    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;

    void load_model(const std::string& path);
    void start();

    const std::vector<TensorSpec>& inputs() const { return _input_specs; }
    const std::vector<TensorSpec>& outputs() const { return _output_specs; }
    std::size_t max_batch_size() const { return _engine_batch_size; }

    InferenceFuture submit(TensorData input);
    InferenceFuture submit(TensorList inputs);

    TensorList infer(TensorData input);
    TensorList infer(TensorList inputs);

private:
    struct DeviceTensor {
        TensorSpec spec;
        void* device_ptr = nullptr;
    };

    struct PendingRequest {
        TensorList inputs;
        std::promise<TensorList> promise;
    };

    Logger _logger;
    nvinfer1::IRuntime* _runtime = nullptr;
    nvinfer1::ICudaEngine* _engine = nullptr;
    nvinfer1::IExecutionContext* _context = nullptr;
    cudaStream_t _stream = nullptr;

    std::vector<TensorSpec> _input_specs;
    std::vector<TensorSpec> _output_specs;
    std::vector<DeviceTensor> _input_buffers;
    std::vector<DeviceTensor> _output_buffers;

    bool _model_loaded = false;
    bool _stop_requested = false;
    bool _worker_started = false;
    std::size_t _queue_batch_size = 1;
    std::size_t _engine_batch_size = 1;

    std::mutex _queue_mutex;
    std::condition_variable _work_available_cv;
    std::queue<PendingRequest> _work_queue;
    std::thread _inference_thread;

    void _ensure_worker_started();
    void _inference_worker();
    void _validate_inputs(const TensorList& inputs) const;
    TensorList _run_inference_batch(const std::vector<PendingRequest>& batch);
};
