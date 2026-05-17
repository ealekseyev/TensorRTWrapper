#pragma once
#include <string>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <future>
#include <memory>
#include <queue>
#include <thread>
#include "NvInfer.h"
#include "logger.hpp"
#include <cuda_runtime.h>

class InferenceResult {
public:
    InferenceResult();

    // block caller until result is ready
    void wait();
    bool is_ready() const;

    const std::vector<std::vector<float>>& data() const;

    // called internally by the engine when inference completes
    void set(std::vector<std::vector<float>> output);

private:
    mutable std::mutex          _mutex;
    std::condition_variable     _cv;
    std::vector<std::vector<float>> _output;
    bool                        _ready = false;
};

class InferenceJob {
public:
    std::vector<std::vector<float>> inputs;  // one entry per input tensor
    std::shared_ptr<InferenceResult> result;

    // multi-input constructor
    InferenceJob(std::vector<std::vector<float>> inputs, std::shared_ptr<InferenceResult> result)
        : inputs(std::move(inputs)), result(std::move(result)) {}

    // convenience: single-input model
    explicit InferenceJob(std::vector<float> input)
        : inputs({std::move(input)}), result(std::make_shared<InferenceResult>()) {}
};

struct TensorBuffer {
    std::string         name;
    nvinfer1::Dims      dims;
    void*               d_ptr  = nullptr;  // device memory
    size_t              bytes  = 0;
};

class InferenceEngine {
public:
    explicit InferenceEngine(Logger::Severity log_level = Logger::Severity::kWARNING, size_t batch_size = 4);
    ~InferenceEngine();

    // no copy — owns GPU resources
    InferenceEngine(const InferenceEngine&)            = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;

    void load_model(const std::string& path);
    void start();

    const std::vector<TensorBuffer>& inputs()  const { return _inputs; }
    const std::vector<TensorBuffer>& outputs() const { return _outputs; }

    // single-input convenience overload
    std::shared_ptr<InferenceResult> enqueue(std::vector<float> input);
    // multi-input: one vector per input tensor
    std::shared_ptr<InferenceResult> enqueue(std::vector<std::vector<float>> inputs);

private:
    Logger                                        _logger;
    nvinfer1::IRuntime*                           _runtime  = nullptr;
    nvinfer1::ICudaEngine*                        _engine   = nullptr;
    nvinfer1::IExecutionContext*                  _context  = nullptr;
    cudaStream_t                                  _stream   = nullptr;
    std::vector<TensorBuffer>                     _inputs;
    std::vector<TensorBuffer>                     _outputs;

    bool model_loaded_ = false;
    bool _stop_event = false;
    size_t batch_size;
    std::mutex _queue_mutex;
    std::condition_variable _work_available_cv;
    std::queue<InferenceJob> _work_queue;
    std::thread _inference_thread;

    void _inference_worker();
    std::vector<std::vector<float>> _run_inference(const std::vector<InferenceJob>& jobs);
};