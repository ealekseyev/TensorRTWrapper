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

class InferenceResult {
public:
    InferenceResult();

    // block caller until result is ready
    void wait();
    bool is_ready() const;

    const std::vector<float>& data() const;

    // called internally by the engine when inference completes
    void set(std::vector<float> output);

private:
    mutable std::mutex          _mutex;
    std::condition_variable     _cv;
    std::vector<float>          _output;
    bool                        _ready = false;
};

class InferenceJob {
public:
    std::vector<float> input;
    std::shared_ptr<InferenceResult> result;
    InferenceJob(std::vector<float> input, std::shared_ptr<InferenceResult> result)
        : input(std::move(input)), result(std::move(result)) {}
    InferenceJob(std::vector<float> input): input(std::move(input)) { result = std::make_shared<InferenceResult>();};
    ~InferenceJob(){};
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

    // non-blocking — returns immediately, result populated async
    std::shared_ptr<InferenceResult> enqueue(const std::vector<float>& input);

private:
    Logger                                        _logger;
    nvinfer1::IRuntime*                           _runtime  = nullptr;
    nvinfer1::ICudaEngine*                        _engine   = nullptr;
    nvinfer1::IExecutionContext*                  _context  = nullptr;

    bool model_loaded_ = false;
    bool _stop_event = false;
    size_t batch_size;
    std::mutex _queue_mutex;
    std::condition_variable _work_available_cv;
    std::queue<InferenceJob> _work_queue;
    std::thread _inference_thread;

    void _inference_worker();
    std::vector<std::vector<float>> _run_inference(const std::vector<InferenceJob>& input);
};