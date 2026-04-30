#include "inference_engine.hpp"
#include "NvInfer.h"
#include <cuda_runtime.h>
#include <fstream>
#include <stdexcept>
#include <iostream>

// ─── InferenceResult ────────────────────────────────────────────────────────

InferenceResult::InferenceResult() {

}

void InferenceResult::wait() {
    std::unique_lock<std::mutex> lock(this->_mutex);
    this->_cv.wait(lock, [this] { return this->_ready; });
}

bool InferenceResult::is_ready() const {
    std::unique_lock<std::mutex> lock(this->_mutex);
    return this->_ready;
}

// must be called after wait(), otherwise results will be undefined
const std::vector<float>& InferenceResult::data() const {
    return this->_output;
}

// called by InferenceEngine via separate thread
void InferenceResult::set(std::vector<float> output) {
    {
        std::unique_lock<std::mutex> lock(this->_mutex);
        this->_output = std::move(output);
        this->_ready = true;
    }
    this->_cv.notify_one();
}

// ─── InferenceEngine ────────────────────────────────────────────────────────

InferenceEngine::InferenceEngine(Logger::Severity log_level, size_t batch_size) {
    this->_logger = Logger(log_level);
    this->batch_size = batch_size;
}

InferenceEngine::~InferenceEngine() {
    using std::cout;
    if(!this->_inference_thread.joinable()) return; // skip if start() was never called
    cout << "quitting. acquiring lock...\n";
    std::unique_lock<std::mutex> lock(this->_queue_mutex);
    this->_stop_event = true;
    this->_work_queue = {}; // drain work queue
    cout << "set flags...\n";

    lock.unlock();
    this->_work_available_cv.notify_one();

    this->_inference_thread.join();
        cout << "joining...\n";
}

void InferenceEngine::load_model(const std::string& path) {
    // load raw engine data from file
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open engine: " + path);
    size_t size = file.tellg();
    file.seekg(0);
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);

    // create runtimes
    this->_runtime = nvinfer1::createInferRuntime(this->_logger);
    this->_engine = this->_runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    this->_context = this->_engine->createExecutionContext();
    int32_t nbIO = this->_engine->getNbIOTensors();
    std::cout << "Number of I/O tensors: " << nbIO << "\n";

    const char* inputName  = nullptr;
    const char* outputName = nullptr;
    for (int i = 0; i < nbIO; ++i) {
        const char* name = this->_engine->getIOTensorName(i);
        auto mode = this->_engine->getTensorIOMode(name);
        auto dims = this->_engine->getTensorShape(name);

        std::cout << "  Tensor[" << i << "]: " << name
                  << (mode == nvinfer1::TensorIOMode::kINPUT ? " [INPUT]" : " [OUTPUT]")
                  << "  shape: [";
        for (int d = 0; d < dims.nbDims; ++d)
            std::cout << dims.d[d] << (d + 1 < dims.nbDims ? "," : "");
        std::cout << "]\n";

        if (mode == nvinfer1::TensorIOMode::kINPUT)  inputName  = name;
        if (mode == nvinfer1::TensorIOMode::kOUTPUT) outputName = name;
        // (multiple I/O: collect into arrays instead)
    }
}

std::shared_ptr<InferenceResult> InferenceEngine::enqueue(const std::vector<float>& input) {
    auto result = std::make_shared<InferenceResult>();
    {   
        std::lock_guard<std::mutex> lock(this->_queue_mutex);
        this->_work_queue.emplace(input, result);
    }
    this->_work_available_cv.notify_one();
    return result;
}

// this should use RVO, no copies necessary
std::vector<std::vector<float>> InferenceEngine::_run_inference(const std::vector<InferenceJob>& input) {
    static int count = 0;
    std::vector<std::vector<float>> results;
    results.reserve(input.size());
    std::cout << "Inference " << count++ << std::endl;
    // run the inference here
    return results;
}

void InferenceEngine::start() {
    this->_inference_thread = std::thread(&InferenceEngine::_inference_worker, this);
}

void InferenceEngine::_inference_worker() {
    // init cuda stuff here
    while(true) {
        std::unique_lock<std::mutex> lock(this->_queue_mutex);

        // sleep until someone appends to work queue or stop event gets set
        this->_work_available_cv.wait(lock, [this]() {
            return !this->_work_queue.empty() || this->_stop_event;
        });

        if(this->_stop_event) {
            // cleanup here
            break;
        }

        // collect jobs from queue
        size_t batch_size = std::min(this->_work_queue.size(), this->batch_size);

        std::vector<InferenceJob> jobs;
        jobs.reserve(batch_size);

        for(size_t i = 0; i < batch_size; i++) {
            jobs.push_back(std::move(this->_work_queue.front()));
            this->_work_queue.pop();
        }
        // weve copied the input and removed from queue, remove it
        lock.unlock();

        // now perform inference
        std::vector<std::vector<float>> inference_results = this->_run_inference(jobs);

        // now write to inferenceresult and flag as done
        for(size_t i = 0; i < batch_size; i++) {
            jobs[i].result->set(std::move(inference_results[i])); // no need for mutex, handled internally by InferenceResult
        }

    }
}