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
const std::vector<std::vector<float>>& InferenceResult::data() const {
    return this->_output;
}

// called by InferenceEngine via separate thread
void InferenceResult::set(std::vector<std::vector<float>> output) {
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
    if(this->_inference_thread.joinable()) {
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

    for (auto& t : _inputs)  cudaFree(t.d_ptr);
    for (auto& t : _outputs) cudaFree(t.d_ptr);
    if (_stream)  cudaStreamDestroy(_stream);

    delete _context;
    delete _engine;
    delete _runtime;
}

void InferenceEngine::load_model(const std::string& path) {
    // --- read engine file ---
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open engine: " + path);
    size_t fsize = static_cast<size_t>(file.tellg());
    file.seekg(0);
    std::vector<char> engineData(fsize);
    file.read(engineData.data(), fsize);

    // --- deserialize ---
    _runtime = nvinfer1::createInferRuntime(_logger);
    if (!_runtime) throw std::runtime_error("createInferRuntime failed");
    _engine = _runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    if (!_engine) throw std::runtime_error("deserializeCudaEngine failed");
    _context = _engine->createExecutionContext();
    if (!_context) throw std::runtime_error("createExecutionContext failed");

    cudaStreamCreate(&_stream);

    auto elem_bytes = [&](const char* name) -> size_t {
        switch (_engine->getTensorDataType(name)) {
            case nvinfer1::DataType::kFLOAT: return 4;
            case nvinfer1::DataType::kHALF:  return 2;
            case nvinfer1::DataType::kINT32: return 4;
            case nvinfer1::DataType::kINT8:  return 1;
            case nvinfer1::DataType::kBOOL:  return 1;
            default: return 4;
        }
    };

    auto vol = [](const nvinfer1::Dims& d) -> size_t {
        size_t n = 1;
        for (int i = 0; i < d.nbDims; ++i) n *= static_cast<size_t>(d.d[i]);
        return n;
    };

    int32_t nbIO = _engine->getNbIOTensors();

    // --- pass 1: inputs — set concrete shapes if dynamic, then allocate ---
    for (int i = 0; i < nbIO; ++i) {
        const char* name = _engine->getIOTensorName(i);
        if (_engine->getTensorIOMode(name) != nvinfer1::TensorIOMode::kINPUT) continue;

        nvinfer1::Dims dims = _engine->getTensorShape(name);

        // if any dim is -1 (dynamic), bind the min-profile shape as default
        bool is_dynamic = false;
        for (int d = 0; d < dims.nbDims; ++d) if (dims.d[d] < 0) { is_dynamic = true; break; }

        if (is_dynamic) {
            nvinfer1::Dims concrete = _engine->getProfileShape(
                name, 0, nvinfer1::OptProfileSelector::kOPT);
            _context->setInputShape(name, concrete);
            dims = concrete;
        }

        size_t bytes = vol(dims) * elem_bytes(name);
        void* dptr = nullptr;
        cudaMalloc(&dptr, bytes);
        _context->setTensorAddress(name, dptr);

        _inputs.push_back({name, dims, dptr, bytes});

        std::cout << "[load_model] INPUT  " << name << "  shape=[";
        for (int d = 0; d < dims.nbDims; ++d)
            std::cout << dims.d[d] << (d + 1 < dims.nbDims ? "," : "");
        std::cout << "]  bytes=" << bytes << "\n";
    }

    // --- pass 2: outputs — shapes resolved after inputs are bound ---
    for (int i = 0; i < nbIO; ++i) {
        const char* name = _engine->getIOTensorName(i);
        if (_engine->getTensorIOMode(name) != nvinfer1::TensorIOMode::kOUTPUT) continue;

        nvinfer1::Dims dims = _context->getTensorShape(name); // concrete after input binding
        size_t bytes = vol(dims) * elem_bytes(name);
        void* dptr = nullptr;
        cudaMalloc(&dptr, bytes);
        _context->setTensorAddress(name, dptr);

        _outputs.push_back({name, dims, dptr, bytes});

        std::cout << "[load_model] OUTPUT " << name << "  shape=[";
        for (int d = 0; d < dims.nbDims; ++d)
            std::cout << dims.d[d] << (d + 1 < dims.nbDims ? "," : "");
        std::cout << "]  bytes=" << bytes << "\n";
    }

    model_loaded_ = true;
}

std::shared_ptr<InferenceResult> InferenceEngine::enqueue(std::vector<float> input) {
    return enqueue(std::vector<std::vector<float>>{std::move(input)});
}

std::shared_ptr<InferenceResult> InferenceEngine::enqueue(std::vector<std::vector<float>> inputs) {
    auto result = std::make_shared<InferenceResult>();
    {
        std::lock_guard<std::mutex> lock(_queue_mutex);
        _work_queue.emplace(std::move(inputs), result);
    }
    _work_available_cv.notify_one();
    return result;
}

// this should use RVO, no copies necessary
std::vector<std::vector<float>> InferenceEngine::_run_inference(const std::vector<InferenceJob>& jobs) {
    const size_t n = jobs.size();

    // H2D: for each input tensor, pack all jobs back-to-back into the device buffer
    for (size_t ti = 0; ti < _inputs.size(); ++ti) {
        const size_t stride = _inputs[ti].bytes / batch_size; // bytes per single sample
        for (size_t ji = 0; ji < n; ++ji) {
            cudaMemcpyAsync(
                static_cast<char*>(_inputs[ti].d_ptr) + ji * stride,
                jobs[ji].inputs[ti].data(),
                stride,
                cudaMemcpyHostToDevice, _stream);
        }
    }

    _context->enqueueV3(_stream);

    // D2H: copy each output tensor directly into a vector — vector::data() is the DMA target
    std::vector<std::vector<float>> results(_outputs.size());
    for (size_t ti = 0; ti < _outputs.size(); ++ti) {
        results[ti].resize(_outputs[ti].bytes / sizeof(float));
        cudaMemcpyAsync(
            results[ti].data(),
            _outputs[ti].d_ptr,
            _outputs[ti].bytes,
            cudaMemcpyDeviceToHost, _stream);
    }

    cudaStreamSynchronize(_stream);

    return results; // [output_tensor][all jobs flat]
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

        // now perform inference — returns [output_tensor][all jobs flat]
        std::vector<std::vector<float>> inference_results = this->_run_inference(jobs);

        // slice per-job and deliver results
        for (size_t i = 0; i < batch_size; ++i) {
            std::vector<std::vector<float>> job_outputs(_outputs.size());
            for (size_t ti = 0; ti < _outputs.size(); ++ti) {
                const size_t per_job = inference_results[ti].size() / batch_size;
                auto beg = inference_results[ti].begin() + i * per_job;
                job_outputs[ti].assign(beg, beg + per_job);
            }
            jobs[i].result->set(std::move(job_outputs));
        }

    }
}