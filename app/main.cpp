#include "inference_engine.hpp"
#include <iostream>
#include <vector>

int main() {
    InferenceEngine engine;
    std::vector<float> input = {1.0f, 2.0f, 3.0f};
    engine.load_model("models/trt/model.trt");
    // engine.start();
    return 0;
}
