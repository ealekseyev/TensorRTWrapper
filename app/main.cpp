#include "inference_engine.hpp"
#include "image_utils.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<Detection> parse_yolo_detections(const std::vector<float>& output) {
    constexpr int kDetections = 8400;
    constexpr int kParameters = 84;
    constexpr int kBoxValues = 4;
    constexpr float kConfidenceThreshold = 0.4f;

    if (output.size() != static_cast<std::size_t>(kDetections * kParameters)) {
        throw std::runtime_error("Unexpected YOLO output size: expected " +
                                 std::to_string(kDetections * kParameters) +
                                 " floats, got " + std::to_string(output.size()));
    }

    std::vector<Detection> detections;
    detections.reserve(kDetections);

    for (int detection_idx = 0; detection_idx < kDetections; ++detection_idx) {
        const float x1 = output[detection_idx];
        const float y1 = output[kDetections + detection_idx];
        const float x2 = output[(2 * kDetections) + detection_idx];
        const float y2 = output[(3 * kDetections) + detection_idx];

        int best_class = -1;
        float best_confidence = 0.0f;
        for (int class_channel = kBoxValues; class_channel < kParameters; ++class_channel) {
            const float confidence =
                output[static_cast<std::size_t>(class_channel) * kDetections + detection_idx];
            if (confidence > best_confidence) {
                best_confidence = confidence;
                best_class = class_channel - kBoxValues;
            }
        }

        if (best_confidence > kConfidenceThreshold) {
            detections.push_back({best_class, best_confidence, {x1, y1, x2, y2}});
        }
    }

    return detections;
}

float intersection_over_union(const YoloBBox& lhs, const YoloBBox& rhs) {
    const YoloBBox a = yolo_bbox_to_corners(lhs);
    const YoloBBox b = yolo_bbox_to_corners(rhs);

    const float inter_x1 = std::max(a.x1, b.x1);
    const float inter_y1 = std::max(a.y1, b.y1);
    const float inter_x2 = std::min(a.x2, b.x2);
    const float inter_y2 = std::min(a.y2, b.y2);

    const float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    const float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    const float intersection = inter_w * inter_h;

    const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    const float union_area = area_a + area_b - intersection;
    if (union_area <= 0.0f) {
        return 0.0f;
    }
    return intersection / union_area;
}

std::vector<Detection> suppress_duplicate_detections(std::vector<Detection> detections) {
    constexpr float kIouThreshold = 0.5f;

    std::sort(detections.begin(), detections.end(), [](const Detection& lhs, const Detection& rhs) {
        return lhs.confidence > rhs.confidence;
    });

    std::vector<Detection> filtered;
    filtered.reserve(detections.size());

    for (const auto& detection : detections) {
        bool overlaps_existing = false;
        for (const auto& kept : filtered) {
            if (kept.class_id != detection.class_id) {
                continue;
            }
            if (intersection_over_union(kept.bbox, detection.bbox) > kIouThreshold) {
                overlaps_existing = true;
                break;
            }
        }

        if (!overlaps_existing) {
            filtered.push_back(detection);
        }
    }

    return filtered;
}

void print_yolo_detections(const std::vector<Detection>& detections,
                           const LetterboxTransform& transform) {
    std::cout << "Detections above confidence threshold:\n";
    std::cout << std::fixed << std::setprecision(4);

    if (detections.empty()) {
        std::cout << "no detections above threshold\n";
        return;
    }

    for (const auto& detection : detections) {
        const YoloBBox mapped = map_yolo_bbox_to_original(detection.bbox, transform);
        std::cout << "class=" << detection.class_id
                  << " conf=" << detection.confidence
                  << " bbox_640=(" << detection.bbox.x1 << ", " << detection.bbox.y1 << ", "
                  << detection.bbox.x2 << ", " << detection.bbox.y2 << ")"
                  << " bbox_original=(" << mapped.x1 << ", " << mapped.y1 << ", "
                  << mapped.x2 << ", " << mapped.y2 << ")\n";
    }
}

constexpr std::size_t kModelInputElements = 3 * 640 * 640;

double benchmark(InferenceEngine& engine, const std::vector<float>& input_image, std::size_t iterations) {
    if (iterations == 0) {
        throw std::invalid_argument("benchmark requires at least one iteration");
    }

    std::vector<InferenceFuture> pending_results;
    pending_results.reserve(iterations);

    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < iterations; ++i) {
        pending_results.push_back(engine.submit(std::vector<float>(input_image)));
    }

    for (auto& result : pending_results) {
        result.get();
    }

    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    const double seconds = elapsed.count();
    if (seconds <= 0.0) {
        throw std::runtime_error("benchmark measured a non-positive runtime");
    }

    const double fps = static_cast<double>(iterations) / seconds;
    std::cout << "Benchmark: " << iterations << " inferences in "
              << seconds << " s -> " << fps << " FPS\n";
    return fps;
}

}

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0] << " <image-path>\n";
            return 1;
        }

        const std::string image_path = argv[1];
        const auto image = load_image_chw(image_path);
        auto resized = resize_to_640x640_padded(image.data.get(), image.height, image.width);
        const LetterboxTransform transform = make_letterbox_transform(image.width, image.height);

        InferenceEngine engine;
        engine.load_model("models/trt/model.trt");

        if (engine.inputs().empty()) {
            throw std::runtime_error("Model exposes no input tensors");
        }

        const std::size_t input_elements = kModelInputElements;
        if (engine.inputs()[0].sample_elements < input_elements) {
            throw std::runtime_error("Model input tensor is smaller than 3x640x640 floats");
        }
        const std::vector<float> input_image = chw_u8_to_float_input(resized.get(), input_elements);
        TensorList model_inputs;
        model_inputs.push_back(input_image);

        std::cout << "Loaded image: " << image_path << " (" << image.width << "x" << image.height
                  << "), resized to 3x640x640\n";

        const TensorList outputs = engine.infer(std::move(model_inputs));
        if (outputs.empty()) {
            throw std::runtime_error("Inference produced no output tensors");
        }

        auto detections = parse_yolo_detections(outputs[0]);
        detections = suppress_duplicate_detections(std::move(detections));
        print_yolo_detections(detections, transform);

        const cv::Mat original_image = load_image_bgr(image_path);
        const cv::Mat annotated = draw_detections(original_image, detections, transform, true);

        const std::filesystem::path input_path(image_path);
        const std::filesystem::path output_path =
            input_path.parent_path() /
            (input_path.stem().string() + "_detections" + input_path.extension().string());
        save_image(output_path.string(), annotated);
        std::cout << "Annotated image saved to: " << output_path << "\n";

        benchmark(engine, input_image, 100);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
