#include "inference_engine.hpp"
#include "viewer.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <filesystem>
#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct ImageCHW {
    std::shared_ptr<std::uint8_t[]> data;
    int channels;
    int height;
    int width;
};

ImageCHW load_image_chw(const std::string& path) {
    int width = 0;
    int height = 0;
    int channels_in_file = 0;
    stbi_uc* hwc = stbi_load(path.c_str(), &width, &height, &channels_in_file, 3);
    if (!hwc) {
        throw std::runtime_error(
            "Failed to load image \"" + path + "\": " + stbi_failure_reason());
    }

    const std::size_t plane_size = static_cast<std::size_t>(height) * width;
    std::shared_ptr<std::uint8_t[]> chw(new std::uint8_t[plane_size * 3],
                                        std::default_delete<std::uint8_t[]>());

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t hwc_idx = static_cast<std::size_t>(y) * width * 3 + x * 3;
            const std::size_t chw_idx = static_cast<std::size_t>(y) * width + x;
            chw[chw_idx] = hwc[hwc_idx];
            chw[plane_size + chw_idx] = hwc[hwc_idx + 1];
            chw[(plane_size * 2) + chw_idx] = hwc[hwc_idx + 2];
        }
    }

    stbi_image_free(hwc);
    return {std::move(chw), 3, height, width};
}

std::shared_ptr<std::uint8_t[]> resize_to_640x640_padded(const std::uint8_t* input_chw,
                                                         int input_height,
                                                         int input_width) {
    if (!input_chw) {
        throw std::invalid_argument("resize_to_640x640_padded received a null image");
    }
    if (input_height <= 0 || input_width <= 0) {
        throw std::invalid_argument("resize_to_640x640_padded requires positive dimensions");
    }

    constexpr int kChannels = 3;
    constexpr int kTargetSize = 640;
    const std::size_t output_plane_size = static_cast<std::size_t>(kTargetSize) * kTargetSize;

    std::shared_ptr<std::uint8_t[]> output(new std::uint8_t[output_plane_size * kChannels](),
                                           std::default_delete<std::uint8_t[]>());

    const float scale = std::min(static_cast<float>(kTargetSize) / input_width,
                                 static_cast<float>(kTargetSize) / input_height);
    const int resized_width = std::max(1, static_cast<int>(input_width * scale));
    const int resized_height = std::max(1, static_cast<int>(input_height * scale));
    const int pad_x = (kTargetSize - resized_width) / 2;
    const int pad_y = (kTargetSize - resized_height) / 2;

    const std::size_t input_plane_size = static_cast<std::size_t>(input_height) * input_width;

    for (int c = 0; c < kChannels; ++c) {
        const std::uint8_t* src_plane = input_chw + (static_cast<std::size_t>(c) * input_plane_size);
        std::uint8_t* dst_plane = output.get() + (static_cast<std::size_t>(c) * output_plane_size);

        for (int y = 0; y < resized_height; ++y) {
            const float src_y = (static_cast<float>(y) + 0.5f) / scale - 0.5f;
            const int y0 = std::clamp(static_cast<int>(src_y), 0, input_height - 1);
            const int y1 = std::min(y0 + 1, input_height - 1);
            const float wy = src_y - y0;

            for (int x = 0; x < resized_width; ++x) {
                const float src_x = (static_cast<float>(x) + 0.5f) / scale - 0.5f;
                const int x0 = std::clamp(static_cast<int>(src_x), 0, input_width - 1);
                const int x1 = std::min(x0 + 1, input_width - 1);
                const float wx = src_x - x0;

                const float top = src_plane[static_cast<std::size_t>(y0) * input_width + x0] * (1.0f - wx) +
                                  src_plane[static_cast<std::size_t>(y0) * input_width + x1] * wx;
                const float bottom = src_plane[static_cast<std::size_t>(y1) * input_width + x0] * (1.0f - wx) +
                                     src_plane[static_cast<std::size_t>(y1) * input_width + x1] * wx;
                const float value = top * (1.0f - wy) + bottom * wy;

                const int out_y = y + pad_y;
                const int out_x = x + pad_x;
                dst_plane[static_cast<std::size_t>(out_y) * kTargetSize + out_x] =
                    static_cast<std::uint8_t>(std::clamp(value, 0.0f, 255.0f));
            }
        }
    }

    return output;
}

std::vector<float> chw_u8_to_float_input(const std::uint8_t* input_chw, std::size_t element_count) {
    if (!input_chw) {
        throw std::invalid_argument("chw_u8_to_float_input received a null image");
    }

    std::vector<float> input(element_count);
    for (std::size_t i = 0; i < element_count; ++i) {
        input[i] = static_cast<float>(input_chw[i]) / 255.0f;
    }
    return input;
}

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

YoloBBox to_corner_box(const YoloBBox& bbox) {
    const float half_w = bbox.x2 * 0.5f;
    const float half_h = bbox.y2 * 0.5f;
    return {
        bbox.x1 - half_w,
        bbox.y1 - half_h,
        bbox.x1 + half_w,
        bbox.y1 + half_h,
    };
}

float intersection_over_union(const YoloBBox& lhs, const YoloBBox& rhs) {
    const YoloBBox a = to_corner_box(lhs);
    const YoloBBox b = to_corner_box(rhs);

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

    std::vector<std::shared_ptr<InferenceResult>> pending_results;
    pending_results.reserve(iterations);

    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < iterations; ++i) {
        pending_results.push_back(engine.enqueue(std::vector<float>(input_image)));
    }

    for (auto& result : pending_results) {
        result->wait();
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

}  // namespace

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
        engine.start();

        if (engine.inputs().empty()) {
            throw std::runtime_error("Model exposes no input tensors");
        }

        const std::size_t input_elements = kModelInputElements;
        if (engine.inputs()[0].bytes / sizeof(float) < input_elements) {
            throw std::runtime_error("Model input tensor is smaller than 3x640x640 floats");
        }
        std::vector<std::vector<float>> model_inputs;
        const std::vector<float> input_image = chw_u8_to_float_input(resized.get(), input_elements);
        model_inputs.push_back(input_image);

        std::cout << "Loaded image: " << image_path << " (" << image.width << "x" << image.height
                  << "), resized to 3x640x640\n";

        auto result = engine.enqueue(std::move(model_inputs));
        result->wait();

        const auto& outputs = result->data();
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
