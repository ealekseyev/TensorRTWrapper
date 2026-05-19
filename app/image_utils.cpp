#include "image_utils.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

float clamp_to_extent(float value, int extent) {
    if (extent <= 0) {
        return 0.0f;
    }
    return std::clamp(value, 0.0f, static_cast<float>(extent - 1));
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

}  // namespace

ImageCHW load_image_chw(const std::string& path) {
    int width = 0;
    int height = 0;
    int channels_in_file = 0;
    stbi_uc* hwc = stbi_load(path.c_str(), &width, &height, &channels_in_file, 3);
    if (hwc == nullptr) {
        throw std::runtime_error("Failed to load image \"" + path + "\": " + stbi_failure_reason());
    }

    const std::size_t plane_size = static_cast<std::size_t>(height) * width;
    std::shared_ptr<std::uint8_t[]> chw(new std::uint8_t[plane_size * 3],
                                        std::default_delete<std::uint8_t[]>());

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t hwc_index = static_cast<std::size_t>(y) * width * 3 + x * 3;
            const std::size_t chw_index = static_cast<std::size_t>(y) * width + x;
            chw[chw_index] = hwc[hwc_index];
            chw[plane_size + chw_index] = hwc[hwc_index + 1];
            chw[(plane_size * 2) + chw_index] = hwc[hwc_index + 2];
        }
    }

    stbi_image_free(hwc);
    return {std::move(chw), 3, height, width};
}

std::shared_ptr<std::uint8_t[]> resize_to_640x640_padded(const std::uint8_t* input_chw,
                                                         int input_height,
                                                         int input_width) {
    if (input_chw == nullptr) {
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
    const int resized_width = std::max(1, static_cast<int>(std::round(input_width * scale)));
    const int resized_height = std::max(1, static_cast<int>(std::round(input_height * scale)));
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
            const float wy = src_y - static_cast<float>(y0);

            for (int x = 0; x < resized_width; ++x) {
                const float src_x = (static_cast<float>(x) + 0.5f) / scale - 0.5f;
                const int x0 = std::clamp(static_cast<int>(src_x), 0, input_width - 1);
                const int x1 = std::min(x0 + 1, input_width - 1);
                const float wx = src_x - static_cast<float>(x0);

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
    if (input_chw == nullptr) {
        throw std::invalid_argument("chw_u8_to_float_input received a null image");
    }

    std::vector<float> input(element_count);
    for (std::size_t i = 0; i < element_count; ++i) {
        input[i] = static_cast<float>(input_chw[i]) / 255.0f;
    }
    return input;
}

LetterboxTransform make_letterbox_transform(int original_width,
                                            int original_height,
                                            int target_width,
                                            int target_height) {
    if (original_width <= 0 || original_height <= 0) {
        throw std::invalid_argument("make_letterbox_transform requires positive original dimensions");
    }
    if (target_width <= 0 || target_height <= 0) {
        throw std::invalid_argument("make_letterbox_transform requires positive target dimensions");
    }

    const float scale = std::min(static_cast<float>(target_width) / original_width,
                                 static_cast<float>(target_height) / original_height);
    const int resized_width = std::max(1, static_cast<int>(std::round(original_width * scale)));
    const int resized_height = std::max(1, static_cast<int>(std::round(original_height * scale)));
    const int pad_x = (target_width - resized_width) / 2;
    const int pad_y = (target_height - resized_height) / 2;

    return {original_width,
            original_height,
            target_width,
            target_height,
            resized_width,
            resized_height,
            pad_x,
            pad_y,
            scale};
}

YoloBBox yolo_bbox_to_corners(const YoloBBox& bbox) {
    return to_corner_box(bbox);
}

YoloBBox map_yolo_bbox_to_original(const YoloBBox& bbox, const LetterboxTransform& transform) {
    if (transform.scale <= 0.0f) {
        throw std::invalid_argument("map_yolo_bbox_to_original requires a positive scale");
    }

    const YoloBBox corners = to_corner_box(bbox);

    YoloBBox mapped{};
    mapped.x1 = clamp_to_extent((corners.x1 - transform.pad_x) / transform.scale,
                                transform.original_width);
    mapped.y1 = clamp_to_extent((corners.y1 - transform.pad_y) / transform.scale,
                                transform.original_height);
    mapped.x2 = clamp_to_extent((corners.x2 - transform.pad_x) / transform.scale,
                                transform.original_width);
    mapped.y2 = clamp_to_extent((corners.y2 - transform.pad_y) / transform.scale,
                                transform.original_height);
    return mapped;
}

cv::Mat load_image_bgr(const std::string& path) {
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Failed to open image with OpenCV: " + path);
    }
    return image;
}

cv::Mat draw_detections(const cv::Mat& image,
                        const std::vector<Detection>& detections,
                        const LetterboxTransform& transform,
                        bool map_to_original) {
    if (image.empty()) {
        throw std::invalid_argument("draw_detections received an empty image");
    }

    cv::Mat annotated = image.clone();
    for (const auto& detection : detections) {
        const YoloBBox box = map_to_original ? map_yolo_bbox_to_original(detection.bbox, transform)
                                             : to_corner_box(detection.bbox);

        const cv::Point top_left(static_cast<int>(std::round(box.x1)),
                                 static_cast<int>(std::round(box.y1)));
        const cv::Point bottom_right(static_cast<int>(std::round(box.x2)),
                                     static_cast<int>(std::round(box.y2)));

        cv::rectangle(annotated, top_left, bottom_right, cv::Scalar(0, 255, 0), 2);
        const std::string label = "class=" + std::to_string(detection.class_id) +
                                  " conf=" + cv::format("%.2f", detection.confidence);
        const cv::Point text_origin(top_left.x, std::max(15, top_left.y - 8));
        cv::putText(annotated,
                    label,
                    text_origin,
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(0, 255, 0),
                    1,
                    cv::LINE_AA);
    }

    return annotated;
}

void save_image(const std::string& path, const cv::Mat& image) {
    if (image.empty()) {
        throw std::invalid_argument("save_image received an empty image");
    }
    if (!cv::imwrite(path, image)) {
        throw std::runtime_error("Failed to write image: " + path);
    }
}
