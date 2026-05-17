#include "viewer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

float clamp_to_extent(float value, int extent) {
    if (extent <= 0) {
        return 0.0f;
    }
    return std::clamp(value, 0.0f, static_cast<float>(extent - 1));
}

}  // namespace

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

YoloBBox map_yolo_bbox_to_original(const YoloBBox& bbox, const LetterboxTransform& transform) {
    if (transform.scale <= 0.0f) {
        throw std::invalid_argument("map_yolo_bbox_to_original requires a positive scale");
    }

    const float half_w = bbox.x2 * 0.5f;
    const float half_h = bbox.y2 * 0.5f;

    YoloBBox mapped{};
    mapped.x1 = clamp_to_extent((bbox.x1 - half_w - transform.pad_x) / transform.scale,
                                transform.original_width);
    mapped.y1 = clamp_to_extent((bbox.y1 - half_h - transform.pad_y) / transform.scale,
                                transform.original_height);
    mapped.x2 = clamp_to_extent((bbox.x1 + half_w - transform.pad_x) / transform.scale,
                                transform.original_width);
    mapped.y2 = clamp_to_extent((bbox.y1 + half_h - transform.pad_y) / transform.scale,
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
        YoloBBox box{};
        if (map_to_original) {
            box = map_yolo_bbox_to_original(detection.bbox, transform);
        } else {
            const float half_w = detection.bbox.x2 * 0.5f;
            const float half_h = detection.bbox.y2 * 0.5f;
            box = {
                detection.bbox.x1 - half_w,
                detection.bbox.y1 - half_h,
                detection.bbox.x1 + half_w,
                detection.bbox.y1 + half_h,
            };
        }

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

void show_image(const std::string& window_name, const cv::Mat& image, int delay_ms) {
    if (image.empty()) {
        throw std::invalid_argument("show_image received an empty image");
    }

    cv::imshow(window_name, image);
    cv::waitKey(delay_ms);
}
