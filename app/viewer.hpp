#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

struct YoloBBox {
    float x1;
    float y1;
    float x2;
    float y2;
};

struct Detection {
    int class_id;
    float confidence;
    YoloBBox bbox;
};

struct LetterboxTransform {
    int original_width;
    int original_height;
    int target_width;
    int target_height;
    int resized_width;
    int resized_height;
    int pad_x;
    int pad_y;
    float scale;
};

LetterboxTransform make_letterbox_transform(int original_width,
                                            int original_height,
                                            int target_width = 640,
                                            int target_height = 640);

YoloBBox map_yolo_bbox_to_original(const YoloBBox& bbox, const LetterboxTransform& transform);

cv::Mat load_image_bgr(const std::string& path);
cv::Mat draw_detections(const cv::Mat& image,
                        const std::vector<Detection>& detections,
                        const LetterboxTransform& transform,
                        bool map_to_original = true);
void save_image(const std::string& path, const cv::Mat& image);
void show_image(const std::string& window_name, const cv::Mat& image, int delay_ms = 0);
