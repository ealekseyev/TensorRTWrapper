#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

struct ImageCHW {
    std::shared_ptr<std::uint8_t[]> data;
    int channels;
    int height;
    int width;
};

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

ImageCHW load_image_chw(const std::string& path);
std::shared_ptr<std::uint8_t[]> resize_to_640x640_padded(const std::uint8_t* input_chw,
                                                         int input_height,
                                                         int input_width);
std::vector<float> chw_u8_to_float_input(const std::uint8_t* input_chw, std::size_t element_count);

LetterboxTransform make_letterbox_transform(int original_width,
                                            int original_height,
                                            int target_width = 640,
                                            int target_height = 640);
YoloBBox yolo_bbox_to_corners(const YoloBBox& bbox);
YoloBBox map_yolo_bbox_to_original(const YoloBBox& bbox, const LetterboxTransform& transform);

cv::Mat load_image_bgr(const std::string& path);
cv::Mat draw_detections(const cv::Mat& image,
                        const std::vector<Detection>& detections,
                        const LetterboxTransform& transform,
                        bool map_to_original = true);
void save_image(const std::string& path, const cv::Mat& image);
