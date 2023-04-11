#ifndef yolov5_post
#define yolov5_post

#include <future>
#include <cmath>
#include <algorithm>
#include <vector>
#include "sp_bpu.h"
#include "yolov5_post_process.hpp"
#include <opencv2/opencv.hpp>

typedef struct
{
    int x_offset;
    int y_offset;
    int width_offset;
    int height_offset;
} ori_image;

struct PTQYolo5Config
{
    std::vector<int> strides;
    std::vector<std::vector<std::pair<double, double>>> anchors_table;
    int class_num;
    std::vector<std::string> class_names;
    std::vector<std::vector<float>> dequantize_scale;
};




template <class ForwardIterator>
inline size_t argmax(ForwardIterator first, ForwardIterator last)
{
    return std::distance(first, std::max_element(first, last));
}

struct YoloV5Result
{
    // 目标类别ID
    int id;
    // 目标检测框
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    // 检测结果的置信度
    float score;
    // 目标类别
    std::string class_name;

    YoloV5Result(int id_,
                 float xmin_,
                 float ymin_,
                 float xmax_,
                 float ymax_,
                 float score_,
                 std::string class_name_)
        : id(id_),
          xmin(xmin_),
          ymin(ymin_),
          xmax(xmax_),
          ymax(ymax_),
          score(score_),
          class_name(class_name_) {}

    friend bool operator>(const YoloV5Result &lhs, const YoloV5Result &rhs)
    {
        return (lhs.score > rhs.score);
    }
};

const float score_threshold_ = 0.4;
const float nms_threshold_ = 0.5;
const int nms_top_k_ = 5000;


extern void ParseTensor(std::shared_ptr<hbDNNTensor> tensor,
                 int layer,
                 std::vector<YoloV5Result> &results,bpu_image_info_t &image_info);

extern void yolo5_nms(std::vector<YoloV5Result> &input,
               float iou_threshold,
               int top_k,
               std::vector<std::shared_ptr<YoloV5Result>> &result,
               bool suppress);

extern int get_tensor_hw(std::shared_ptr<hbDNNTensor> tensor, int *height, int *width);

//extern void get_ori_image(uint8_t *addr, int ori_height, int ori_width, int model_height, int model_width, std::vector<std::shared_ptr<YoloV5Result>> results);

#endif