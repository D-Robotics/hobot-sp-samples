#ifndef yolov3_post
#define yolov3_post

#include <future>
#include <cmath>
#include <algorithm>
#include <vector>
#include "sp_bpu.h"
#include "yolov3_post_process.hpp"
#include <opencv2/opencv.hpp>

typedef struct
{
    int x_offset;
    int y_offset;
    int width_offset;
    int height_offset;
} yolov3_ori_image;

struct PTQYolo3Config
{
    std::vector<int> strides;
    std::vector<std::vector<std::pair<double, double>>> anchors_table;
    int class_num;
    std::vector<std::string> class_names;
    std::vector<std::vector<float>> dequantize_scale;
};


template <class ForwardIterator>
inline size_t yolov3_argmax(ForwardIterator first, ForwardIterator last)
{
    return std::distance(first, std::max_element(first, last));
}

struct YoloV3Result
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

    YoloV3Result(int id_,
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

    friend bool operator>(const YoloV3Result &lhs, const YoloV3Result &rhs)
    {
        return (lhs.score > rhs.score);
    }
};

const float yolov3_score_threshold_ = 0.3;
const float yolov3_nms_threshold_ = 0.45;
const int yolov3_nms_top_k_ = 500;
const int yolov3_output_nums_ = 3;


extern void yolov3_ParseTensor(std::shared_ptr<hbDNNTensor> tensor,
                 int layer,
                 std::vector<YoloV3Result> &results, bpu_image_info_t &image_info);

extern void yolo3_nms(std::vector<YoloV3Result> &input,
               float iou_threshold,
               int top_k,
               std::vector<std::shared_ptr<YoloV3Result>> &result,
               bool suppress);

extern int yolov3_get_tensor_hw(std::shared_ptr<hbDNNTensor> tensor, int *height, int *width);

//extern void get_ori_image(uint8_t *addr, int ori_height, int ori_width, int model_height, int model_width, std::vector<std::shared_ptr<YoloV3Result>> results);

#endif