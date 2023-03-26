
#include "yolov5_post_process.hpp"

PTQYolo5Config yolo5_config_ = {
    {8, 16, 32},
    {{{10, 13}, {16, 30}, {33, 23}},
     {{30, 61}, {62, 45}, {59, 119}},
     {{116, 90}, {156, 198}, {373, 326}}},
    80,
    {"person", "bicycle", "car",
     "motorcycle", "airplane", "bus",
     "train", "truck", "boat",
     "traffic light", "fire hydrant", "stop sign",
     "parking meter", "bench", "bird",
     "cat", "dog", "horse",
     "sheep", "cow", "elephant",
     "bear", "zebra", "giraffe",
     "backpack", "umbrella", "handbag",
     "tie", "suitcase", "frisbee",
     "skis", "snowboard", "sports ball",
     "kite", "baseball bat", "baseball glove",
     "skateboard", "surfboard", "tennis racket",
     "bottle", "wine glass", "cup",
     "fork", "knife", "spoon",
     "bowl", "banana", "apple",
     "sandwich", "orange", "broccoli",
     "carrot", "hot dog", "pizza",
     "donut", "cake", "chair",
     "couch", "potted plant", "bed",
     "dining table", "toilet", "tv",
     "laptop", "mouse", "remote",
     "keyboard", "cell phone", "microwave",
     "oven", "toaster", "sink",
     "refrigerator", "book", "clock",
     "vase", "scissors", "teddy bear",
     "hair drier", "toothbrush"}};

void ParseTensor(std::shared_ptr<hbDNNTensor> tensor,
                 int layer,
                 std::vector<YoloV5Result> &results, bpu_image_info_t &image_info)
{
    //printf("start parse,tensor[0].vptr:0x%x\n",tensor->sysMem[0].virAddr);
    hbSysFlushMem(&(tensor->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
    int num_classes = yolo5_config_.class_num;
    int stride = yolo5_config_.strides[layer];
    int num_pred = yolo5_config_.class_num + 4 + 1;

    std::vector<float> class_pred(yolo5_config_.class_num, 0.0);
    std::vector<std::pair<double, double>> &anchors =
        yolo5_config_.anchors_table[layer];

    double h_ratio = image_info.m_model_h * 1.0 / image_info.m_ori_height;
    double w_ratio = image_info.m_model_w * 1.0 / image_info.m_ori_width;
    //double resize_ratio = std::min(w_ratio, h_ratio);

    //  int *shape = tensor->data_shape.d;
    int height = 0, width = 0;
    auto ret = get_tensor_hw(tensor, &height, &width);
    if (ret != 0)
    {
        printf("Yolo5_detection_parser\n");
    }

    int anchor_num = anchors.size();
    auto *data = reinterpret_cast<float *>(tensor->sysMem[0].virAddr);
    for (int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            for (int k = 0; k < anchor_num; k++)
            {
                double anchor_x = anchors[k].first;
                double anchor_y = anchors[k].second;
                float *cur_data = data + k * num_pred;
                float objness = cur_data[4];

                int id = argmax(cur_data + 5, cur_data + 5 + num_classes);
                double x1 = 1 / (1 + std::exp(-objness)) * 1;
                double x2 = 1 / (1 + std::exp(-cur_data[id + 5]));
                double confidence = x1 * x2;

                if (confidence < score_threshold_)
                {
                    continue;
                }

                float center_x = cur_data[0];
                float center_y = cur_data[1];
                float scale_x = cur_data[2];
                float scale_y = cur_data[3];

                double box_center_x =
                    ((1.0 / (1.0 + std::exp(-center_x))) * 2 - 0.5 + w) * stride;
                double box_center_y =
                    ((1.0 / (1.0 + std::exp(-center_y))) * 2 - 0.5 + h) * stride;

                double box_scale_x =
                    std::pow((1.0 / (1.0 + std::exp(-scale_x))) * 2, 2) * anchor_x;
                double box_scale_y =
                    std::pow((1.0 / (1.0 + std::exp(-scale_y))) * 2, 2) * anchor_y;

                double xmin = (box_center_x - box_scale_x / 2.0);
                double ymin = (box_center_y - box_scale_y / 2.0);
                double xmax = (box_center_x + box_scale_x / 2.0);
                double ymax = (box_center_y + box_scale_y / 2.0);
                // padding

                double w_padding =
                    (image_info.m_model_w - w_ratio * image_info.m_ori_width) / 2.0;
                double h_padding =
                    (image_info.m_model_h - h_ratio * image_info.m_ori_height) / 2.0;

                double xmin_org = (xmin - w_padding) / w_ratio;
                double xmax_org = (xmax - w_padding) / w_ratio;
                double ymin_org = (ymin - h_padding) / h_ratio;
                double ymax_org = (ymax - h_padding) / h_ratio;
                // padding
                if (xmax <= 0 || ymax <= 0)
                {
                    continue;
                }

                if (xmin > xmax || ymin > ymax)
                {
                    continue;
                }
                // padding
                xmin_org = std::max(xmin_org, 0.0);
                xmax_org = std::min(xmax_org, image_info.m_ori_width - 1.0);
                ymin_org = std::max(ymin_org, 0.0);
                ymax_org = std::min(ymax_org, image_info.m_ori_height - 1.0);

                results.emplace_back(
                    YoloV5Result(static_cast<int>(id),
                                 xmin_org,
                                 ymin_org,
                                 xmax_org,
                                 ymax_org,
                                 confidence,
                                 yolo5_config_.class_names[static_cast<int>(id)]));
                // padding
                //  results.emplace_back(
                //      YoloV5Result(static_cast<int>(id),
                //                   xmin,
                //                   ymin,
                //                   xmax,
                //                   ymax,
                //                   confidence,
                //                   yolo5_config_.class_names[static_cast<int>(id)]));
            }
            data = data + num_pred * anchors.size();
        }
    }
}

void yolo5_nms(std::vector<YoloV5Result> &input,
               float iou_threshold,
               int top_k,
               std::vector<std::shared_ptr<YoloV5Result>> &result,
               bool suppress)
{
    //printf("start nms\n");

    // sort order by score desc
    std::stable_sort(input.begin(), input.end(), std::greater<YoloV5Result>());

    std::vector<bool> skip(input.size(), false);

    // pre-calculate boxes area
    std::vector<float> areas;
    areas.reserve(input.size());
    for (size_t i = 0; i < input.size(); i++)
    {
        float width = input[i].xmax - input[i].xmin;
        float height = input[i].ymax - input[i].ymin;
        areas.push_back(width * height);
    }

    int count = 0;
    for (size_t i = 0; count < top_k && i < skip.size(); i++)
    {
        if (skip[i])
        {
            continue;
        }
        skip[i] = true;
        ++count;

        for (size_t j = i + 1; j < skip.size(); ++j)
        {
            if (skip[j])
            {
                continue;
            }
            if (suppress == false)
            {
                if (input[i].id != input[j].id)
                {
                    continue;
                }
            }

            // intersection area
            float xx1 = std::max(input[i].xmin, input[j].xmin);
            float yy1 = std::max(input[i].ymin, input[j].ymin);
            float xx2 = std::min(input[i].xmax, input[j].xmax);
            float yy2 = std::min(input[i].ymax, input[j].ymax);

            if (xx2 > xx1 && yy2 > yy1)
            {
                float area_intersection = (xx2 - xx1) * (yy2 - yy1);
                float iou_ratio =
                    area_intersection / (areas[j] + areas[i] - area_intersection);
                if (iou_ratio > iou_threshold)
                {
                    skip[j] = true;
                }
            }
        }

        auto yolo_res = std::make_shared<YoloV5Result>(input[i].id,
                                                       input[i].xmin,
                                                       input[i].ymin,
                                                       input[i].xmax,
                                                       input[i].ymax,
                                                       input[i].score,
                                                       input[i].class_name);
        if (!yolo_res)
        {
            printf("Yolo5_detection_parser");
        }

        result.push_back(yolo_res);
    }
}

int get_tensor_hw(std::shared_ptr<hbDNNTensor> tensor, int *height, int *width)
{
    int h_index = 0;
    int w_index = 0;
    if (tensor->properties.tensorLayout == HB_DNN_LAYOUT_NHWC)
    {
        h_index = 1;
        w_index = 2;
    }
    else if (tensor->properties.tensorLayout == HB_DNN_LAYOUT_NCHW)
    {
        h_index = 2;
        w_index = 3;
    }
    else
    {
        return -1;
    }
    *height = tensor->properties.validShape.dimensionSize[h_index];
    *width = tensor->properties.validShape.dimensionSize[w_index];
    return 0;
}