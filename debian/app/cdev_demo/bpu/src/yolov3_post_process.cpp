
#include "yolov3_post_process.hpp"

PTQYolo3Config yolo3_config_ = {
    {32, 16, 8},
    {{{3.625, 2.8125}, {4.875, 6.1875}, {11.65625, 10.1875}},
     {{1.875, 3.8125}, {3.875, 2.8125}, {3.6875, 7.4375}},
     {{1.25, 1.625}, {2.0, 3.75}, {4.125, 2.875}}},
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

void yolov3_ParseTensor(std::shared_ptr<hbDNNTensor> tensor,
                 int layer,
                 std::vector<YoloV3Result> &results, bpu_image_info_t &image_info) {
  hbSysFlushMem(&(tensor->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
  auto *data = reinterpret_cast<float *>(tensor->sysMem[0].virAddr);
  int num_classes = yolo3_config_.class_num;
  int stride = yolo3_config_.strides[layer];
  int num_pred = yolo3_config_.class_num + 4 + 1;

  std::vector<float> class_pred(yolo3_config_.class_num, 0.0);
  std::vector<std::pair<double, double>> &anchors =
      yolo3_config_.anchors_table[layer];

  double h_ratio = image_info.m_model_h * 1.0 / image_info.m_ori_height;
  double w_ratio = image_info.m_model_w * 1.0 / image_info.m_ori_width;

  int height = 0, width = 0;
  auto ret = yolov3_get_tensor_hw(tensor, &height, &width);
  if (ret != 0) {
    printf("yolov3 get_tensor_hw failed\n");
  }

  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      for (size_t k = 0; k < anchors.size(); k++) {
        double anchor_x = anchors[k].first;
        double anchor_y = anchors[k].second;
        float *cur_data = data + k * num_pred;
        float objness = cur_data[4];
        for (int index = 0; index < num_classes; ++index) {
          class_pred[index] = cur_data[5 + index];
        }

        float id = yolov3_argmax(class_pred.begin(), class_pred.end());
        double x1 = 1 / (1 + std::exp(-objness)) * 1;
        double x2 = 1 / (1 + std::exp(-class_pred[id]));
        double confidence = x1 * x2;

        if (confidence < yolov3_score_threshold_) {
          continue;
        }

        float center_x = cur_data[0];
        float center_y = cur_data[1];
        float scale_x = cur_data[2];
        float scale_y = cur_data[3];

        double box_center_x =
            ((1.0 / (1.0 + std::exp(-center_x))) + w) * stride;
        double box_center_y =
            ((1.0 / (1.0 + std::exp(-center_y))) + h) * stride;

        double box_scale_x = std::exp(scale_x) * anchor_x * stride;
        double box_scale_y = std::exp(scale_y) * anchor_y * stride;

        double xmin = (box_center_x - box_scale_x / 2.0);
        double ymin = (box_center_y - box_scale_y / 2.0);
        double xmax = (box_center_x + box_scale_x / 2.0);
        double ymax = (box_center_y + box_scale_y / 2.0);

        if (xmin > xmax || ymin > ymax) {
          continue;
        }
        
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
        xmin_org = std::max(xmin_org, 0.0);
        xmax_org = std::min(xmax_org, image_info.m_ori_width - 1.0);
        ymin_org = std::max(ymin_org, 0.0);
        ymax_org = std::min(ymax_org, image_info.m_ori_height - 1.0);

        results.push_back(
            YoloV3Result(static_cast<int>(id),
                    xmin_org,
                    ymin_org,
                    xmax_org,
                    ymax_org,
                    confidence,
                    yolo3_config_.class_names[static_cast<int>(id)].c_str()));
      }
      data = data + num_pred * anchors.size();
    }
  }
}

void yolo3_nms(std::vector<YoloV3Result> &input,
               float iou_threshold,
               int top_k,
               std::vector<std::shared_ptr<YoloV3Result>> &result,
               bool suppress) {
  // sort order by score desc
  std::stable_sort(input.begin(), input.end(), std::greater<YoloV3Result>());

  std::vector<bool> skip(input.size(), false);

  // pre-calculate boxes area
  std::vector<float> areas;
  areas.reserve(input.size());
  for (size_t i = 0; i < input.size(); i++) {
    float width = input[i].xmax - input[i].xmin;
    float height = input[i].ymax - input[i].ymin;
    areas.push_back(width * height);
  }

  int count = 0;
  for (size_t i = 0; count < top_k && i < skip.size(); i++) {
    if (skip[i]) {
      continue;
    }
    skip[i] = true;
    ++count;

    for (size_t j = i + 1; j < skip.size(); ++j) {
      if (skip[j]) {
        continue;
      }
      if (suppress == false) {
        if (input[i].id != input[j].id) {
          continue;
        }
      }

      // intersection area
      float xx1 = std::max(input[i].xmin, input[j].xmin);
      float yy1 = std::max(input[i].ymin, input[j].ymin);
      float xx2 = std::min(input[i].xmax, input[j].xmax);
      float yy2 = std::min(input[i].ymax, input[j].ymax);

      if (xx2 > xx1 && yy2 > yy1) {
        float area_intersection = (xx2 - xx1) * (yy2 - yy1);
        float iou_ratio =
            area_intersection / (areas[j] + areas[i] - area_intersection);
        if (iou_ratio > iou_threshold) {
          skip[j] = true;
        }
      }
    }

    auto yolo_res = std::make_shared<YoloV3Result>(input[i].id,
                                                input[i].xmin,
                                                input[i].ymin,
                                                input[i].xmax,
                                                input[i].ymax,
                                                input[i].score,
                                                input[i].class_name);

    result.push_back(yolo_res);
  }
}

int yolov3_get_tensor_hw(std::shared_ptr<hbDNNTensor> tensor, int *height, int *width)
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