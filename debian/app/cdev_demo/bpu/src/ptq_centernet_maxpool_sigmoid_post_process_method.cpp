#include "ptq_centernet_maxpool_sigmoid_post_process_method.hpp"

static float centernet_maxpool_sigmoid_score_threshold_ = 0.1;
static int centernet_maxpool_sigmoid_top_k_ = 100;

PTQCenternetMaxPoolSigmoidConfig default_ptq_centernet_maxpool_sigmoid_config =
    {80, {"person",        "bicycle",      "car",
          "motorcycle",    "airplane",     "bus",
          "train",         "truck",        "boat",
          "traffic light", "fire hydrant", "stop sign",
          "parking meter", "bench",        "bird",
          "cat",           "dog",          "horse",
          "sheep",         "cow",          "elephant",
          "bear",          "zebra",        "giraffe",
          "backpack",      "umbrella",     "handbag",
          "tie",           "suitcase",     "frisbee",
          "skis",          "snowboard",    "sports ball",
          "kite",          "baseball bat", "baseball glove",
          "skateboard",    "surfboard",    "tennis racket",
          "bottle",        "wine glass",   "cup",
          "fork",          "knife",        "spoon",
          "bowl",          "banana",       "apple",
          "sandwich",      "orange",       "broccoli",
          "carrot",        "hot dog",      "pizza",
          "donut",         "cake",         "chair",
          "couch",         "potted plant", "bed",
          "dining table",  "toilet",       "tv",
          "laptop",        "mouse",        "remote",
          "keyboard",      "cell phone",   "microwave",
          "oven",          "toaster",      "sink",
          "refrigerator",  "book",         "clock",
          "vase",          "scissors",     "teddy bear",
          "hair drier",    "toothbrush"}};

struct Centernet_DataNode {
  float value;
  int indx;

  friend bool operator>(const Centernet_DataNode &ldt, const Centernet_DataNode &rdt) {
    return (ldt.value > rdt.value);
  }
};

// order topK data in node
static void top_k_helper(Centernet_DataNode *node, int topk, int len) {
  std::priority_queue<int, std::vector<Centernet_DataNode>, std::greater<Centernet_DataNode>> heap;
  int i = 0;
  while (i < len) {
    if (i < topk) {
      heap.push(node[i]);
    } else {
      if (heap.top().value < node[i].value) {
        heap.pop();
        heap.push(node[i]);
      }
    }
    i++;
  }
  for (int j = 0; j < topk; j++) {
    node[j] = heap.top();
    heap.pop();
  }
}

template <typename DType>
float quanti_scale_function(DType data, float scale) {
  return static_cast<float>(data) * scale;
}

int filter_func(hbDNNTensor &tensor,
                std::vector<Centernet_DataNode> &node,
                float &t_value,
                float *scale) {
  int h_index{2}, w_index{3}, c_index{1};
  int *shape = tensor.properties.validShape.dimensionSize;
  int input_c = shape[c_index];
  int input_h = shape[h_index];
  int input_w = shape[w_index];

  int num_elements = input_c * input_h * input_w;
  int16_t *raw_heat_map_data =
      reinterpret_cast<int16_t *>(tensor.sysMem[0].virAddr);

  // per-tensor way
  float threshold = t_value / scale[0];
  Centernet_DataNode tmp_node;
  for (int i = 0; i < num_elements; ++i) {
    // compare with int16 data
    if (raw_heat_map_data[i] > static_cast<int16_t>(threshold)) {
      // dequantize
      tmp_node.value = quanti_scale_function(raw_heat_map_data[i], scale[0]);
      tmp_node.indx = i;
      node.emplace_back(tmp_node);
    }
  }
  return 0;
}

int CenternetMaxPoolSigmoidPostProcess(hbDNNTensor *tensors, bpu_image_info_t &image_info, std::vector<Detection> &centernet_det_restuls, bool is_pad_resize) {

  int h_index{2}, w_index{3}, c_index{1};
  int *shape = tensors[0].properties.validShape.dimensionSize;
  int area = shape[h_index] * shape[w_index];

  int origin_height = image_info.m_ori_height;
  int origin_width = image_info.m_ori_width;
  float scale_x = 1.0;
  float scale_y = 1.0;
  float offset_x = 0.0;
  float offset_y = 0.0;
  if (is_pad_resize) {
    float pad_len = origin_height > origin_width ? origin_height : origin_width;
    scale_x = pad_len / static_cast<float>(shape[w_index]);
    scale_y = pad_len / static_cast<float>(shape[h_index]);
    offset_x = (pad_len - origin_width) / 2;
    offset_y = (pad_len - origin_height) / 2;
  } else {
    scale_x = origin_width / 128.0;
    scale_y = origin_height / 128.0;
  }

  // Determine whether the model contains a dequnatize node by the first tensor
  auto quanti_type = tensors[0].properties.quantiType;

  std::vector<Centernet_DataNode> node;

  if (quanti_type == hbDNNQuantiType::SCALE) {
    auto &scales0 = tensors[0].properties.scale.scaleData;
    // filter score with dequantize
    filter_func(tensors[0], node, centernet_maxpool_sigmoid_score_threshold_, scales0);
  } else {
    printf("centernet unsupport shift dequantzie now!\n");
    return -1;
  }

  // topk sort
  int topk = node.size() > centernet_maxpool_sigmoid_top_k_ ? centernet_maxpool_sigmoid_top_k_ : node.size();
  if (topk != 0) top_k_helper(node.data(), topk, node.size());

  std::vector<float> reg_x(topk);
  std::vector<float> reg_y(topk);
  std::vector<Detection> tmp_box(topk);

  if (tensors[1].properties.quantiType == hbDNNQuantiType::SCALE) {
    int32_t *wh = reinterpret_cast<int32_t *>(tensors[1].sysMem[0].virAddr);
    int32_t *reg = reinterpret_cast<int32_t *>(tensors[2].sysMem[0].virAddr);
    for (int i = 0; i < topk; i++) {
      float topk_score = node[i].value;

      if (topk_score <= centernet_maxpool_sigmoid_score_threshold_) {
        continue;
      }

      // bbox decode with dequantize
      int topk_clses = node[i].indx / area;
      int topk_inds = node[i].indx % area;
      float topk_ys = static_cast<float>(topk_inds / shape[h_index]);
      float topk_xs = static_cast<float>(topk_inds % shape[w_index]);

      auto &wh_scale = tensors[1].properties.scale.scaleData;
      auto &reg_scale = tensors[2].properties.scale.scaleData;
      // per-channel way
      topk_xs += quanti_scale_function(reg[topk_inds], *reg_scale);
      topk_ys += quanti_scale_function(reg[area + topk_inds], *(reg_scale + 1));
      // per-channel way
      float wh_0 = quanti_scale_function(wh[topk_inds], *wh_scale);
      float wh_1 = quanti_scale_function(wh[area + topk_inds], *(wh_scale + 1));

      tmp_box[i].bbox.xmin = topk_xs - wh_0 / 2;
      tmp_box[i].bbox.xmax = topk_xs + wh_0 / 2;
      tmp_box[i].bbox.ymin = topk_ys - wh_1 / 2;
      tmp_box[i].bbox.ymax = topk_ys + wh_1 / 2;

      tmp_box[i].score = topk_score;
      tmp_box[i].id = topk_clses;
      tmp_box[i].class_name = default_ptq_centernet_maxpool_sigmoid_config.class_names[topk_clses].c_str();
      centernet_det_restuls.push_back(tmp_box[i]);
    }
  } else {
    printf("centernet unsupport now!\n");
    return -1;
  }

  auto &detections = centernet_det_restuls;
  int det_num = centernet_det_restuls.size();
  printf("qat-det.size(): %d\n", det_num);
  for (int i = 0; i < det_num; i++) {
    detections[i].bbox.xmin = detections[i].bbox.xmin * scale_x - offset_x;
    detections[i].bbox.xmax = detections[i].bbox.xmax * scale_x - offset_x;
    detections[i].bbox.ymin = detections[i].bbox.ymin * scale_y - offset_y;
    detections[i].bbox.ymax = detections[i].bbox.ymax * scale_y - offset_y;
  }

  return 0;
}
