#include "ptq_centernet_post_process_method.hpp"

float centernet_score_threshold_ = 0.4;
int centernet_top_k_ = 50;

PTQCenternetConfig default_ptq_centernet_config = {
    80, {"person",        "bicycle",      "car",
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

struct DecodeData {
  float topk_score;
  int topk_inds;
  int topk_clses;
  float topk_ys;
  float topk_xs;

  friend bool operator>(const DecodeData &ldt, const DecodeData &rdt) {
    return (ldt.topk_clses > rdt.topk_clses);
  }
};

struct DataNode {
  float value;
  int indx;

  friend bool operator>(const DataNode &ldt, const DataNode &rdt) {
    return (ldt.value > rdt.value);
  }
};

static float fastExp(float x) {
  union {
    uint32_t i;
    float f;
  } v;
  v.i = (12102203.1616540672f * x + 1064807160.56887296f);
  return v.f;
}

// order topK data in node
static void top_k_helper(DataNode *node, int topk, int len) {
  std::priority_queue<int, std::vector<DataNode>, std::greater<DataNode>> heap;
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

#define BSWAP_32(x) static_cast<int32_t>(__builtin_bswap32(x))

#define r_int32(x, big_endian) \
  (big_endian) ? BSWAP_32((x)) : static_cast<int32_t>((x))

static float DequantiScale(int32_t data, bool big_endian, float &scale_value) {
  return static_cast<float>(r_int32(data, big_endian)) * scale_value;
}

int NMSMaxPool2dDequanti(hbDNNTensor &tensor,
                         std::vector<DataNode> &node,
                         float &t_value,
                         float *scale) {
  int h_index{2}, w_index{3}, c_index{1};
  int *shape = tensor.properties.validShape.dimensionSize;
  int input_c = shape[c_index];
  int input_h = shape[h_index];
  int input_w = shape[w_index];

  auto *raw_heat_map_data =
      reinterpret_cast<int32_t *>(tensor.sysMem[0].virAddr);

  std::vector<int> edge1 = {-1, +1, +input_w - 1, +input_w, +input_w + 1};

  std::vector<int> edge2 = {
      -1,
      +1,
      -input_w - 1,
      -input_w,
      -input_w + 1,
  };

  std::vector<int> edge3 = {
      +1,
      +input_w,
      +input_w + 1,
      -input_w,
      -input_w + 1,
  };

  std::vector<int> edge4 = {
      -1,
      +input_w - 1,
      +input_w,
      -input_w - 1,
      -input_w,
  };

  int pos[8] = {+1,
                -1,
                +input_w - 1,
                +input_w,
                +input_w + 1,
                -input_w - 1,
                -input_w,
                -input_w + 1};

  bool big_endian = false;

  for (int c = 0; c < input_c; c++) {
    int channel_offset = c * input_h * input_w;
    int32_t *iptr = raw_heat_map_data + channel_offset;
    float scale_value = scale[c];
    // VLOG(EXAMPLE_REPORT) << "c: " << c << "; scale1:" << scale[c];
    DataNode tmp_node;

    std::vector<int32_t> point1(4, 0.f);
    point1[0] = iptr[0];
    point1[1] = iptr[1];
    point1[2] = iptr[input_w + 0];
    point1[3] = iptr[input_w + 1];
    int i = 0;
    for (; i < 4; ++i) {
      if (point1[0] < point1[i]) break;
    }
    if (i == 4) {
      tmp_node.value = DequantiScale(point1[0], big_endian, scale_value);
      if (tmp_node.value > t_value) {
        tmp_node.indx = channel_offset;
        node.emplace_back(tmp_node);
      }
    }

    std::vector<int32_t> point2(4, 0.f);
    point2[0] = iptr[input_w - 2];
    point2[1] = iptr[input_w - 1];  //
    point2[2] = iptr[input_w + input_w - 2];
    point2[3] = iptr[input_w + input_w - 1];
    i = 0;
    for (; i < 4; ++i) {
      if (point2[1] < point2[i]) break;
    }
    if (i == 4) {
      tmp_node.value = DequantiScale(point2[1], big_endian, scale_value);  //
      if (tmp_node.value > t_value) {
        tmp_node.indx = channel_offset + input_w - 1;
        node.emplace_back(tmp_node);
      }
    }

    std::vector<int32_t> point3(4, 0.f);
    point3[0] = iptr[(input_h - 2) * input_w + 0];
    point3[1] = iptr[(input_h - 2) * input_w + 1];
    point3[2] = iptr[(input_h - 1) * input_w + 0];  //
    point3[3] = iptr[(input_h - 1) * input_w + 1];
    i = 0;
    for (; i < 4; ++i) {
      if (point3[2] < point3[i]) break;
    }
    if (i == 4) {
      tmp_node.value = DequantiScale(point3[2], big_endian, scale_value);
      if (tmp_node.value > t_value) {
        tmp_node.indx = channel_offset + (input_h - 1) * input_w;
        node.emplace_back(tmp_node);
      }
    }

    std::vector<int32_t> point4(4, 0.f);
    point4[0] = iptr[(input_h - 2) * input_w + (input_w - 2)];
    point4[1] = iptr[(input_h - 2) * input_w + (input_w - 1)];
    point4[2] = iptr[(input_h - 1) * input_w + (input_w - 2)];
    point4[3] = iptr[(input_h - 1) * input_w + (input_w - 1)];  //
    i = 0;
    for (; i < 4; ++i) {
      if (point4[3] < point4[i]) break;
    }
    if (i == 4) {
      tmp_node.value = DequantiScale(point4[3], big_endian, scale_value);
      if (tmp_node.value > t_value) {
        tmp_node.indx = channel_offset + (input_h - 1) * input_w + input_w - 1;
        node.emplace_back(tmp_node);
      }
    }

    // top + bottom
    for (int w = 1; w < input_w - 1; ++w) {
      i = 0;
      int32_t cur_value = iptr[w];
      for (; i < 5; ++i) {
        if (cur_value < iptr[w + edge1[i]]) break;
      }
      if (i == 5) {
        tmp_node.value = DequantiScale(cur_value, big_endian, scale_value);
        if (tmp_node.value > t_value) {
          tmp_node.indx = channel_offset + w;
          node.emplace_back(tmp_node);
        }
      }

      i = 0;
      int cur_pos = (input_h - 1) * input_w + w;
      cur_value = iptr[cur_pos];
      for (; i < 5; ++i) {
        if (cur_value < iptr[cur_pos + edge2[i]]) break;
      }
      if (i == 5) {
        tmp_node.value = DequantiScale(cur_value, big_endian, scale_value);
        if (tmp_node.value > t_value) {
          tmp_node.indx = channel_offset + cur_pos;
          node.emplace_back(tmp_node);
        }
      }
    }

    // left + right
    for (int h = 1; h < input_h - 1; ++h) {
      i = 0;
      int cur_pos = h * input_w;
      int32_t cur_value = iptr[cur_pos];
      for (; i < 5; ++i) {
        if (cur_value < iptr[cur_pos + edge3[i]]) break;
      }
      if (i == 5) {
        tmp_node.value = DequantiScale(cur_value, big_endian, scale_value);
        if (tmp_node.value > t_value) {
          tmp_node.indx = channel_offset + cur_pos;
          node.emplace_back(tmp_node);
        }
      }

      i = 0;
      cur_pos = h * input_w + input_w - 1;
      cur_value = iptr[cur_pos];
      for (; i < 5; ++i) {
        if (cur_value < iptr[cur_pos + edge3[i]]) break;
      }
      if (i == 5) {
        tmp_node.value = DequantiScale(cur_value, big_endian, scale_value);
        if (tmp_node.value > t_value) {
          tmp_node.indx = channel_offset + cur_pos;
          node.emplace_back(tmp_node);
        }
      }
    }

    // center
    for (int h = 1; h < input_h - 1; h++) {
      int offset = h * input_w;
      for (int w = 1; w < input_w - 1; w++) {
        int cur_pos = offset + w;
        int32_t cur_value = iptr[cur_pos];
        i = 0;
        for (; i < 8; ++i) {
          if (cur_value < iptr[cur_pos + pos[i]]) break;
        }
        if (i == 8) {
          tmp_node.value = DequantiScale(cur_value, big_endian, scale_value);
          if (tmp_node.value > t_value) {
            tmp_node.indx = channel_offset + cur_pos;
            node.emplace_back(tmp_node);
          }
        }
      }
    }
  }
  return 0;
}

int NMSMaxPool2d(hbDNNTensor &tensor,
                 std::vector<DataNode> &node,
                 float &t_value) {
  int h_index{2}, w_index{3}, c_index{1};
  int *shape = tensor.properties.validShape.dimensionSize;
  int input_c = shape[c_index];
  int input_h = shape[h_index];
  int input_w = shape[w_index];

  float *raw_heat_map_data =
      reinterpret_cast<float *>(tensor.sysMem[0].virAddr);

  std::vector<int> edge1 = {-1, +1, +input_w - 1, +input_w, +input_w + 1};

  std::vector<int> edge2 = {
      -1,
      +1,
      -input_w - 1,
      -input_w,
      -input_w + 1,
  };

  std::vector<int> edge3 = {
      +1,
      +input_w,
      +input_w + 1,
      -input_w,
      -input_w + 1,
  };

  std::vector<int> edge4 = {
      -1,
      +input_w - 1,
      +input_w,
      -input_w - 1,
      -input_w,
  };

  int pos[8] = {+1,
                -1,
                +input_w - 1,
                +input_w,
                +input_w + 1,
                -input_w - 1,
                -input_w,
                -input_w + 1};

  for (int c = 0; c < input_c; c++) {
    int channel_offset = c * input_h * input_w;
    float *iptr = raw_heat_map_data + channel_offset;
    DataNode tmp_node;

    std::vector<float> point1(4, 0.f);
    point1[0] = iptr[0];  //
    point1[1] = iptr[1];
    point1[2] = iptr[input_w + 0];
    point1[3] = iptr[input_w + 1];
    int i = 0;
    for (; i < 4; ++i) {
      if (point1[0] < point1[i]) break;
    }
    if (i == 4 && point1[0] > t_value) {
      tmp_node.indx = channel_offset;
      tmp_node.value = point1[0];
      node.emplace_back(tmp_node);
    }

    std::vector<float> point2(4, 0.f);
    point2[0] = iptr[input_w - 2];
    point2[1] = iptr[input_w - 1];  //
    point2[2] = iptr[input_w + input_w - 2];
    point2[3] = iptr[input_w + input_w - 1];
    i = 0;
    for (; i < 4; ++i) {
      if (point2[1] < point2[i]) break;
    }
    if (i == 4 && point2[1] > t_value) {
      tmp_node.indx = channel_offset + input_w - 1;
      tmp_node.value = point2[1];  //
      node.emplace_back(tmp_node);
    }

    std::vector<float> point3(4, 0.f);
    point3[0] = iptr[(input_h - 2) * input_w + 0];
    point3[1] = iptr[(input_h - 2) * input_w + 1];
    point3[2] = iptr[(input_h - 1) * input_w + 0];  //
    point3[3] = iptr[(input_h - 1) * input_w + 1];
    i = 0;
    for (; i < 4; ++i) {
      if (point3[2] < point3[i]) break;
    }
    if (i == 4 && point3[2] > t_value) {
      tmp_node.indx = channel_offset + (input_h - 1) * input_w;
      tmp_node.value = point3[2];
      node.emplace_back(tmp_node);
    }

    std::vector<float> point4(4, 0.f);
    point4[0] = iptr[(input_h - 2) * input_w + (input_w - 2)];
    point4[1] = iptr[(input_h - 2) * input_w + (input_w - 1)];
    point4[2] = iptr[(input_h - 1) * input_w + (input_w - 2)];
    point4[3] = iptr[(input_h - 1) * input_w + (input_w - 1)];  //
    i = 0;
    for (; i < 4; ++i) {
      if (point4[3] < point4[i]) break;
    }
    if (i == 4 && point4[3] > t_value) {
      tmp_node.indx = channel_offset + (input_h - 1) * input_w + input_w - 1;
      tmp_node.value = point4[3];
      node.emplace_back(tmp_node);
    }

    // top + bottom
    for (int w = 1; w < input_w - 1; ++w) {
      i = 0;
      float cur_value = iptr[w];
      for (; i < 5; ++i) {
        if (cur_value < iptr[w + edge1[i]]) break;
      }
      if (i == 5 && cur_value > t_value) {
        tmp_node.indx = channel_offset + w;
        tmp_node.value = cur_value;
        node.emplace_back(tmp_node);
      }

      i = 0;
      int cur_pos = (input_h - 1) * input_w + w;
      cur_value = iptr[cur_pos];
      for (; i < 5; ++i) {
        if (cur_value < iptr[cur_pos + edge2[i]]) break;
      }
      if (i == 5 && cur_value > t_value) {
        tmp_node.indx = channel_offset + cur_pos;
        tmp_node.value = cur_value;
        node.emplace_back(tmp_node);
      }
    }

    // left + right
    for (int h = 1; h < input_h - 1; ++h) {
      i = 0;
      int cur_pos = h * input_w;
      float cur_value = iptr[cur_pos];
      for (; i < 5; ++i) {
        if (cur_value < iptr[cur_pos + edge3[i]]) break;
      }
      if (i == 5 && cur_value > t_value) {
        tmp_node.indx = channel_offset + cur_pos;
        tmp_node.value = cur_value;
        node.emplace_back(tmp_node);
      }

      i = 0;
      cur_pos = h * input_w + input_w - 1;
      cur_value = iptr[cur_pos];
      for (; i < 5; ++i) {
        if (iptr[cur_pos] < iptr[cur_pos + edge4[i]]) break;
      }
      if (i == 5 && cur_value > t_value) {
        tmp_node.indx = channel_offset + cur_pos;
        tmp_node.value = cur_value;
        node.emplace_back(tmp_node);
      }
    }

    // center
    for (int h = 1; h < input_h - 1; h++) {
      int offset = h * input_w;
      for (int w = 1; w < input_w - 1; w++) {
        int cur_pos = offset + w;
        float cur_value = iptr[cur_pos];
        i = 0;
        for (; i < 8; ++i) {
          if (cur_value < iptr[cur_pos + pos[i]]) break;
        }
        if (i == 8 && cur_value > t_value) {
          tmp_node.indx = channel_offset + cur_pos;
          tmp_node.value = cur_value;
          node.emplace_back(tmp_node);
        }
      }
    }
  }
  return 0;
}


int CenternetPostProcess(hbDNNTensor *tensors, bpu_image_info_t &image_info, std::vector<Detection> &centernet_det_restuls, bool is_pad_resize) {

  int h_index{2}, w_index{3}, c_index{1};
  int *shape = tensors[0].properties.validShape.dimensionSize;
  int area = shape[w_index] * shape[w_index];

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

  std::vector<DataNode> node;
  float t_value =
      log(centernet_score_threshold_ / (1.f - centernet_score_threshold_));  // ln (2.f/3.f)
  if (quanti_type == hbDNNQuantiType::NONE) {
    NMSMaxPool2d(tensors[0], node, t_value);
  } else if (quanti_type == hbDNNQuantiType::SCALE) {
    auto &scales0 = tensors[0].properties.scale.scaleData;
    NMSMaxPool2dDequanti(tensors[0], node, t_value, scales0);
  } else {
    printf("centernet unsupport shift dequantzie now!\n");
    return -1;
  }

  int topk = node.size() > centernet_top_k_ ? centernet_top_k_ : node.size();
  if (topk != 0) top_k_helper(node.data(), topk, node.size());

  std::vector<float> reg_x(topk);
  std::vector<float> reg_y(topk);
  std::vector<Detection> tmp_box(topk);

  if (quanti_type == hbDNNQuantiType::NONE) {
    float *wh = reinterpret_cast<float *>(tensors[1].sysMem[0].virAddr);
    float *reg = reinterpret_cast<float *>(tensors[2].sysMem[0].virAddr);

    for (int i = 0; i < topk; i++) {
      // float topk_score = 1.0 / (1.0 + exp(-node[i].value));
      float topk_score = 1.0 / (1.0 + fastExp(-node[i].value));
      if (topk_score <= centernet_score_threshold_) {
        continue;
      }

      int topk_clses = node[i].indx / area;
      int topk_inds = node[i].indx % area;
      float topk_ys = static_cast<float>(topk_inds / shape[w_index]);
      float topk_xs = static_cast<float>(topk_inds % shape[w_index]);

      topk_xs += reg[topk_inds];
      topk_ys += reg[area + topk_inds];

      tmp_box[i].bbox.xmin = topk_xs - wh[topk_inds] / 2;
      tmp_box[i].bbox.xmax = topk_xs + wh[topk_inds] / 2;
      tmp_box[i].bbox.ymin = topk_ys - wh[area + topk_inds] / 2;
      tmp_box[i].bbox.ymax = topk_ys + wh[area + topk_inds] / 2;

      tmp_box[i].score = topk_score;
      tmp_box[i].id = topk_clses;
      tmp_box[i].class_name = default_ptq_centernet_config.class_names[topk_clses].c_str();
      centernet_det_restuls.push_back(tmp_box[i]);
    }

  } else if (quanti_type == hbDNNQuantiType::SCALE) {
    bool big_endian = false;
    int32_t *reg = reinterpret_cast<int32_t *>(tensors[2].sysMem[0].virAddr);
    int32_t *wh = reinterpret_cast<int32_t *>(tensors[1].sysMem[0].virAddr);

    for (int i = 0; i < topk; i++) {
      // float topk_score = 1.0 / (1.0 + exp(-node[i].value));
      float topk_score = 1.0 / (1.0 + fastExp(-node[i].value));
      if (topk_score <= centernet_score_threshold_) {
        continue;
      }

      int topk_clses = node[i].indx / area;
      int topk_inds = node[i].indx % area;
      float topk_ys = static_cast<float>(topk_inds / shape[w_index]);
      float topk_xs = static_cast<float>(topk_inds % shape[w_index]);

      auto &scales1 = tensors[1].properties.scale.scaleData;
      auto &scales2 = tensors[2].properties.scale.scaleData;
      topk_xs += DequantiScale(reg[topk_inds], big_endian, *scales2);
      topk_ys +=
          DequantiScale(reg[area + topk_inds], big_endian, *(scales2 + 1));

      float wh_0 = DequantiScale(wh[topk_inds], big_endian, *scales1);

      float wh_1 =
          DequantiScale(wh[area + topk_inds], big_endian, *(scales1 + 1));

      tmp_box[i].bbox.xmin = topk_xs - wh_0 / 2;
      tmp_box[i].bbox.xmax = topk_xs + wh_0 / 2;
      tmp_box[i].bbox.ymin = topk_ys - wh_1 / 2;
      tmp_box[i].bbox.ymax = topk_ys + wh_1 / 2;

      tmp_box[i].score = topk_score;
      tmp_box[i].id = topk_clses;
      tmp_box[i].class_name = default_ptq_centernet_config.class_names[topk_clses].c_str();
      centernet_det_restuls.push_back(tmp_box[i]);
    }
  } else {
    printf("centernet unsupport shift dequantzie now!\n");
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
