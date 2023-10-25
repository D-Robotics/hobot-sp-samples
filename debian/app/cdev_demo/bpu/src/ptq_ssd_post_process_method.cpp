#include "ptq_ssd_post_process_method.hpp"

std::vector<std::vector<Anchor>> anchors_table_;
float ssd_score_threshold_ = 0.25;
float ssd_nms_threshold_ = 0.45;
bool ssd_is_performance_ = true;
int ssd_nms_top_k_ = 200;

inline float fastExp(float x) {
  union {
    uint32_t i;
    float f;
  } v;
  v.i = (12102203.1616540672f * x + 1064807160.56887296f);
  return v.f;
}


SSDConfig default_ssd_config = {
    {0.1, 0.1, 0.2, 0.2},
    {0, 0, 0, 0},
    {0.5, 0.5},
    {15, 30, 60, 100, 150, 300},
    {{60, -1}, {105, 150}, {150, 195}, {195, 240}, {240, 285}, {285, 300}},
    {{2, 0.5, 0, 0},
     {2, 0.5, 3, 1.0 / 3},
     {2, 0.5, 3, 1.0 / 3},
     {2, 0.5, 3, 1.0 / 3},
     {2, 0.5, 3, 1.0 / 3},
     {2, 0.5, 3, 1.0 / 3}},
    0,
    20,
    {"aeroplane",   "bicycle", "bird",  "boaupdate", "bottle",
     "bus",         "car",     "cat",   "chair",     "cow",
     "diningtable", "dog",     "horse", "motorbike", "person",
     "pottedplant", "sheep",   "sofa",  "train",     "tvmonitor"}};


int SsdAnchors(std::vector<Anchor> &anchors,
                                        int layer,
                                        int layer_height,
                                        int layer_width) {
  int step = default_ssd_config.step[layer];
  float min_size = default_ssd_config.anchor_size[layer].first;
  float max_size = default_ssd_config.anchor_size[layer].second;
  auto &anchor_ratio = default_ssd_config.anchor_ratio[layer];
  for (int i = 0; i < layer_height; i++) {
    for (int j = 0; j < layer_width; j++) {
      float cy = (i + default_ssd_config.offset[0]) * step;
      float cx = (j + default_ssd_config.offset[1]) * step;
      anchors.emplace_back(Anchor(cx, cy, min_size, min_size));
      if (max_size > 0) {
        anchors.emplace_back(Anchor(cx,
                                    cy,
                                    std::sqrt(max_size * min_size),
                                    std::sqrt(max_size * min_size)));
      }
      for (int k = 0; k < 4; k++) {
        if (anchor_ratio[k] == 0) continue;
        float sr = std::sqrt(anchor_ratio[k]);
        float w = min_size * sr;
        float h = min_size / sr;
        anchors.emplace_back(Anchor(cx, cy, w, h));
      }
    }
  }
  return 0;
}

float DequantiScale(int32_t data,
                                             bool big_endian,
                                             float &scale_value) {
  return static_cast<float>(r_int32(data, big_endian)) * scale_value;
}

int GetBboxAndScoresQuantiNONE(
    hbDNNTensor *bbox_tensor,
    hbDNNTensor *cls_tensor,
    std::vector<Detection> &dets,
    std::vector<Anchor> &anchors,
    int class_num,
    bpu_image_info_t &image_info) {
  int *shape = cls_tensor->properties.validShape.dimensionSize;
  int32_t c_batch_size = shape[0];

  int32_t c_hnum = shape[1];
  int32_t c_wnum = shape[2];
  int32_t c_cnum = shape[3];
  uint32_t anchor_num_per_pixel = c_cnum / class_num;

  shape = bbox_tensor->properties.validShape.dimensionSize;
  int32_t b_batch_size = shape[0];

  int32_t b_hnum = shape[1];
  int32_t b_wnum = shape[2];
  int32_t b_cnum = shape[3];

  assert(anchor_num_per_pixel == b_cnum / 4);
  assert(c_batch_size == b_batch_size && c_hnum == b_hnum && c_wnum == b_wnum);
  auto box_num = b_batch_size * b_hnum * b_wnum * anchor_num_per_pixel;

  auto *raw_cls_data = reinterpret_cast<float *>(cls_tensor->sysMem[0].virAddr);
  auto *raw_box_data =
      reinterpret_cast<float *>(bbox_tensor->sysMem[0].virAddr);

  for (int i = 0; i < box_num; i++) {
    uint32_t res_id_cur_anchor = i * class_num;
    // get softmax sum
    double sum = 0;
    int max_id = 0;
    // TODO(@horizon.ai): fastExp only affect the final score value
    // confirm whether it affects the accuracy
    double background_score;
    bool is_performance_ = true;
    if (is_performance_) {
      background_score = fastExp(
          raw_cls_data[res_id_cur_anchor + default_ssd_config.background_index]);
    } else {
      background_score = std::exp(
          raw_cls_data[res_id_cur_anchor + default_ssd_config.background_index]);
    }

    double max_score = 0;
    for (int cls = 0; cls < class_num; ++cls) {
      float cls_score;
      if (is_performance_) {
        cls_score = fastExp(raw_cls_data[res_id_cur_anchor + cls]);
      } else {
        cls_score = std::exp(raw_cls_data[res_id_cur_anchor + cls]);
      }
      /* 1. scores should be larger than background score, or else will not be
      selected
      2. For Location for class_name, to add background to class-names list when
      background is not lastest
      3. For ssd_mobilenetv1_300x300_nv12.bin, background_index is 0, the value
      can be modified according to different model*/
      if (cls != default_ssd_config.background_index && cls_score > max_score &&
          cls_score > background_score) {
        max_id = cls;
        max_score = cls_score;
      }
      sum += cls_score;
    }
    // get softmax score
    max_score = max_score / sum;

    if (max_score <= ssd_score_threshold_) {
      continue;
    }

    int start = i * 4;
    float dx = raw_box_data[start];
    float dy = raw_box_data[start + 1];
    float dw = raw_box_data[start + 2];
    float dh = raw_box_data[start + 3];

    auto x_min = (anchors[i].cx - anchors[i].w / 2) / image_info.m_model_w;
    auto y_min = (anchors[i].cy - anchors[i].h / 2) / image_info.m_model_h;
    auto x_max = (anchors[i].cx + anchors[i].w / 2) / image_info.m_model_w;
    auto y_max = (anchors[i].cy + anchors[i].h / 2) / image_info.m_model_h;

    auto prior_w = x_max - x_min;
    auto prior_h = y_max - y_min;
    auto prior_center_x = (x_max + x_min) / 2;
    auto prior_center_y = (y_max + y_min) / 2;
    auto decode_x = default_ssd_config.std[0] * dx * prior_w + prior_center_x;
    auto decode_y = default_ssd_config.std[1] * dy * prior_h + prior_center_y;
    auto decode_w = std::exp(default_ssd_config.std[2] * dw) * prior_w;
    auto decode_h = std::exp(default_ssd_config.std[3] * dh) * prior_h;

    auto xmin_org = (decode_x - decode_w * 0.5) * image_info.m_ori_width;
    auto ymin_org = (decode_y - decode_h * 0.5) * image_info.m_ori_height;
    auto xmax_org = (decode_x + decode_w * 0.5) * image_info.m_ori_width;
    auto ymax_org = (decode_y + decode_h * 0.5) * image_info.m_ori_height;

    xmin_org = std::max(xmin_org, 0.0);
    xmax_org = std::min(xmax_org, image_info.m_ori_width - 1.0);
    ymin_org = std::max(ymin_org, 0.0);
    ymax_org = std::min(ymax_org, image_info.m_ori_height - 1.0);

    if (xmax_org <= 0 || ymax_org <= 0) continue;
    if (xmin_org > xmax_org || ymin_org > ymax_org) continue;

    Bbox bbox(xmin_org, ymin_org, xmax_org, ymax_org);
    dets.emplace_back(Detection(
        (int)max_id, max_score, bbox, default_ssd_config.class_names[max_id].c_str()));
  }
  return 0;
}

int GetBboxAndScoresQuantiSCALE(
    hbDNNTensor *bbox_tensor,
    hbDNNTensor *cls_tensor,
    std::vector<Detection> &dets,
    std::vector<Anchor> &anchors,
    int class_num,
    bpu_image_info_t &image_info) {
  int h_idx{1}, w_idx{2}, c_idx{3};

  // bbox
  int32_t bbox_n = bbox_tensor->properties.validShape.dimensionSize[0];
  int32_t bbox_h = bbox_tensor->properties.validShape.dimensionSize[h_idx];
  int32_t bbox_w = bbox_tensor->properties.validShape.dimensionSize[w_idx];
  int32_t bbox_c_valid =
      bbox_tensor->properties.validShape.dimensionSize[c_idx];
  int32_t bbox_c_aligned =
      bbox_tensor->properties.alignedShape.dimensionSize[c_idx];
  int32_t *bbox_data =
      reinterpret_cast<int32_t *>(bbox_tensor->sysMem[0].virAddr);
  float *bbox_scale = bbox_tensor->properties.scale.scaleData;

  // cls shape
  int32_t cls_n = cls_tensor->properties.validShape.dimensionSize[0];
  int32_t cls_h = cls_tensor->properties.validShape.dimensionSize[h_idx];
  int32_t cls_w = cls_tensor->properties.validShape.dimensionSize[w_idx];
  int32_t cls_c_valid = cls_tensor->properties.validShape.dimensionSize[c_idx];
  int32_t cls_c_aligned =
      cls_tensor->properties.alignedShape.dimensionSize[c_idx];
  int32_t *cls_data =
      reinterpret_cast<int32_t *>(cls_tensor->sysMem[0].virAddr);
  float *cls_scale = cls_tensor->properties.scale.scaleData;

  auto stride = cls_c_valid / class_num;
  auto bbox_num_pred = bbox_c_valid / stride;

  for (int h = 0; h < bbox_h; ++h) {
    for (int w = 0; w < bbox_w; ++w) {
      for (int k = 0; k < stride; ++k) {
        int32_t *cur_cls_data = cls_data + k * class_num;
        float *cur_cls_scale = cls_scale + k * class_num;
        float tmp = DequantiScale(cur_cls_data[0], false, cur_cls_scale[0]);
        bool is_performance_ = true;
        double background_score =
            is_performance_ ? fastExp(tmp) : std::exp(tmp);
        double sum = 0;
        int max_id = 0;
        double max_score = 0;
        for (int index = 0; index < class_num; ++index) {
          tmp = DequantiScale(cur_cls_data[index], false, cur_cls_scale[index]);

          float cls_score = is_performance_ ? fastExp(tmp) : std::exp(tmp);

          sum += cls_score;
          if (index != 0 && cls_score > max_score &&
              cls_score > background_score) {
            max_id = index - 1;
            max_score = cls_score;
          }
        }
        max_score = max_score / sum;

        if (max_score <= ssd_score_threshold_) {
          continue;
        }

        int32_t *cur_bbox_data = bbox_data + k * bbox_num_pred;
        float *cur_bbox_scale = bbox_scale + k * bbox_num_pred;
        float dx = DequantiScale(cur_bbox_data[0], false, cur_bbox_scale[0]);
        float dy = DequantiScale(cur_bbox_data[1], false, cur_bbox_scale[1]);
        float dw = DequantiScale(cur_bbox_data[2], false, cur_bbox_scale[2]);
        float dh = DequantiScale(cur_bbox_data[3], false, cur_bbox_scale[3]);

        int i = h * bbox_w * stride + w * stride + k;
        auto x_min = (anchors[i].cx - anchors[i].w / 2) / image_info.m_model_w;
        auto y_min = (anchors[i].cy - anchors[i].h / 2) / image_info.m_model_h;
        auto x_max = (anchors[i].cx + anchors[i].w / 2) / image_info.m_model_w;
        auto y_max = (anchors[i].cy + anchors[i].h / 2) / image_info.m_model_h;

        auto prior_w = x_max - x_min;
        auto prior_h = y_max - y_min;
        auto prior_center_x = (x_max + x_min) / 2;
        auto prior_center_y = (y_max + y_min) / 2;
        auto decode_x = default_ssd_config.std[0] * dx * prior_w + prior_center_x;
        auto decode_y = default_ssd_config.std[1] * dy * prior_h + prior_center_y;
        auto decode_w = std::exp(default_ssd_config.std[2] * dw) * prior_w;
        auto decode_h = std::exp(default_ssd_config.std[3] * dh) * prior_h;

        auto xmin_org = (decode_x - decode_w * 0.5) * image_info.m_ori_width;
        auto ymin_org = (decode_y - decode_h * 0.5) * image_info.m_ori_height;
        auto xmax_org = (decode_x + decode_w * 0.5) * image_info.m_ori_width;
        auto ymax_org = (decode_y + decode_h * 0.5) * image_info.m_ori_height;

        xmin_org = std::max(xmin_org, 0.0);
        xmax_org = std::min(xmax_org, image_info.m_ori_width - 1.0);
        ymin_org = std::max(ymin_org, 0.0);
        ymax_org = std::min(ymax_org, image_info.m_ori_height - 1.0);

        if (xmax_org <= 0 || ymax_org <= 0) continue;
        if (xmin_org > xmax_org || ymin_org > ymax_org) continue;

        Bbox bbox(xmin_org, ymin_org, xmax_org, ymax_org);
        dets.emplace_back(Detection((int)max_id,
                                    max_score,
                                    bbox,
                                    default_ssd_config.class_names[max_id].c_str()));
      }
      bbox_data = bbox_data + bbox_c_aligned;
      cls_data = cls_data + cls_c_aligned;
    }
  }
  return 0;
}

int GetBboxAndScores(hbDNNTensor *bbox_tensor,
                                              hbDNNTensor *cls_tensor,
                                              std::vector<Detection> &dets,
                                              std::vector<Anchor> &anchors,
                                              int class_num,
                                              bpu_image_info_t &image_info) {
  auto quanti_type = bbox_tensor->properties.quantiType;
  if (quanti_type == hbDNNQuantiType::SCALE) {
    return GetBboxAndScoresQuantiSCALE(
        bbox_tensor, cls_tensor, dets, anchors, class_num, image_info);
  } else if (quanti_type == hbDNNQuantiType::NONE) {
    return GetBboxAndScoresQuantiNONE(
        bbox_tensor, cls_tensor, dets, anchors, class_num, image_info);
  } else {
    printf("error quanti_type: %d\n", quanti_type);
    return -1;
  }
}

#define NMS_MAX_INPUT (400)
void ssd_nms(std::vector<Detection> &input,
         float iou_threshold,
         int top_k,
         std::vector<Detection> &result,
         bool suppress) {
  // sort order by score desc
  std::stable_sort(input.begin(), input.end(), std::greater<Detection>());
  if (input.size() > NMS_MAX_INPUT) {
    input.resize(NMS_MAX_INPUT);
  }

  std::vector<bool> skip(input.size(), false);

  // pre-calculate boxes area
  std::vector<float> areas;
  areas.reserve(input.size());
  for (size_t i = 0; i < input.size(); i++) {
    float width = input[i].bbox.xmax - input[i].bbox.xmin;
    float height = input[i].bbox.ymax - input[i].bbox.ymin;
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
      float xx1 = std::max(input[i].bbox.xmin, input[j].bbox.xmin);
      float yy1 = std::max(input[i].bbox.ymin, input[j].bbox.ymin);
      float xx2 = std::min(input[i].bbox.xmax, input[j].bbox.xmax);
      float yy2 = std::min(input[i].bbox.ymax, input[j].bbox.ymax);

      if (xx2 > xx1 && yy2 > yy1) {
        float area_intersection = (xx2 - xx1) * (yy2 - yy1);
        float iou_ratio =
            area_intersection / (areas[j] + areas[i] - area_intersection);
        if (iou_ratio > iou_threshold) {
          skip[j] = true;
        }
      }
    }
    result.push_back(input[i]);
  }
}

int SSDPostProcess(hbDNNTensor *tensors,
                                         bpu_image_info_t &image_info,
                                         std::vector<Detection> &ssd_det_restuls) {
  int layer_num = default_ssd_config.step.size();
  if (anchors_table_.empty()) {
    // Note: note thread safe
    anchors_table_.resize(layer_num);
    for (int i = 0; i < layer_num; i++) {
      std::vector<Anchor> &anchors = anchors_table_[i];
      int height = tensors[i * 2].properties.alignedShape.dimensionSize[1];
      int width = tensors[i * 2].properties.alignedShape.dimensionSize[2];
      SsdAnchors(anchors_table_[i], i, height, width);
    }
  }

  std::vector<Detection> dets;
  for (int i = 0; i < layer_num; i++) {
    std::vector<Anchor> &anchors = anchors_table_[i];
    GetBboxAndScores(&tensors[i * 2],
                     &tensors[i * 2 + 1],
                     dets,
                     anchors,
                     default_ssd_config.class_num + 1,
                     image_info);
  }
  ssd_nms(dets, ssd_nms_threshold_, ssd_nms_top_k_, ssd_det_restuls, false);
  return 0;
}
