#ifndef ptq_ssd_post_process_method
#define ptq_ssd_post_process_method

#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <queue>
#include <arm_neon.h>
#include <cassert>
#include "sp_bpu.h"
#include <opencv2/opencv.hpp>
#include "ptq_ssd_post_process_method.hpp"
#include "fcos_post_process.hpp"

#define BSWAP_32(x) static_cast<int32_t>(__builtin_bswap32(x))

#define r_int32(x, big_endian) \
  (big_endian) ? BSWAP_32((x)) : static_cast<int32_t>((x))

typedef struct Anchor {
  float cx{0.0};
  float cy{0.0};
  float w{0.0};
  float h{0.0};
  Anchor(float cx, float cy, float w, float h) : cx(cx), cy(cy), w(w), h(h) {}

  friend std::ostream &operator<<(std::ostream &os, const Anchor &anchor) {
    os << "[" << anchor.cx << "," << anchor.cy << "," << anchor.w << ","
       << anchor.h << "]";
    return os;
  }
} Anchor;

/**
 * Config definition for SSD
 */
struct SSDConfig {
  std::vector<float> std;
  std::vector<float> mean;
  std::vector<float> offset;
  std::vector<int> step;
  std::vector<std::pair<float, float>> anchor_size;
  std::vector<std::vector<float>> anchor_ratio;
  int background_index;
  int class_num;
  std::vector<std::string> class_names;
};

/**
 * Default ssd config
 * std: [0.1, 0.1, 0.2, 0.2]
 * mean: [0, 0, 0, 0]
 * offset: [0.5, 0.5]
 * step: [15, 30, 60, 100, 150, 300]
 * anchor_size: [[60, -1], [105, 150], [150, 195],
 *              [195, 240], [240, 285], [285,300]]
 * anchor_ratio: [[2, 0.5, 0, 0], [2, 0.5, 3, 1.0 / 3],
 *              [2, 0.5, 3, 1.0 / 3], [2, 0.5, 3, 1.0 / 3],
 *              [2, 0.5, 1.0 / 3], [2, 0.5, 1.0 / 3]]
 * background_index 0
 * class_num: 20
 * class_names: ["aeroplane",   "bicycle", "bird",  "boaupdate", "bottle",
     "bus",         "car",     "cat",   "chair",     "cow",
     "diningtable", "dog",     "horse", "motorbike", "person",
     "pottedplant", "sheep",   "sofa",  "train",     "tvmonitor"]
 */
extern SSDConfig default_ssd_config;


extern int SSDPostProcess(hbDNNTensor *tensors,
                bpu_image_info_t &image_info, std::vector<Detection> &ssd_det_restuls);

int SsdAnchors(std::vector<Anchor> &anchors,
                int layer,
                int layer_height,
                int layer_width);

int GetBboxAndScores(hbDNNTensor *c_tensor,
                      hbDNNTensor *bbox_tensor,
                      std::vector<Detection> &dets,
                      std::vector<Anchor> &anchors,
                      int class_num,
                      bpu_image_info_t &image_info);

int GetBboxAndScoresQuantiNONE(hbDNNTensor *c_tensor,
                                hbDNNTensor *bbox_tensor,
                                std::vector<Detection> &dets,
                                std::vector<Anchor> &anchors,
                                int class_num,
                                bpu_image_info_t &image_info);

int GetBboxAndScoresQuantiSCALE(hbDNNTensor *c_tensor,
                                hbDNNTensor *bbox_tensor,
                                std::vector<Detection> &dets,
                                std::vector<Anchor> &anchors,
                                int class_num,
                                bpu_image_info_t &image_info);

float DequantiScale(int32_t data, bool big_endian, float &scale_value);

#endif 
