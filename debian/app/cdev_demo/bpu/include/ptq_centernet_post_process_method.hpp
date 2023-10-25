#ifndef ptq_centernet_post_process_method
#define ptq_centernet_post_process_method

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
#include "ptq_centernet_post_process_method.hpp"
#include "fcos_post_process.hpp"

/**
 * Config definition for Centernet
 */
struct PTQCenternetConfig {
  int class_num;
  std::vector<std::string> class_names;
};


extern int CenternetPostProcess(hbDNNTensor *tensors, bpu_image_info_t &image_info, std::vector<Detection> &centernet_det_restuls, bool is_pad_resize);

#endif
