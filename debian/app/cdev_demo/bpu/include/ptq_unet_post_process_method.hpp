#ifndef ptq_unet_post_process_method
#define ptq_unet_post_process_method

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
#include "ptq_unet_post_process_method.hpp"


typedef struct Segmentation {
  std::vector<int8_t> seg;
  int32_t num_classes = 0;
  int32_t width = 0;
  int32_t height = 0;
}Segmentation;

extern void UnetPostProcess(hbDNNTensor *tensors, bpu_image_info_t &image_info, Segmentation &unet_restuls);


#endif  // ptq_unet_post_process_method
