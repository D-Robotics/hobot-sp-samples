#include "ptq_unet_post_process_method.hpp"

static  int num_classes_ = 20;

static inline uint32x4x4_t CalculateIndex(uint32_t idx,
                                          float32x4_t a,
                                          float32x4_t b,
                                          uint32x4x4_t c) {
  uint32x4_t mask{0};
  mask = vcltq_f32(b, a);
  uint32x4_t vec_idx = {idx, idx + 1, idx + 2, idx + 3};
  uint32x4x4_t res = {{vbslq_u32(mask, vec_idx, c.val[0]), 0, 0, 0}};
  return res;
}

static inline float32x2_t CalculateMax(float32x4_t in) {
  auto pmax = vpmax_f32(vget_high_f32(in), vget_low_f32(in));
  return vpmax_f32(pmax, pmax);
}

static inline uint32_t CalculateVectorIndex(uint32x4x4_t vec_res_idx,
                                            float32x4_t vec_res_value) {
  uint32x4_t res_idx_mask{0};
  uint32x4_t mask_ones = vdupq_n_u32(0xFFFFFFFF);

  auto pmax = CalculateMax(vec_res_value);
  auto mask = vceqq_f32(vec_res_value, vcombine_f32(pmax, pmax));
  res_idx_mask = vandq_u32(vec_res_idx.val[0], mask);
  res_idx_mask = vaddq_u32(res_idx_mask, mask_ones);
  auto pmin =
      vpmin_u32(vget_high_u32(res_idx_mask), vget_low_u32(res_idx_mask));
  pmin = vpmin_u32(pmin, pmin);
  uint32_t res = vget_lane_u32(pmin, 0);
  return (res - 0xFFFFFFFF);
}

static std::pair<float, int> MaxScoreID(int32_t *input,
                                        float *scale,
                                        int length) {
  float init_res_value = input[0] * scale[0];
  float32x4_t vec_res_value = vdupq_n_f32(init_res_value);
  uint32x4x4_t vec_res_idx{{0}};
  int i = 0;
  for (; i <= (length - 4); i += 4) {
    int32x4_t vec_input = vld1q_s32(input + i);
    float32x4_t vec_scale = vld1q_f32(scale + i);

    float32x4_t vec_elements = vmulq_f32(vcvtq_f32_s32(vec_input), vec_scale);
    float32x4_t temp_vec_res_value = vmaxq_f32(vec_elements, vec_res_value);
    vec_res_idx =
        CalculateIndex(i, temp_vec_res_value, vec_res_value, vec_res_idx);
    vec_res_value = temp_vec_res_value;
  }

  uint32_t idx = CalculateVectorIndex(vec_res_idx, vec_res_value);
  float res = vget_lane_f32(CalculateMax(vec_res_value), 0);

  // Compute left elements
  for (; i < length; ++i) {
    float score = input[i] * scale[i];
    if (score > res) {
      idx = i;
      res = score;
    }
  }
  std::pair<float, int> result_id_score = {res, idx};
  return result_id_score;
}


int PostProcessNone(hbDNNTensor *tensors, bpu_image_info_t &image_info, Segmentation &unet_restuls) {

  int height = tensors->properties.validShape.dimensionSize[1];
  int width = tensors->properties.validShape.dimensionSize[2];
  int channel = tensors->properties.validShape.dimensionSize[3];

  float *data = reinterpret_cast<float *>(tensors->sysMem[0].virAddr);
  unet_restuls.seg.resize(height * width);
  unet_restuls.width = width;
  unet_restuls.height = height;
  unet_restuls.num_classes = num_classes_;

  // argmax, operate in NHWC format
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      float top_score = -1000000.0f;
      int top_index = 0;
      float *c_data = data + (width * h + w) * channel;
      for (int c = 0; c < channel; c++) {
        if (c_data[c] > top_score) {
          top_score = c_data[c];
          top_index = c;
        }
      }
      unet_restuls.seg[h * width + w] = top_index;
    }
  }
  return 0;
}

int PostProcessScale(hbDNNTensor *tensors, bpu_image_info_t &image_info, Segmentation &unet_restuls) {

  // get shape
  int height = tensors->properties.validShape.dimensionSize[1];
  int width = tensors->properties.validShape.dimensionSize[2];
  int channel = tensors->properties.validShape.dimensionSize[3];
  float *scale = tensors->properties.scale.scaleData;
  int c_stride = tensors->properties.alignedShape.dimensionSize[3];

  int32_t *data = reinterpret_cast<int32_t *>(tensors->sysMem[0].virAddr);
  unet_restuls.seg.resize(height * width);
  unet_restuls.width = width;
  unet_restuls.height = height;
  unet_restuls.num_classes = num_classes_;

  // argmax, operate in NHWC format
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      float top_score = -1000000.0f;
      int top_index = 0;
      int32_t *c_data = data + (width * h + w) * c_stride;
      auto max_score_id = MaxScoreID(c_data, scale, channel);
      top_score = max_score_id.first;
      top_index = max_score_id.second;
      unet_restuls.seg[h * width + w] = top_index;
    }
  }
  return 0;
}

void UnetPostProcess(hbDNNTensor *tensors, bpu_image_info_t &image_info, Segmentation &unet_restuls) {

  auto quanti_type = tensors->properties.quantiType;
  if (quanti_type == hbDNNQuantiType::SCALE) {
    PostProcessScale(tensors, image_info, unet_restuls);
  } else if (quanti_type == hbDNNQuantiType::NONE) {
    PostProcessNone(tensors, image_info, unet_restuls);
  } else {
    printf("error quanti_type: %d\n", quanti_type);
  }

}