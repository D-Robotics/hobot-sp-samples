#ifndef ptq_classification_post_process_method
#define ptq_classification_post_process_method

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
#include "ptq_classification_post_process_method.hpp"

typedef struct Classification {
  int id;
  float score;
  const char *class_name;

  Classification() : class_name(0) {}

  Classification(int id, float score, const char *class_name)
      : id(id), score(score), class_name(class_name) {}

  friend std::ostream &operator<<(std::ostream &os, const Classification &cls) {
    const auto precision = os.precision();
    const auto flags = os.flags();
    os << "{"
       << R"("prob")"
       << ":" << std::fixed << std::setprecision(5) << cls.score << ","
       << R"("label")"
       << ":" << cls.id << ","
       << R"("class_name")"
       << ":"
       << "\"" << cls.class_name << "\""
       << "}";
    os.flags(flags);
    os.precision(precision);
    return os;
  }

  ~Classification() {}
} Classification;


extern void ClassificationPostProcess(hbDNNTensor *tensors, bpu_image_info_t &image_info, std::vector<Classification> &classification_restuls);


#endif  // ptq_classification_post_process_method
