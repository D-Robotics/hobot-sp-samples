/*
 * @Author: jiale01.luo
 * @Date: 2022-10-24 11:19:28
 * @Last Modified by: jiale01.luo
 * @Last Modified time: 2022-10-26 14:08:30
 */
#ifndef hb_dnn_test
#define hb_dnn_test

#include "sp_bpu.h"
#include "sp_vio.h"
#include "yolov5_post_process.hpp"
#include "fcos_post_process.hpp"
#include "yolov3_post_process.hpp"
#include "ptq_ssd_post_process_method.hpp"
#include "ptq_centernet_post_process_method.hpp"
#include "ptq_centernet_maxpool_sigmoid_post_process_method.hpp"
#include "ptq_classification_post_process_method.hpp"
#include "ptq_unet_post_process_method.hpp"
#include "sp_display.h"
#include "sp_codec.h"
#include "sp_sys.h"
#include "vio/hb_common_vot.h"
#include <dnn/hb_dnn.h>
#include <dnn/hb_sys.h>
#include <chrono>
#include <time.h>

typedef struct 
{
    hbDNNTensor * payload;
    std::chrono::system_clock::time_point start_time;
}bpu_work;

static char doc[] = "bpu sample -- An C++ example of using bpu";
struct arguments
{
    int type;
    std::string modle_file;
    std::string video_path;
    int height;
    int width;
    bool debug;
};
static struct argp_option options[] = {
    {"mode", 'm', "type", 0, "0:yolov5;1:fcos"},
    {"file", 'f', "modle_file", 0, "path of model file"},
    {"input_video", 'i', "video path", 0, "path of video"},
    {"video_height", 'h', "height", 0, "height of video"},
    {"video_width", 'w', "width", 0, "width of video"},
    {"debug", 'd', 0, 0, "Print lots of debugging information."},
    {0}};
#endif
