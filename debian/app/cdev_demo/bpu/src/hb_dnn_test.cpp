/***************************************************************************
 * @COPYRIGHT NOTICE
 * @Copyright 2023 Horizon Robotics, Inc.
 * @All rights reserved.
 * @Date: 2023-04-24 14:36:15
 * @LastEditTime: 2023-04-24 15:06:41
 ***************************************************************************/
#include <stdint.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <future>
#include <vector>
#include <string>
#include <opencv2/world.hpp>
#include <opencv2/opencv.hpp>  
#include <deque>
#include <thread>
#include <signal.h>
#include <argp.h>
#include "hb_dnn_test.hpp"


// global var area
std::deque<bpu_work> yolov5_work_deque;//work deque,contains processed tensor
std::deque<bpu_work> fcos_work_deque;//work deque,contains processed tensor
std::deque<bpu_work> yolov3_work_deque;//work deque,contains processed tensor
std::deque<bpu_work> ssd_work_deque;//work deque,contains processed tensor
std::deque<bpu_work> centernet_resnet50_work_deque;//work deque,contains processed tensor
std::deque<bpu_work> centernet_resnet101_work_deque;//work deque,contains processed tensor
std::deque<bpu_work> classification_work_deque;//work deque,contains processed tensor
std::deque<bpu_work> unet_work_deque;//work deque,contains processed tensor

static std::mutex yolo_mtx;
static std::atomic_bool yolo_finish;
static std::condition_variable yolo_cv;

static std::mutex fcos_mtx;
static std::atomic_bool fcos_finish;
static std::condition_variable fcos_cv;

static std::mutex ssd_mtx;
static std::atomic_bool ssd_finish;
static std::condition_variable ssd_cv;

static std::mutex centernet_resnet50_mtx;
static std::atomic_bool centernet_resnet50_finish;
static std::condition_variable centernet_resnet50_cv;

static std::mutex centernet_resnet101_mtx;
static std::atomic_bool centernet_resnet101_finish;
static std::condition_variable centernet_resnet101_cv;

static std::mutex classification_mtx;
static std::atomic_bool classification_finish;
static std::condition_variable classification_cv;

static std::mutex unet_mtx;
static std::atomic_bool unet_finish;
static std::condition_variable unet_cv;

static std::atomic<bool> is_stop;//runing flag

static int disp_w = 0, disp_h = 0;//display resolution
static int video_w = 0, video_h = 0;//only used on fcos,input video resolution
static std::string stream_file;//only used on fcos,input video file path
static bool debug = false;

void yolov5_do_post(void *display);
void yolov5_feed_bpu(void *camera, bpu_module *bpu_obj, std::shared_ptr<char> &buffer_672p);
void fcos_do_post(void *display);
void fcos_feed_bpu(void *vps, bpu_module *bpu_handle);
void yolov3_do_post(void *display);
void yolov3_feed_bpu(void *camera, bpu_module *bpu_obj, std::shared_ptr<char> &buffer_416p);
void ssd_do_post(void *display);
void ssd_feed_bpu(void *camera, bpu_module *bpu_obj, std::shared_ptr<char> &buffer_300p);
void centernet_resnet50_do_post(void *display);
void centernet_resnet50_feed_bpu(void *camera, bpu_module *bpu_obj, std::shared_ptr<char> &buffer_512p);
void centernet_resnet101_do_post(void *display);
void centernet_resnet101_feed_bpu(void *camera, bpu_module *bpu_obj, std::shared_ptr<char> &buffer_512p);
void classification_do_post(void *display);
void classification_feed_bpu(void *camera, bpu_module *bpu_obj, std::shared_ptr<char> &buffer_224p);
void unet_do_post(void *display);
void unet_feed_bpu(void *camera, bpu_module *bpu_obj, std::shared_ptr<char> &buffer_1024p);


static error_t parse_opt(int key, char *arg, struct argp_state *state)//args parse handle
{
    struct arguments *args = static_cast<arguments *>(state->input);
    switch (key)
    {
    case 'm':
        args->type = atoi(arg);
        break;
    case 'f':
        args->modle_file = arg;
        break;
    case 'i':
        args->video_path = arg;
        break;
    case 'h':
        args->height = atoi(arg);
        break;
    case 'w':
        args->width = atoi(arg);
        break;
    case 'd':
        args->debug = true;
        break;
    case ARGP_KEY_END:
    {
        if (state->argc < 5)//minimal arg num is 5:./sample --file=PATH --type=TYPE
        {
            argp_state_help(state, stdout, ARGP_HELP_STD_HELP);
        }
        if (args->type == 1 && (args->video_path.empty() || args->height == 0 || args->width == 0))
        {
            argp_state_help(state, stdout, ARGP_HELP_STD_HELP);
        }
    }
    break;
    default:
        return ARGP_ERR_UNKNOWN;
    }
    return 0;
}
static struct argp argp = {options, parse_opt, 0, doc};
void signal_handler_func(int signum)
{
    printf("\nrecv:%d,Stoping...\n", signum);
    is_stop = true;
}
int main(int argc, char *argv[])
{
    signal(SIGINT, signal_handler_func);
    struct arguments args{};
    // memset(&args, 0, sizeof(args));
    argp_parse(&argp, argc, argv, ARGP_IN_ORDER, 0, &args);
    std::string model_file = args.modle_file;
    int post_mode = args.type;
    stream_file = args.video_path;
    video_w = args.width;
    video_h = args.height;
    debug = args.debug;
    sp_get_display_resolution(&disp_w, &disp_h);//get display resolution 
    if (post_mode == 0)//yolov5 pipeline
    {
        std::shared_ptr<char> buffer_672p(new char[FRAME_BUFFER_SIZE(672, 672)]);//create buffer for saving resized frame

        bpu_module *bpu_obj = sp_init_bpu_module(model_file.c_str());
        int width[2] = {672, disp_w};//create 2 chn,672*672 for bpu input tensors,another for diplay.
        int height[2] = {672, disp_h};

        auto camera = sp_init_vio_module();
        auto display = sp_init_display_module();
        int ret = sp_open_camera(camera, 0, -1, 2, &(width[0]), &(height[0]));//open camera
        sleep(1);//for isp to stabilize
        ret = sp_start_display(display, 1, disp_w, disp_h);//display on 1 chn,this will not destroy the desktop chn
        sp_module_bind(camera, SP_MTYPE_VIO, display, SP_MTYPE_DISPLAY);//bind first
        ret = sp_start_display(display, 3, disp_w, disp_h); //after bind 1 chn to camera,open 3 chn to draw rectangle
        if (ret)
        {
            printf("display error!");
            return -1;
        }

        std::thread t1(yolov5_feed_bpu, std::ref(camera), std::ref(bpu_obj), std::ref(buffer_672p));//start pre processing thread
        std::thread t2(yolov5_do_post, std::ref(display));//start post processing thread

        t1.join();
        t2.join();
        sp_module_unbind(camera, SP_MTYPE_VIO, display, SP_MTYPE_DISPLAY);
        sp_stop_display(display);
        sp_release_display_module(display);
        sp_vio_close(camera);
        sp_release_vio_module(camera);
        printf("stop bpu!\n");
        sp_release_bpu_module(bpu_obj);
    }
    else if (post_mode == 1)
    {
        int ret = 0;
        int width[] = {512, disp_w};//open 2 chn,512x512 for bpu tensor input,another for display
        int height[] = {512, disp_h};
        // vio module init
        auto vps = sp_init_vio_module();
        // display module init
        auto display = sp_init_display_module();
        // bpu modue init,using args as filename
        auto bpu_handle = sp_init_bpu_module(model_file.c_str());
        // start display module,display on 1 chn,this will not destroy the desktop chn
        ret = sp_start_display(display, 1, disp_w, disp_h);
        printf("dispaly init ret = %d\n", ret);
        //NOTE!!!!!!!!!!
        //IF GET ERROR LIKE BAD ATTR,PLEASE CHECK YOUR INPUT RESOLUTION AND OUTPUT RESOLUTION!!!!! 
        ret = sp_open_vps(vps, 0, 2, SP_VPS_SCALE, video_w, video_h, width, height, NULL, NULL, NULL, NULL, NULL);
        printf("vps open ret = %d\n", ret);
        ret = sp_module_bind(vps, SP_MTYPE_VIO, display, SP_MTYPE_DISPLAY);//bind first
        printf("module bind vps & display ret = %d\n", ret);
        ret = sp_start_display(display, 3, disp_w, disp_h); // after binding 1 chn to camera,open 3 chn to draw rectangle
        printf("display start ret = %d\n", ret);

        std::thread t1(fcos_feed_bpu, std::ref(vps), std::ref(bpu_handle));//fcos pre processing thread start 
        std::thread t2(fcos_do_post, std::ref(display));//fcos post processing thread start

        t1.join();
        t2.join();

        // unbind
        sp_module_unbind(vps, SP_MTYPE_VIO, display, SP_MTYPE_DISPLAY);
        // module stop
        sp_stop_display(display);
        sp_release_display_module(display);
        sp_vio_close(vps);
        // module release
        sp_release_vio_module(vps);
    }
    else if (post_mode == 2)//yolov3 pipeline
    {
        std::shared_ptr<char> buffer_416p(new char[FRAME_BUFFER_SIZE(416, 416)]);//create buffer for saving resized frame

        bpu_module *bpu_obj = sp_init_bpu_module(model_file.c_str());
        int width[2] = {416, disp_w};//create 2 chn,416*416 for bpu input tensors,another for diplay.
        int height[2] = {416, disp_h};

        auto camera = sp_init_vio_module();
        auto display = sp_init_display_module();
        int ret = sp_open_camera(camera, 0, -1, 2, &(width[0]), &(height[0]));//open camera
        sleep(1);//for isp to stabilize
        ret = sp_start_display(display, 1, disp_w, disp_h);//display on 1 chn,this will not destroy the desktop chn
        sp_module_bind(camera, SP_MTYPE_VIO, display, SP_MTYPE_DISPLAY);//bind first
        ret = sp_start_display(display, 3, disp_w, disp_h); //after bind 1 chn to camera,open 3 chn to draw rectangle
        if (ret)
        {
            printf("display error!");
            return -1;
        }

        std::thread t1(yolov3_feed_bpu, std::ref(camera), std::ref(bpu_obj), std::ref(buffer_416p));//start pre processing thread
        std::thread t2(yolov3_do_post, std::ref(display));//start post processing thread

        t1.join();
        t2.join();
        sp_module_unbind(camera, SP_MTYPE_VIO, display, SP_MTYPE_DISPLAY);
        sp_stop_display(display);
        sp_release_display_module(display);
        sp_vio_close(camera);
        sp_release_vio_module(camera);
        printf("stop bpu!\n");
        sp_release_bpu_module(bpu_obj);
    }
    else if (post_mode == 4)//yolov5x pipeline
    {
        std::shared_ptr<char> buffer_672p(new char[FRAME_BUFFER_SIZE(672, 672)]);//create buffer for saving resized frame

        bpu_module *bpu_obj = sp_init_bpu_module(model_file.c_str());
        int width[2] = {672, disp_w};//create 2 chn,672*672 for bpu input tensors,another for diplay.
        int height[2] = {672, disp_h};

        auto camera = sp_init_vio_module();
        auto display = sp_init_display_module();
        int ret = sp_open_camera(camera, 0, -1, 2, &(width[0]), &(height[0]));//open camera
        sleep(1);//for isp to stabilize
        ret = sp_start_display(display, 1, disp_w, disp_h);//display on 1 chn,this will not destroy the desktop chn
        sp_module_bind(camera, SP_MTYPE_VIO, display, SP_MTYPE_DISPLAY);//bind first
        ret = sp_start_display(display, 3, disp_w, disp_h); //after bind 1 chn to camera,open 3 chn to draw rectangle
        if (ret)
        {
            printf("display error!");
            return -1;
        }

        std::thread t1(yolov5_feed_bpu, std::ref(camera), std::ref(bpu_obj), std::ref(buffer_672p));//start pre processing thread
        std::thread t2(yolov5_do_post, std::ref(display));//start post processing thread

        t1.join();
        t2.join();
        sp_module_unbind(camera, SP_MTYPE_VIO, display, SP_MTYPE_DISPLAY);
        sp_stop_display(display);
        sp_release_display_module(display);
        sp_vio_close(camera);
        sp_release_vio_module(camera);
        printf("stop bpu!\n");
        sp_release_bpu_module(bpu_obj);
    }
    else if (post_mode == 5) // ssd_mobilenetv1 pipeline
    {
        std::shared_ptr<char> buffer_300p(new char[FRAME_BUFFER_SIZE(300, 300)]);//create buffer for saving resized frame

        bpu_module *bpu_obj = sp_init_bpu_module(model_file.c_str());
        int width[2] = {300, disp_w};//create 2 chn,300*300 for bpu input tensors,another for diplay.
        int height[2] = {300, disp_h};

        auto camera = sp_init_vio_module();
        auto display = sp_init_display_module();
        int ret = sp_open_camera(camera, 0, -1, 2, &(width[0]), &(height[0]));//open camera
        sleep(1);//for isp to stabilize
        ret = sp_start_display(display, 1, disp_w, disp_h);//display on 1 chn,this will not destroy the desktop chn
        sp_module_bind(camera, SP_MTYPE_VIO, display, SP_MTYPE_DISPLAY);//bind first
        ret = sp_start_display(display, 3, disp_w, disp_h); //after bind 1 chn to camera,open 3 chn to draw rectangle
        if (ret)
        {
            printf("display error!");
            return -1;
        }

        std::thread t1(ssd_feed_bpu, std::ref(camera), std::ref(bpu_obj), std::ref(buffer_300p));//start pre processing thread
        std::thread t2(ssd_do_post, std::ref(display));//start post processing thread

        t1.join();
        t2.join();
        sp_module_unbind(camera, SP_MTYPE_VIO, display, SP_MTYPE_DISPLAY);
        sp_stop_display(display);
        sp_release_display_module(display);
        sp_vio_close(camera);
        sp_release_vio_module(camera);
        printf("stop bpu!\n");
        sp_release_bpu_module(bpu_obj);
    }
    else if (post_mode == 6) // X3 centernet resnet50 pipeline
    {
        std::shared_ptr<char> buffer_512p(new char[FRAME_BUFFER_SIZE(512, 512)]);//create buffer for saving resized frame

        bpu_module *bpu_obj = sp_init_bpu_module(model_file.c_str());
        int width[2] = {512, disp_w};//create 2 chn,512*512 for bpu input tensors,another for diplay.
        int height[2] = {512, disp_h};

        auto camera = sp_init_vio_module();
        auto display = sp_init_display_module();
        int ret = sp_open_camera(camera, 0, -1, 2, &(width[0]), &(height[0]));//open camera
        sleep(1);//for isp to stabilize
        ret = sp_start_display(display, 1, disp_w, disp_h);//display on 1 chn,this will not destroy the desktop chn
        sp_module_bind(camera, SP_MTYPE_VIO, display, SP_MTYPE_DISPLAY);//bind first
        ret = sp_start_display(display, 3, disp_w, disp_h); //after bind 1 chn to camera,open 3 chn to draw rectangle
        if (ret)
        {
            printf("display error!");
            return -1;
        }

        std::thread t1(centernet_resnet50_feed_bpu, std::ref(camera), std::ref(bpu_obj), std::ref(buffer_512p));//start pre processing thread
        std::thread t2(centernet_resnet50_do_post, std::ref(display));//start post processing thread

        t1.join();
        t2.join();
        sp_module_unbind(camera, SP_MTYPE_VIO, display, SP_MTYPE_DISPLAY);
        sp_stop_display(display);
        sp_release_display_module(display);
        sp_vio_close(camera);
        sp_release_vio_module(camera);
        printf("stop bpu!\n");
        sp_release_bpu_module(bpu_obj);
    }
    else if (post_mode == 7) // RDK Ultra centernet resnet101 pipeline
    {
        std::shared_ptr<char> buffer_512p(new char[FRAME_BUFFER_SIZE(512, 512)]);//create buffer for saving resized frame

        bpu_module *bpu_obj = sp_init_bpu_module(model_file.c_str());
        int width[2] = {512, disp_w};//create 2 chn,512*512 for bpu input tensors,another for diplay.
        int height[2] = {512, disp_h};

        auto camera = sp_init_vio_module();
        auto display = sp_init_display_module();
        int ret = sp_open_camera(camera, 0, -1, 2, &(width[0]), &(height[0]));//open camera
        sleep(1);//for isp to stabilize
        ret = sp_start_display(display, 1, disp_w, disp_h);//display on 1 chn,this will not destroy the desktop chn
        sp_module_bind(camera, SP_MTYPE_VIO, display, SP_MTYPE_DISPLAY);//bind first
        ret = sp_start_display(display, 3, disp_w, disp_h); //after bind 1 chn to camera,open 3 chn to draw rectangle
        if (ret)
        {
            printf("display error!");
            return -1;
        }

        std::thread t1(centernet_resnet101_feed_bpu, std::ref(camera), std::ref(bpu_obj), std::ref(buffer_512p));//start pre processing thread
        std::thread t2(centernet_resnet101_do_post, std::ref(display));//start post processing thread

        t1.join();
        t2.join();
        sp_module_unbind(camera, SP_MTYPE_VIO, display, SP_MTYPE_DISPLAY);
        sp_stop_display(display);
        sp_release_display_module(display);
        sp_vio_close(camera);
        sp_release_vio_module(camera);
        printf("stop bpu!\n");
        sp_release_bpu_module(bpu_obj);
    }
    else if (post_mode == 8) // Classification mobilenetv1 pipeline
    {
        std::shared_ptr<char> buffer_224p(new char[FRAME_BUFFER_SIZE(300, 300)]);//create buffer for saving resized frame

        bpu_module *bpu_obj = sp_init_bpu_module(model_file.c_str());
        int width[2] = {300, disp_w};//create 2 chn,224*224 for bpu input tensors,another for diplay.
        int height[2] = {300, disp_h};

        auto camera = sp_init_vio_module();
        auto display = sp_init_display_module();
        int ret = sp_open_camera(camera, 0, -1, 2, &(width[0]), &(height[0]));//open camera
        sleep(1);//for isp to stabilize
        ret = sp_start_display(display, 1, disp_w, disp_h);//display on 1 chn,this will not destroy the desktop chn
        sp_module_bind(camera, SP_MTYPE_VIO, display, SP_MTYPE_DISPLAY);//bind first
        ret = sp_start_display(display, 3, disp_w, disp_h); //after bind 1 chn to camera,open 3 chn to draw rectangle
        if (ret)
        {
            printf("display error!");
            return -1;
        }

        std::thread t1(classification_feed_bpu, std::ref(camera), std::ref(bpu_obj), std::ref(buffer_224p));//start pre processing thread
        std::thread t2(classification_do_post, std::ref(display));//start post processing thread

        t1.join();
        t2.join();
        sp_module_unbind(camera, SP_MTYPE_VIO, display, SP_MTYPE_DISPLAY);
        sp_stop_display(display);
        sp_release_display_module(display);
        sp_vio_close(camera);
        sp_release_vio_module(camera);
        printf("stop bpu!\n");
        sp_release_bpu_module(bpu_obj);
    }
    else if (post_mode == 9) // Segmentation unet pipeline
    {
        std::shared_ptr<char> buffer_1024p(new char[FRAME_BUFFER_SIZE(512, 512)]);//create buffer for saving resized frame

        bpu_module *bpu_obj = sp_init_bpu_module(model_file.c_str());
        int width[2] = {512, disp_w};//create 2 chn,512*512 for bpu input tensors,another for diplay.
        int height[2] = {512, disp_h};

        auto camera = sp_init_vio_module();
        auto display = sp_init_display_module();
        int ret = sp_open_camera(camera, 0, -1, 2, &(width[0]), &(height[0]));//open camera
        sleep(1);//for isp to stabilize
        ret = sp_start_display(display, 1, disp_w, disp_h);//display on 1 chn,this will not destroy the desktop chn
        sp_module_bind(camera, SP_MTYPE_VIO, display, SP_MTYPE_DISPLAY);//bind first
        ret = sp_start_display(display, 3, disp_w, disp_h); //after bind 1 chn to camera,open 3 chn to draw rectangle
        if (ret)
        {
            printf("display error!");
            return -1;
        }

        std::thread t1(unet_feed_bpu, std::ref(camera), std::ref(bpu_obj), std::ref(buffer_1024p));//start pre processing thread
        std::thread t2(unet_do_post, std::ref(display));//start post processing thread

        t1.join();
        t2.join();
        sp_module_unbind(camera, SP_MTYPE_VIO, display, SP_MTYPE_DISPLAY);
        sp_stop_display(display);
        sp_release_display_module(display);
        sp_vio_close(camera);
        sp_release_vio_module(camera);
        printf("stop bpu!\n");
        sp_release_bpu_module(bpu_obj);
    }
    return 0;
}
void fcos_feed_bpu(void *vps, bpu_module *bpu_handle)
{
    int ret = 0;
    auto decoder = sp_init_decoder_module();

    // vio module setting
    //  decoder -> vps -> display
    std::shared_ptr<char> buffer_512p(new char[FRAME_BUFFER_SIZE(512, 512)]);
    //start decode
    ret = sp_start_decode(decoder, stream_file.c_str(), 0, SP_ENCODER_H264, video_w, video_h);
    printf("decode start ret = %d\n", ret);
    ret = sp_module_bind(decoder, SP_MTYPE_DECODER, vps, SP_MTYPE_VIO);//bind decode to vps,this binding is for scale
    printf("module bind decoder & vps ret = %d\n", ret);
    //using 5 group tensors as ring buffer.
    hbDNNTensor output_tensors[5][15];//The number of tensors in each group can be known by calling sp_init_bpu_tensors
    //fcos has 15 output tensor
    int cur_ouput_buf_idx = 0;
    for (int i = 0; i < 5; i++)
    {
        ret = sp_init_bpu_tensors(bpu_handle, output_tensors[i]);//init tensor.
        if (ret)
        {
            printf("prepare model output tensor failed\n");
            is_stop = true;
        }
    }

    while (!is_stop)
    {
        bpu_work fcos_work;
        ret = sp_vio_get_frame(vps, buffer_512p.get(), 512, 512, 500);//get frame from vps,512*512 resolution is for bpu input
        if (ret != 0)//if get frame fail,restart decode pipeline
        {
            sp_module_unbind(decoder, SP_MTYPE_DECODER, vps, SP_MTYPE_VIO);
            sp_stop_decode(decoder);
            sp_release_decoder_module(decoder);
            decoder = sp_init_decoder_module();
            ret = sp_start_decode(decoder, stream_file.c_str(), 0, SP_ENCODER_H264, video_w, video_h);
            sp_module_bind(decoder, SP_MTYPE_DECODER, vps, SP_MTYPE_VIO);
            if (ret)
            {
                printf("[Error] sp_start_decode failed\n");
                break;
            }
            continue;
        }
        bpu_handle->output_tensor = &output_tensors[cur_ouput_buf_idx][0];//get an tensor buffer from ring buffer
        fcos_work.start_time = std::chrono::high_resolution_clock::now();//get timestamp
        sp_bpu_start_predict(bpu_handle, buffer_512p.get());//start bpu predict
        fcos_work.payload = bpu_handle->output_tensor;//bpu processed tensor
        fcos_work_deque.push_back(fcos_work);//push back work struct to deque
        cur_ouput_buf_idx++;
        cur_ouput_buf_idx %= 5;
    }
    fcos_finish = true;
    for (size_t i = 0; i < 5; i++)
    {
        sp_deinit_bpu_tensor(output_tensors[i], 15);//release tensor buffer
    }
    sp_module_unbind(decoder, SP_MTYPE_DECODER, vps, SP_MTYPE_VIO);
    sp_stop_decode(decoder);
    sp_release_decoder_module(decoder);
    sp_release_bpu_module(bpu_handle);
}
void fcos_do_post(void *display)
{
    bpu_image_info_t image_info;//using for mapping the tensor result coordinates back to the original image
    image_info.m_model_h = 512;
    image_info.m_model_w = 512;//input tensor size
    image_info.m_ori_height = disp_h;
    image_info.m_ori_width = disp_w;//origin size
    std::vector<Detection> results;//store processed result
    do
    {
        while (!fcos_work_deque.empty() && !is_stop)
        {
            results.clear();
            auto work = fcos_work_deque.front();
            auto output = work.payload;
            auto stime = work.start_time;
            fcos_post_process(output, &image_info, results);//do post process
            fcos_work_deque.pop_front();
            sp_display_draw_rect(display, 0, 0, 0, 0, 3, 1, 0x00000000, 2);//flush display
            if (debug) {
                // fps
                auto delta_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - stime).count();
                double fps = 1000.0 / delta_time;
                printf("fps:%lf,processing time:%ld\n", fps, delta_time);
            }
            for (size_t i = 0; i < results.size(); i++)
            {
                sp_display_draw_rect(display, results[i].bbox.xmin, results[i].bbox.ymin,
                                     results[i].bbox.xmax, results[i].bbox.ymax, 3, 0, 0xFFFF0000, 2);//draw rectangle
            }
        }

    } while (!fcos_finish);
}

void yolov5_feed_bpu(void *camera, bpu_module *bpu_handle, std::shared_ptr<char> &buffer_672p)
{
    //using 5 group tensors as ring buffer
    //The number of tensors in each group can be known by calling sp_init_bpu_tensors
    hbDNNTensor output_tensors[5][3];
    int cur_ouput_buf_idx = 0;
    int ret = -1;
    for (int i = 0; i < 5; i++)
    {
        //init tensor
        ret = sp_init_bpu_tensors(bpu_handle, output_tensors[i]);
        if (ret)
        {
            printf("prepare model output tensor failed\n");
            is_stop = true;
        }
    }
    while (!is_stop)
    {
        bpu_work yolov5_work;
        sp_vio_get_frame(camera, buffer_672p.get(), 672, 672, 2000);//get frame,672*672 is for bpu input tensors
        bpu_handle->output_tensor = &output_tensors[cur_ouput_buf_idx][0];
        yolov5_work.start_time = std::chrono::high_resolution_clock::now();//get timestamp
        sp_bpu_start_predict(bpu_handle, buffer_672p.get());//star bpu predict
        yolov5_work.payload = bpu_handle->output_tensor;
        yolov5_work_deque.push_back(yolov5_work);//push back work strcut to deque
        cur_ouput_buf_idx++;
        cur_ouput_buf_idx %= 5;
    }
    yolo_finish = true;
    printf("%s,finish!\n", __func__);
    {
        std::unique_lock<std::mutex> lock(yolo_mtx);
        for (size_t i = 0; i < 5; i++)
        {
            sp_deinit_bpu_tensor(output_tensors[i], 3);//relaese tensor
        }
    }
}

void yolov5_do_post(void *display)
{
    bpu_image_info_t image_info;//using for mapping the tensor result coordinates back to the original image
    image_info.m_model_h = 672;
    image_info.m_model_w = 672;//input tensor size
    image_info.m_ori_height = disp_h;
    image_info.m_ori_width = disp_w;//origin size
    std::vector<std::shared_ptr<YoloV5Result>> results;//store process result
    std::vector<YoloV5Result> parse_results;
    do
    {
        while (!yolov5_work_deque.empty() && !is_stop)
        {

            results.clear();
            parse_results.clear();
            auto work = yolov5_work_deque.front();
            auto output = work.payload;
            auto stime = work.start_time;

            for (size_t j = 0; j < 3; j++)
            {
                {
                    std::unique_lock<std::mutex> lock(yolo_mtx);
                    if (!is_stop)
                        ParseTensor(std::make_shared<hbDNNTensor>(output[j]), static_cast<int>(j), parse_results, image_info);//do post process part 1
                }
            }
            yolo5_nms(parse_results, nms_threshold_, nms_top_k_, results, false);//do post process part 2
            if (debug) {
                // fps
                auto delta_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - stime).count();
                double fps = 1000.0 / delta_time;
                printf("%s fps:%lf,processing time :%ld\n", __func__, fps, delta_time);
            }
            yolov5_work_deque.pop_front();
            sp_display_draw_rect(display, 0, 0, 0, 0, 3, 1, 0x00000000, 2);//flush display
            for (size_t i = 0; i < results.size(); i++)
            {
                sp_display_draw_rect(display, results[i]->xmin, results[i]->ymin,
                                     results[i]->xmax, results[i]->ymax, 3, 0, 0xFFFF0000, 2);//draw rectangle
                sp_display_draw_string(display, results[i]->xmin, results[i]->ymin,
                                     const_cast<char*>(results[i]->class_name.c_str()), 3, 0, 0xFFFF0000, 2); //draw string
            }
        }

    } while (!yolo_finish);
    printf("%s,finish!\n", __func__);
}

void yolov3_feed_bpu(void *camera, bpu_module *bpu_handle, std::shared_ptr<char> &buffer_416p)
{
    //using 5 group tensors as ring buffer
    //The number of tensors in each group can be known by calling sp_init_bpu_tensors
    hbDNNTensor output_tensors[5][yolov3_output_nums_];
    printf("%d", yolov3_output_nums_);
    int cur_ouput_buf_idx = 0;
    int ret = -1;
    for (int i = 0; i < 5; i++)
    {
        //init tensor
        ret = sp_init_bpu_tensors(bpu_handle, output_tensors[i]);
        if (ret)
        {
            printf("prepare model output tensor failed\n");
            is_stop = true;
        }
    }
    while (!is_stop)
    {
        bpu_work yolov3_work;
        sp_vio_get_frame(camera, buffer_416p.get(), 416, 416, 2000);//get frame,416*416 is for bpu input tensors
        bpu_handle->output_tensor = &output_tensors[cur_ouput_buf_idx][0];
        yolov3_work.start_time = std::chrono::high_resolution_clock::now();//get timestamp
        sp_bpu_start_predict(bpu_handle, buffer_416p.get());//star bpu predict
        yolov3_work.payload = bpu_handle->output_tensor;
        yolov3_work_deque.push_back(yolov3_work);//push back work strcut to deque
        cur_ouput_buf_idx++;
        cur_ouput_buf_idx %= 5;
    }
    yolo_finish = true;
    printf("%s,finish!\n", __func__);
    {
        std::unique_lock<std::mutex> lock(yolo_mtx);
        for (size_t i = 0; i < 5; i++)
        {
            sp_deinit_bpu_tensor(output_tensors[i], yolov3_output_nums_);//relaese tensor
        }
    }
}

void yolov3_do_post(void *display)
{
    bpu_image_info_t image_info;//using for mapping the tensor result coordinates back to the original image
    image_info.m_model_h = 416;
    image_info.m_model_w = 416;//input tensor size
    image_info.m_ori_height = disp_h;
    image_info.m_ori_width = disp_w;//origin size
    std::vector<std::shared_ptr<YoloV3Result>> results;//store process result
    std::vector<YoloV3Result> parse_results;
    do
    {
        while (!yolov3_work_deque.empty() && !is_stop)
        {

            results.clear();
            parse_results.clear();
            auto work = yolov3_work_deque.front();
            auto output = work.payload;
            auto stime = work.start_time;

            for (size_t j = 0; j < yolov3_output_nums_; j++)
            {
                {
                    std::unique_lock<std::mutex> lock(yolo_mtx);
                    if (!is_stop)
                        yolov3_ParseTensor(std::make_shared<hbDNNTensor>(output[j]), static_cast<int>(j), parse_results, image_info);//do post process part 1
                }
            }
            yolo3_nms(parse_results, yolov3_nms_threshold_, yolov3_nms_top_k_, results, false);//do post process part 2
            if (debug) {
                // fps
                auto delta_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - stime).count();
                double fps = 1000.0 / delta_time;
                printf("%s fps:%lf,processing time :%ld\n", __func__, fps, delta_time);
            }
            yolov3_work_deque.pop_front();
            sp_display_draw_rect(display, 0, 0, 0, 0, 3, 1, 0x00000000, 2);//flush display
            for (size_t i = 0; i < results.size(); i++)
            {
                sp_display_draw_rect(display, results[i]->xmin, results[i]->ymin,
                                     results[i]->xmax, results[i]->ymax, 3, 0, 0xFFFF0000, 2); //draw rectangle
                sp_display_draw_string(display, results[i]->xmin, results[i]->ymin,
                                     const_cast<char*>(results[i]->class_name.c_str()), 3, 0, 0xFFFF0000, 2); //draw string
            }
        }

    } while (!yolo_finish);
    printf("%s,finish!\n", __func__);
}

void ssd_feed_bpu(void *camera, bpu_module *bpu_handle, std::shared_ptr<char> &buffer_300p)
{
    //using 5 group tensors as ring buffer
    //The number of tensors in each group can be known by calling sp_init_bpu_tensors
    hbDNNTensor output_tensors[5][12];
    int cur_ouput_buf_idx = 0;
    int ret = -1;
    for (int i = 0; i < 5; i++)
    {
        //init tensor
        ret = sp_init_bpu_tensors(bpu_handle, output_tensors[i]);
        if (ret)
        {
            printf("prepare model output tensor failed\n");
            is_stop = true;
        }
    }
    while (!is_stop)
    {
        bpu_work ssd_work;
        sp_vio_get_frame(camera, buffer_300p.get(), 300, 300, 2000);//get frame,300*300 is for bpu input tensors
        bpu_handle->output_tensor = &output_tensors[cur_ouput_buf_idx][0];
        ssd_work.start_time = std::chrono::high_resolution_clock::now();//get timestamp
        sp_bpu_start_predict(bpu_handle, buffer_300p.get());//star bpu predict
        ssd_work.payload = bpu_handle->output_tensor;
        ssd_work_deque.push_back(ssd_work);//push back work strcut to deque
        cur_ouput_buf_idx++;
        cur_ouput_buf_idx %= 5;
    }
    ssd_finish = true;
    printf("%s,finish!\n", __func__);
    {
        std::unique_lock<std::mutex> lock(ssd_mtx);
        for (size_t i = 0; i < 5; i++)
        {
            sp_deinit_bpu_tensor(output_tensors[i], 12);//relaese tensor
        }
    }
}

void ssd_do_post(void *display)
{
    bpu_image_info_t image_info;//using for mapping the tensor result coordinates back to the original image
    image_info.m_model_h = 300;
    image_info.m_model_w = 300;//input tensor size
    image_info.m_ori_height = disp_h;
    image_info.m_ori_width = disp_w;//origin size
    std::vector<Detection> results;
  
    do
    {
        while (!ssd_work_deque.empty() && !is_stop)
        {

            results.clear();
            auto work = ssd_work_deque.front();
            auto output = work.payload;
            auto stime = work.start_time;

            std::unique_lock<std::mutex> lock(ssd_mtx);
            if (!is_stop){
                SSDPostProcess(output, image_info, results);
            }

            if (debug) {
                // fps
                auto delta_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - stime).count();
                double fps = 1000.0 / delta_time;
                printf("%s fps:%lf,processing time :%ld\n", __func__, fps, delta_time);
            }
            ssd_work_deque.pop_front();
            sp_display_draw_rect(display, 0, 0, 0, 0, 3, 1, 0x00000000, 2);//flush display
            for (size_t i = 0; i < results.size(); i++)
            {
                sp_display_draw_rect(display, results[i].bbox.xmin, results[i].bbox.ymin,
                        results[i].bbox.xmax, results[i].bbox.ymax, 3, 0, 0xFFFF0000, 2);//draw rectangle
                sp_display_draw_string(display, results[i].bbox.xmin, results[i].bbox.ymin,
                                     const_cast<char*>(results[i].class_name), 3, 0, 0xFFFF0000, 2); //draw string
             
            }
        }

    } while (!ssd_finish);
    printf("%s,finish!\n", __func__);
}

void centernet_resnet50_feed_bpu(void *camera, bpu_module *bpu_handle, std::shared_ptr<char> &buffer_512p)
{
    //using 5 group tensors as ring buffer
    //The number of tensors in each group can be known by calling sp_init_bpu_tensors
    hbDNNTensor output_tensors[5][3];
    int cur_ouput_buf_idx = 0;
    int ret = -1;
    for (int i = 0; i < 5; i++)
    {
        //init tensor
        ret = sp_init_bpu_tensors(bpu_handle, output_tensors[i]);
        if (ret)
        {
            printf("prepare model output tensor failed\n");
            is_stop = true;
        }
    }
    while (!is_stop)
    {
        bpu_work centernet_resnet50_work;
        sp_vio_get_frame(camera, buffer_512p.get(), 512, 512, 2000);//get frame,512*512 is for bpu input tensors
        bpu_handle->output_tensor = &output_tensors[cur_ouput_buf_idx][0];
        centernet_resnet50_work.start_time = std::chrono::high_resolution_clock::now();//get timestamp
        sp_bpu_start_predict(bpu_handle, buffer_512p.get());//star bpu predict
        centernet_resnet50_work.payload = bpu_handle->output_tensor;
        centernet_resnet50_work_deque.push_back(centernet_resnet50_work);//push back work strcut to deque
        cur_ouput_buf_idx++;
        cur_ouput_buf_idx %= 5;
    }
    centernet_resnet50_finish = true;
    printf("%s,finish!\n", __func__);
    {
        std::unique_lock<std::mutex> lock(centernet_resnet50_mtx);
        for (size_t i = 0; i < 5; i++)
        {
            sp_deinit_bpu_tensor(output_tensors[i], 3);//relaese tensor
        }
    }
}

void centernet_resnet50_do_post(void *display)
{
    bpu_image_info_t image_info;//using for mapping the tensor result coordinates back to the original image
    image_info.m_model_h = 512;
    image_info.m_model_w = 512;//input tensor size
    image_info.m_ori_height = disp_h;
    image_info.m_ori_width = disp_w;//origin size
    std::vector<Detection> results;
  
    do
    {
        while (!centernet_resnet50_work_deque.empty() && !is_stop)
        {

            results.clear();
            auto work = centernet_resnet50_work_deque.front();
            auto output = work.payload;
            auto stime = work.start_time;

            std::unique_lock<std::mutex> lock(centernet_resnet50_mtx);
            if (!is_stop){
                CenternetPostProcess(output, image_info, results, 0);
            }

            if (debug) {
                // fps
                auto delta_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - stime).count();
                double fps = 1000.0 / delta_time;
                printf("%s fps:%lf,processing time :%ld\n", __func__, fps, delta_time);
            }
            centernet_resnet50_work_deque.pop_front();
            sp_display_draw_rect(display, 0, 0, 0, 0, 3, 1, 0x00000000, 2);//flush display
            for (size_t i = 0; i < results.size(); i++)
            {
                sp_display_draw_rect(display, results[i].bbox.xmin, results[i].bbox.ymin,
                        results[i].bbox.xmax, results[i].bbox.ymax, 3, 0, 0xFFFF0000, 2);//draw rectangle
                sp_display_draw_string(display, results[i].bbox.xmin, results[i].bbox.ymin,
                                     const_cast<char*>(results[i].class_name), 3, 0, 0xFFFF0000, 2); //draw string
             
            }
        }

    } while (!centernet_resnet50_finish);
    printf("%s,finish!\n", __func__);
}

void centernet_resnet101_feed_bpu(void *camera, bpu_module *bpu_handle, std::shared_ptr<char> &buffer_512p)
{
    //using 5 group tensors as ring buffer
    //The number of tensors in each group can be known by calling sp_init_bpu_tensors
    hbDNNTensor output_tensors[5][3];
    int cur_ouput_buf_idx = 0;
    int ret = -1;
    for (int i = 0; i < 5; i++)
    {
        //init tensor
        ret = sp_init_bpu_tensors(bpu_handle, output_tensors[i]);
        if (ret)
        {
            printf("prepare model output tensor failed\n");
            is_stop = true;
        }
    }
    while (!is_stop)
    {
        bpu_work centernet_resnet101_work;
        sp_vio_get_frame(camera, buffer_512p.get(), 512, 512, 2000);//get frame,512*512 is for bpu input tensors
        bpu_handle->output_tensor = &output_tensors[cur_ouput_buf_idx][0];
        centernet_resnet101_work.start_time = std::chrono::high_resolution_clock::now();//get timestamp
        sp_bpu_start_predict(bpu_handle, buffer_512p.get());//star bpu predict
        centernet_resnet101_work.payload = bpu_handle->output_tensor;
        centernet_resnet101_work_deque.push_back(centernet_resnet101_work);//push back work strcut to deque
        cur_ouput_buf_idx++;
        cur_ouput_buf_idx %= 5;
    }
    centernet_resnet101_finish = true;
    printf("%s,finish!\n", __func__);
    {
        std::unique_lock<std::mutex> lock(centernet_resnet101_mtx);
        for (size_t i = 0; i < 5; i++)
        {
            sp_deinit_bpu_tensor(output_tensors[i], 3);//relaese tensor
        }
    }
}

void centernet_resnet101_do_post(void *display)
{
    bpu_image_info_t image_info;//using for mapping the tensor result coordinates back to the original image
    image_info.m_model_h = 512;
    image_info.m_model_w = 512;//input tensor size
    image_info.m_ori_height = disp_h;
    image_info.m_ori_width = disp_w;//origin size
    std::vector<Detection> results;
  
    do
    {
        while (!centernet_resnet101_work_deque.empty() && !is_stop)
        {

            results.clear();
            auto work = centernet_resnet101_work_deque.front();
            auto output = work.payload;
            auto stime = work.start_time;

            std::unique_lock<std::mutex> lock(centernet_resnet101_mtx);
            if (!is_stop){
                CenternetMaxPoolSigmoidPostProcess(output, image_info, results, 0);
            }

            if (debug) {
                // fps
                auto delta_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - stime).count();
                double fps = 1000.0 / delta_time;
                printf("%s fps:%lf,processing time :%ld\n", __func__, fps, delta_time);
            }
            centernet_resnet101_work_deque.pop_front();
            sp_display_draw_rect(display, 0, 0, 0, 0, 3, 1, 0x00000000, 2);//flush display
            for (size_t i = 0; i < results.size(); i++)
            {
                sp_display_draw_rect(display, results[i].bbox.xmin, results[i].bbox.ymin,
                        results[i].bbox.xmax, results[i].bbox.ymax, 3, 0, 0xFFFF0000, 2);//draw rectangle
                sp_display_draw_string(display, results[i].bbox.xmin, results[i].bbox.ymin,
                                     const_cast<char*>(results[i].class_name), 3, 0, 0xFFFF0000, 2); //draw string
             
            }
        }

    } while (!centernet_resnet101_finish);
    printf("%s,finish!\n", __func__);
}

void classification_feed_bpu(void *camera, bpu_module *bpu_handle, std::shared_ptr<char> &buffer_224p)
{
    //using 5 group tensors as ring buffer
    //The number of tensors in each group can be known by calling sp_init_bpu_tensors
    hbDNNTensor output_tensors[5][1];
    int cur_ouput_buf_idx = 0;
    int ret = -1;
    for (int i = 0; i < 5; i++)
    {
        //init tensor
        ret = sp_init_bpu_tensors(bpu_handle, output_tensors[i]);
        if (ret)
        {
            printf("prepare model output tensor failed\n");
            is_stop = true;
        }
    }
    while (!is_stop)
    {
        bpu_work classification_work;
        sp_vio_get_frame(camera, buffer_224p.get(), 300, 300, 2000);//get frame,512*512 is for bpu input tensors
        bpu_handle->output_tensor = &output_tensors[cur_ouput_buf_idx][0];
        classification_work.start_time = std::chrono::high_resolution_clock::now();//get timestamp
        
        // mobilenetv1 输入224x224， 将300 缩放到224 送给BPU做推理
        auto srcData = buffer_224p.get(); // 获取NV12图像数据的指针  
        int srcWidth = 300;  
        int srcHeight = 300;  

        int dstWidth = 224;
        int dstHeight = 224;
        unsigned char* dstData = new unsigned char[dstWidth * dstHeight * 3 / 2];

        // NV12数据resize函数  
        cv::Mat srcY(srcHeight + (srcHeight / 2), srcWidth, CV_8UC1, srcData); // 创建Y分量Mat对象  
        cv::Mat srcUV(srcHeight / 2, srcWidth / 2, CV_16UC2, srcData + srcHeight * srcWidth); // 创建UV分量Mat对象  
    
        cv::Mat dstY(dstHeight + (dstHeight / 2), dstWidth, CV_8UC1); // 创建目标Y分量Mat对象  
        cv::Mat dstUV(dstHeight / 2, dstWidth / 2, CV_16UC2); // 创建目标UV分量Mat对象  
    
        // 对Y分量进行resize  
        cv::resize(srcY(cv::Rect(0, 0, srcWidth, srcHeight)), dstY(cv::Rect(0, 0, dstWidth, dstHeight)), cv::Size(dstWidth, dstHeight));  
    
        // 对UV分量进行resize  
        cv::resize(srcUV(cv::Rect(0, 0, srcWidth / 2, srcHeight / 2)), dstUV(cv::Rect(0, 0, dstWidth / 2, dstHeight / 2)), cv::Size(dstWidth / 2, dstHeight / 2));  
    
        // 将resize后的数据复制到目标NV12数据中  
        memcpy(dstData, dstY.data, dstY.total() * sizeof(unsigned char));  
        memcpy(dstData + dstHeight * dstWidth, dstUV.data, dstUV.total() * sizeof(unsigned short));  
    
        // sp_bpu_start_predict(bpu_handle, buffer_224p.get());//star bpu predict
        sp_bpu_start_predict(bpu_handle, (char *)dstData);//star bpu predict
        delete[] dstData; 

        classification_work.payload = bpu_handle->output_tensor;
        classification_work_deque.push_back(classification_work);//push back work strcut to deque
        cur_ouput_buf_idx++;
        cur_ouput_buf_idx %= 5;
    }
    classification_finish = true;
    printf("%s,finish!\n", __func__);
    {
        std::unique_lock<std::mutex> lock(classification_mtx);
        for (size_t i = 0; i < 5; i++)
        {
            sp_deinit_bpu_tensor(output_tensors[i], 1);//relaese tensor
        }
    }
}

void classification_do_post(void *display)
{
    bpu_image_info_t image_info;//using for mapping the tensor result coordinates back to the original image
    image_info.m_model_h = 224;
    image_info.m_model_w = 224;//input tensor size
    image_info.m_ori_height = disp_h;
    image_info.m_ori_width = disp_w;//origin size
    std::vector<Classification> results;
  
    do
    {
        while (!classification_work_deque.empty() && !is_stop)
        {

            results.clear();
            auto work = classification_work_deque.front();
            auto output = work.payload;
            auto stime = work.start_time;

            std::unique_lock<std::mutex> lock(classification_mtx);
            if (!is_stop){
                ClassificationPostProcess(output, image_info, results);
            }

            if (debug) {
                // fps
                auto delta_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - stime).count();
                double fps = 1000.0 / delta_time;
                printf("%s fps:%lf,processing time :%ld\n", __func__, fps, delta_time);
            }
            classification_work_deque.pop_front();
            // sp_display_draw_rect(display, 0, 0, 0, 0, 3, 1, 0x00000000, 2);//flush display

            printf("classification_result: \n");
            for (size_t i = 0; i < results.size(); i++) {
                std::cout << results[i] << std::endl; 
            }
        }

    } while (!classification_finish);
    printf("%s,finish!\n", __func__);
}

void unet_feed_bpu(void *camera, bpu_module *bpu_handle, std::shared_ptr<char> &buffer_1024p)
{
    //using 5 group tensors as ring buffer
    //The number of tensors in each group can be known by calling sp_init_bpu_tensors
    hbDNNTensor output_tensors[5][1];
    int cur_ouput_buf_idx = 0;
    int ret = -1;
    for (int i = 0; i < 5; i++)
    {
        //init tensor
        ret = sp_init_bpu_tensors(bpu_handle, output_tensors[i]);
        if (ret)
        {
            printf("prepare model output tensor failed\n");
            is_stop = true;
        }
    }
    while (!is_stop)
    {
        bpu_work unet_work;
        sp_vio_get_frame(camera, buffer_1024p.get(), 512, 512, 2000);//get frame,512*512 is for bpu input tensors
        bpu_handle->output_tensor = &output_tensors[cur_ouput_buf_idx][0];
        unet_work.start_time = std::chrono::high_resolution_clock::now();//get timestamp
        
        // mobilenetv1 输入224x224， 将300 缩放到224 送给BPU做推理
        auto srcData = buffer_1024p.get(); // 获取NV12图像数据的指针  
        int srcWidth = 512;  
        int srcHeight = 512;  

        int dstWidth = 1024;
        int dstHeight = 2048;
        unsigned char* dstData = new unsigned char[dstWidth * dstHeight * 3 / 2];
        
        // NV12数据resize函数  
        cv::Mat srcY(srcHeight + (srcHeight / 2), srcWidth, CV_8UC1, srcData); // 创建Y分量Mat对象  
        cv::Mat srcUV(srcHeight / 2, srcWidth / 2, CV_16UC2, srcData + srcHeight * srcWidth); // 创建UV分量Mat对象  
    
        cv::Mat dstY(dstHeight + (dstHeight / 2), dstWidth, CV_8UC1); // 创建目标Y分量Mat对象  
        cv::Mat dstUV(dstHeight / 2, dstWidth / 2, CV_16UC2); // 创建目标UV分量Mat对象  
    
        // 对Y分量进行resize  
        cv::resize(srcY(cv::Rect(0, 0, srcWidth, srcHeight)), dstY(cv::Rect(0, 0, dstWidth, dstHeight)), cv::Size(dstWidth, dstHeight));  
    
        // 对UV分量进行resize  
        cv::resize(srcUV(cv::Rect(0, 0, srcWidth / 2, srcHeight / 2)), dstUV(cv::Rect(0, 0, dstWidth / 2, dstHeight / 2)), cv::Size(dstWidth / 2, dstHeight / 2));  

        // 将resize后的数据复制到目标NV12数据中  
        memcpy(dstData, dstY.data, dstY.total() * sizeof(unsigned char));  
        memcpy(dstData + dstHeight * dstWidth, dstUV.data, dstUV.total() * sizeof(unsigned short));  

        // sp_bpu_start_predict(bpu_handle, buffer_224p.get());//star bpu predict
        sp_bpu_start_predict(bpu_handle, (char *)dstData);//star bpu predict
        delete[] dstData; 

        unet_work.payload = bpu_handle->output_tensor;
        unet_work_deque.push_back(unet_work);//push back work strcut to deque
        cur_ouput_buf_idx++;
        cur_ouput_buf_idx %= 5;
    }
    unet_finish = true;
    printf("%s,finish!\n", __func__);
    {
        std::unique_lock<std::mutex> lock(unet_mtx);
        for (size_t i = 0; i < 5; i++)
        {
            sp_deinit_bpu_tensor(output_tensors[i], 1);//relaese tensor
        }
    }
}

void unet_do_post(void *display)
{
    bpu_image_info_t image_info;//using for mapping the tensor result coordinates back to the original image
    image_info.m_model_h = 1024;
    image_info.m_model_w = 2048;//input tensor size
    image_info.m_ori_height = disp_h;
    image_info.m_ori_width = disp_w;//origin size
    Segmentation results;

    do
    {
        while (!unet_work_deque.empty() && !is_stop)
        {
            memset(&results, 0, sizeof(Segmentation));
            auto work = unet_work_deque.front();
            auto output = work.payload;
            auto stime = work.start_time;

            std::unique_lock<std::mutex> lock(unet_mtx);
            if (!is_stop){
                UnetPostProcess(output, image_info, results);
            }

            if (debug) {
                // fps
                auto delta_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - stime).count();
                double fps = 1000.0 / delta_time;
                printf("%s fps:%lf,processing time :%ld\n", __func__, fps, delta_time);
            }
            unet_work_deque.pop_front();
            // sp_display_draw_rect(display, 0, 0, 0, 0, 3, 1, 0x00000000, 2);//flush display

            printf("unet_result: results.seg.size():%ld, num_classes:%d, width:%d, height:%d\n", results.seg.size(), results.num_classes, results.width, results.height);
            // for (size_t i = 0; i < results.seg.size(); i++) {
            //     std::cout << results.seg[i] << std::endl; 
            // }
        }

    } while (!unet_finish);
    printf("%s,finish!\n", __func__);
}