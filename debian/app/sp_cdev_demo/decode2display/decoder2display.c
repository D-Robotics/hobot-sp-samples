#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <argp.h>
#include <stdatomic.h>
#include <signal.h>
#include "sp_codec.h"
#include "sp_vio.h"
#include "sp_sys.h"
#include "sp_display.h"

static char doc[] = "decode2display sample -- An example of streaming video decoding to the display";
atomic_bool is_stop;
struct arguments
{
    char *input_path;
    int height;
    int width;
};
static struct argp_option options[] = {
    {"input", 'i', "path", 0, "input video path"},
    {"width", 'w', "width", 0, "width of input video"},
    {"height", 'h', "height", 0, "height of input video"},
    {0}};
static error_t parse_opt(int key, char *arg, struct argp_state *state)
{
    struct arguments *args = state->input;
    switch (key)
    {
    case 'i':
        args->input_path = arg;
        break;
    case 'w':
        args->width = atoi(arg);
        break;
    case 'h':
        args->height = atoi(arg);
        break;
    case ARGP_KEY_END:
    {
        if (state->argc != 7)
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
    is_stop = 1;
}
int main(int argc, char **argv)
{
    signal(SIGINT, signal_handler_func);
    struct arguments args;
    memset(&args, 0, sizeof(args));
    argp_parse(&argp, argc, argv, 0, 0, &args);
    int stream_width = args.width;
    int stream_height = args.height;
    char *stream_file = args.input_path;

    int ret = 0;
    void *vio_object;
    // 获取显示器支持的分辨率
    int disp_w = 0, disp_h = 0;
    sp_get_display_resolution(&disp_w, &disp_h);
    int widths[] = {disp_w};
    int heights[] = {disp_h};
    printf("disp_w=%d, disp_h=%d\n", disp_w, disp_h);

    int frame_buffer_size = FRAME_BUFFER_SIZE(stream_width, stream_height);
    char *frame_buffer = malloc(frame_buffer_size);
    void *decoder = sp_init_decoder_module();
    // 初始化解码通道，视频流的分辨率需要和显示器分辨率相同，否则会无法正常显示
    ret = sp_start_decode(decoder, stream_file, 0, SP_ENCODER_H264, stream_width, stream_height);
    if (ret)
    {
        printf("[Error] sp_start_decode failed\n");
        goto error;
    }
    printf("sp_start_decode success!\n");

    // display
    void *display_obj = sp_init_display_module();
    // 使用通道1，这样不会破坏图形化系统，在程序退出后还能恢复桌面
    ret = sp_start_display(display_obj, 1, disp_w, disp_h);
    if (ret)
    {
        printf("[Error] sp_start_display failed, ret = %d\n", ret);
        goto error;
    }
    printf("sp_start_display success!\n");

    // 如果视频分辨率和显示分辨率不同，需要创建一路 VPS 对图像进行缩放
    if (disp_w != stream_width || disp_h != stream_height)
    {
        vio_object = sp_init_vio_module();
        ret = sp_open_vps(vio_object, 0, 1, SP_VPS_SCALE, stream_width, stream_height,
                          widths, heights, NULL, NULL, NULL, NULL, NULL);
        if (ret != 0)
        {
            printf("[Error] sp_open_vps failed!\n");
            return -1;
        }

        printf("sp_open_vps success!\n");

        // 绑定 DECODERE 和 VIO
        ret = sp_module_bind(decoder, SP_MTYPE_DECODER, vio_object, SP_MTYPE_VIO);
        if (ret)
        {
            printf("[Error] sp_module_bind failed, ret = %d\n", ret);
            goto error;
        }

        // 绑定 VIO 和 display
        ret = sp_module_bind(vio_object, SP_MTYPE_VIO, display_obj, SP_MTYPE_DISPLAY);
        if (ret)
        {
            printf("[Error] sp_module_bind failed, ret = %d\n", ret);
            goto error;
        }
    }

    while (!is_stop)
    {
        memset(frame_buffer, 0, frame_buffer_size);
        ret = sp_decoder_get_image(decoder, frame_buffer);
        if (ret != 0)
        { // 解码结束，重新初始化解码通道
            if (disp_w != stream_width || disp_h != stream_height)
                sp_module_unbind(decoder, SP_MTYPE_DECODER, vio_object, SP_MTYPE_VIO);
            sp_stop_decode(decoder);
            sp_release_decoder_module(decoder);

            decoder = sp_init_decoder_module();
            ret = sp_start_decode(decoder, stream_file, 0, SP_ENCODER_H264, stream_width, stream_height);
            if (ret)
            {
                printf("[Error] sp_start_decode failed\n");
                goto error;
            }
            if (disp_w != stream_width || disp_h != stream_height)
                sp_module_bind(decoder, SP_MTYPE_DECODER, vio_object, SP_MTYPE_VIO);
            continue;
            ;
        }
        if (disp_w == stream_width || disp_h == stream_height)
        {
            sp_display_set_image(display_obj, frame_buffer, FRAME_BUFFER_SIZE(disp_w, disp_h), 1);
        }
    }

error:
    /*heap memory release*/
    free(frame_buffer);
    if (disp_w != stream_width || disp_h != stream_height)
    {
        sp_module_unbind(decoder, SP_MTYPE_DECODER, vio_object, SP_MTYPE_VIO);
        sp_module_unbind(vio_object, SP_MTYPE_VIO, display_obj, SP_MTYPE_DISPLAY);
        sp_vio_close(vio_object);
        sp_release_vio_module(vio_object);
    }

    /*stop module*/
    sp_stop_display(display_obj);
    sp_stop_decode(decoder);
    /*release object*/
    sp_release_display_module(display_obj);
    sp_release_decoder_module(decoder);
    return 0;
}