/***************************************************************************
 * @COPYRIGHT NOTICE
 * @Copyright 2023 Horizon Robotics, Inc.
 * @All rights reserved.
 * @Date: 2023-04-11 15:57:16
 * @LastEditTime: 2023-06-07 16:59:26
 ***************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <argp.h>

#include "sp_codec.h"
#include "sp_vio.h"
#include "sp_sys.h"
#include "sp_display.h"

struct arguments
{
    int width;
    int height;
};
static char doc[] = "vio(sensors) display sample -- An example of display sensor's image";
static struct argp_option options[] = {
    {"width", 'w', "width", 0, "sensor output width"},
    {"height", 'h', "height", 0, "sensor output height"},
    {0}};
static error_t parse_opt(int key, char *arg, struct argp_state *state)
{
    struct arguments *args = state->input;
    switch (key)
    {
    case 'w':
        args->width = atoi(arg);
        break;
    case 'h':
        args->height = atoi(arg);
        break;
    case ARGP_KEY_END:
    {
        if (state->argc < 4)
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

int main(int argc, char **argv)
{
    char ch;
    int is_enter;
    struct arguments args;
    memset(&args, 0, sizeof(args));
    argp_parse(&argp, argc, argv, 0, 0, &args);
    // 获取显示器支持的分辨率
    int disp_w = 0, disp_h = 0;
    sp_get_display_resolution(&disp_w, &disp_h);
    int widths[] = {disp_w};
    int heights[] = {disp_h};
    printf("disp_w=%d, disp_h=%d\n", disp_w, disp_h);
    sp_sensors_parameters parms;
    parms.fps = -1;
    parms.raw_height = args.height;
    parms.raw_width = args.width;
    void *vio_object = sp_init_vio_module();
    // int ret = sp_open_camera(vio_object, 0, -1, 1, widths, heights);
    int ret = sp_open_camera_v2(vio_object, 0, -1, 1, &parms, widths, heights);
    if (ret != 0)
    {
        printf("[Error] sp_open_camera failed!\n");
        return -1;
    }

    printf("sp_open_camera success!\n");

    // display
    void *display_obj = sp_init_display_module();
    // 使用通道1，这样不会破坏图形化系统，在程序退出后还能恢复桌面
    ret = sp_start_display(display_obj, 1, disp_w, disp_h);
    if (ret)
    {
        printf("[Error] sp_start_display failed, ret = %d\n", ret);
        goto error;
    }
    ret = sp_module_bind(vio_object, SP_MTYPE_VIO, display_obj, SP_MTYPE_DISPLAY);
    if (ret)
    {
        printf("[Error] sp_module_bind failed, ret = %d\n", ret);
        goto error0;
    }

    // 输出 q 退出
    while (1)
    {
        printf("\nPress 'q' to Exit !\n");
        while ('\n' == (ch = (char)getchar()))
            ;
        is_enter = getchar();
        if ('q' == ch && is_enter == 10)
        {
            break;
        }
        else
        {
            while ('\n' != (ch = (char)getchar()))
                ;
        }
    }
    printf("Exit!\n");
error0:
    sp_module_unbind(vio_object, SP_MTYPE_VIO, display_obj, SP_MTYPE_DISPLAY);
error:
    /*stop module*/
    sp_stop_display(display_obj);
    sp_vio_close(vio_object);
    /*release object*/
    sp_release_display_module(display_obj);
    sp_release_vio_module(vio_object);

    return 0;
}
