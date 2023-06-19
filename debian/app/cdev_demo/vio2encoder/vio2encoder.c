#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <argp.h>
#include <signal.h>
#include <stdatomic.h>
#include "sp_codec.h"
#include "sp_vio.h"
#include "sp_sys.h"
#include "sp_display.h"

#define STREAM_FRAME_SIZE 2097152
static char doc[] = "vio2encode sample -- An example of using the camera to record and encode";
atomic_bool is_stop;
struct arguments
{
    char *output_path;
    int output_height;
    int output_width;
    int input_height;
    int input_width;
};
static struct argp_option options[] = {
    {"output", 'o', "path", 0, "output file path"},
    {"owidth", 'w', "width", 0, "width of output video"},
    {"oheight", 'h', "height", 0, "height of output video"},
    {"iwidth", 0x81, "width", 0, "sensor output width"},
    {"iheight", 0x82, "height", 0, "sensor output height"},
    {0}};
static error_t parse_opt(int key, char *arg, struct argp_state *state)
{
    struct arguments *args = state->input;
    switch (key)
    {
    case 'o':
        args->output_path = arg;
        break;
    case 'w':
        args->output_width = atoi(arg);
        break;
    case 'h':
        args->output_height = atoi(arg);
        break;
    case 0x81:
        args->input_width = atoi(arg);
        break;
    case 0x82:
        args->input_height = atoi(arg);
        break;
    case ARGP_KEY_END:
    {
        if (state->argc != 11)
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
    // singal handle,stop program while press ctrl + c
    signal(SIGINT, signal_handler_func);
    int ret = 0, i = 0;
    int stream_frame_size = 0;
    // parse args
    struct arguments args;
    memset(&args, 0, sizeof(args));
    argp_parse(&argp, argc, argv, 0, 0, &args);
    int width = args.output_width;
    int height = args.output_height;
    // int widths[1] = {width};
    // int heights[1] = {height};

    sp_sensors_parameters parms;
    parms.fps = -1;
    parms.raw_height = args.input_height;
    parms.raw_width = args.input_width;

    // init module
    void *vio_object = sp_init_vio_module();
    void *encoder = sp_init_encoder_module();
    char *stream_buffer = malloc(sizeof(char) * STREAM_FRAME_SIZE);
    /** open camera **/
    // ret = sp_open_camera(vio_object, 0, -1, 1, &width, &height);
    ret = sp_open_camera_v2(vio_object, 0, -1, 1, &parms, &width, &height);
    if (ret != 0)
    {
        printf("[Error] sp_open_camera failed!\n");
        goto exit;
    }
    printf("sp_open_camera success!\n");
    /** init encoder **/
    ret = sp_start_encode(encoder, 0, SP_ENCODER_H264, width, height, 8000);
    if (ret != 0)
    {
        printf("[Error] sp_start_encode failed!\n");
        goto exit;
    }
    printf("sp_start_encode success!\n");
    // bind camera(vio) and decoder
    ret = sp_module_bind(vio_object, SP_MTYPE_VIO, encoder, SP_MTYPE_ENCODER);
    if (ret != 0)
    {
        printf("sp_module_bind(vio -> encoder) failed\n");
        goto exit1;
    }
    printf("sp_module_bind(vio -> encoder) success!\n");

    FILE *stream = fopen(args.output_path, "wb+");
    while (!is_stop)
    {
        memset(stream_buffer, 0, STREAM_FRAME_SIZE);
        stream_frame_size = sp_encoder_get_stream(encoder, stream_buffer); // get stream from encoder
        // printf("size:%d\n", stream_frame_size);
        if (stream_frame_size == -1)
        {
            printf("encoder_get_image error! ret = %d,i = %d\n", ret, i++);
            goto exit1;
        }
        fwrite(stream_buffer, sizeof(char), stream_frame_size, stream); // write stream to file
    }
exit1:
    sp_module_unbind(vio_object, SP_MTYPE_VIO, encoder, SP_MTYPE_ENCODER);
exit:
    /* file close*/
    fclose(stream);
    /*head memory release*/
    free(stream_buffer);
    /*stop module*/
    sp_stop_encode(encoder);
    sp_vio_close(vio_object);
    /*release object*/
    sp_release_encoder_module(encoder);
    sp_release_vio_module(vio_object);

    return 0;
}
