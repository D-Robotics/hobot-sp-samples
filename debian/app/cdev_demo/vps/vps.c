#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <argp.h>
#include <sys/stat.h>
#include "sp_codec.h"
#include "sp_vio.h"
#include "sp_sys.h"
#include "sp_display.h"

int file2vps(char *input, char *output, int input_width, int input_height, int output_width, int out_height);
int decoder2vps(char *input, char *output, int input_width, int input_height, int output_width, int out_height, int skip);
struct arguments
{
    int mode;
    char *input_path;
    char *output_path;
    int input_width;
    int input_height;
    int output_height;
    int output_width;
    int skip;
};
static char doc[] = "vps sample -- An example of using a vps interface";
static struct argp_option options[] = {
    {"mode", 'm', "mode", 0, "input mode: 1:stream;2:file"},
    {"input", 'i', "path", 0, "input file path"},
    {"output", 'o', "path", 0, "output file path"},
    {"iwidth", 0x80, "width", 0, "input width"},
    {"iheight", 0x81, "height", 0, "input height"},
    {"owidth", 0x82, "width", 0, "output width"},
    {"oheight", 0x83, "height", 0, "output height"},
    {"skip", 0x84, "skip", 0, "skip frame"},
    {0}};
static error_t parse_opt(int key, char *arg, struct argp_state *state)
{
    struct arguments *args = state->input;
    switch (key)
    {
    case 'm':
        args->mode = atoi(arg);
        break;
    case 'i':
        args->input_path = arg;
        break;
    case 'o':
        args->output_path = arg;
        break;
    case 0x80:
        args->input_width = atoi(arg);
        break;
    case 0x81:
        args->input_height = atoi(arg);
        break;
    case 0x82:
        args->output_width = atoi(arg);
        break;
    case 0x83:
        args->output_height = atoi(arg);
        break;
    case 0x84:
        args->skip = atoi(arg);
        break;
    case ARGP_KEY_END:
    {
        if (state->argc < 15 || state->argc > 17)
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
    int ret = 0;
    struct arguments args;
    memset(&args, 0, sizeof(args));
    argp_parse(&argp, argc, argv, 0, 0, &args);
    if (args.mode == 1)
    {
        ret = decoder2vps(args.input_path, args.output_path, args.input_width, args.input_height, args.output_width, args.output_height, args.skip);
        if (ret != 0)
        {
            printf("decoder2vps error!\n");
            return -1;
        }
    }
    else
    {
        ret = file2vps(args.input_path, args.output_path, args.input_width, args.input_height, args.output_width, args.output_height);
        if (ret != 0)
        {
            printf("file2vps error!\n");
            return -1;
        }
    }

    return 0;
}

int file2vps(char *input, char *output, int input_width, int input_height, int output_width, int out_height)
{
    // init vps
    void *vps = sp_init_vio_module();
    // init heap memory
    int input_size = FRAME_BUFFER_SIZE(input_width, input_height);
    int output_size = FRAME_BUFFER_SIZE(output_width, out_height);
    char *frame_buffer_input = malloc(input_size);
    char *frame_buffer_output = malloc(output_size);
    // set vps to scale only mode
    int ret = sp_open_vps(vps, 0, 1, SP_VPS_SCALE, input_width, input_height, &output_width, &out_height, NULL, NULL, NULL, NULL, NULL);
    if (ret != 0)
    {
        printf("[Error] sp_open_vps failed!\n");
        goto exit;
    }
    // read image from file
    struct stat buffer;
    if (stat(input, &buffer))
    {
        perror("file2vps");
        goto exit;
    }
    FILE *input_image = fopen(input, "rb");
    fread(frame_buffer_input, sizeof(char), input_size, input_image);
    // send image to vps
    ret = sp_vio_set_frame(vps, frame_buffer_input, input_size);
    if (ret != 0)
    {
        printf("[Error] sp_vio_set_frame from vps failed!\n");
        goto exit;
    }
    // Get the processed image
    ret = sp_vio_get_frame(vps, frame_buffer_output, output_width, out_height, 2000);
    if (ret != 0)
    {
        printf("[Error] sp_vio_get_frame from vps failed!\n");
        goto exit;
    }
    FILE *output_image = fopen(output, "wb");
    fwrite(frame_buffer_output, sizeof(char), output_size, output_image);
exit:
    // Heap memory release
    free(frame_buffer_output);
    free(frame_buffer_input);
    // Module stop
    sp_vio_close(vps);
    // Object release
    sp_release_vio_module(vps);
    // File handle close
    fclose(input_image);
    fclose(output_image);
    return 0;
}
int decoder2vps(char *input, char *output, int input_width, int input_height, int output_width, int out_height, int skip)
{
    int input_size = FRAME_BUFFER_SIZE(input_width, input_height);
    int output_size = FRAME_BUFFER_SIZE(output_width, out_height);
    char *frame_buffer_input = malloc(input_size);
    char *frame_buffer_output = malloc(output_size);
    //initialize module
    void *decoder = sp_init_decoder_module();
    void *vps = sp_init_vio_module();
    struct stat buffer;
    if (stat(input, &buffer))
    {
        perror("decoder2vps");
        goto exit;
    }
    //start decode
    int ret = sp_start_decode(decoder, input, 0, SP_ENCODER_H264, input_width, input_height);
    if (ret != 0)
    {
        printf("[Error] sp_start_decode failed!\n");
        goto exit;
    }
    // skip the first few frames
    // of course you can choose not to skip
    skip = skip == 0 ? 1 : skip;
    for (size_t i = 0; i < skip; i++)
    {
        sp_decoder_get_image(decoder, frame_buffer_input);
    }
    // Start configuring VPS
    // Set vps mode to scale only
    ret = sp_open_vps(vps, 0, 1, SP_VPS_SCALE, input_width, input_height, &output_width, &out_height, NULL, NULL, NULL, NULL, NULL);
    if (ret != 0)
    {
        printf("[Error] sp_open_vps failed!\n");
        goto exit;
    }
    FILE *file_input = fopen("origin.yuv", "wb");
    fwrite(frame_buffer_input, sizeof(char), input_size, file_input);
    // Send picture to vps
    ret = sp_vio_set_frame(vps, frame_buffer_input, input_size);
    if (ret != 0)
    {
        printf("[Error] sp_vio_set_frame from vps failed!\n");
        goto exit;
    }
    // Get the processed image
    sp_vio_get_frame(vps, frame_buffer_output, output_width, out_height, 2000);
    // write image
    FILE *image = fopen(output, "wb");
    fwrite(frame_buffer_output, sizeof(char), output_size, image);
exit:
    // Heap memory release
    free(frame_buffer_output);
    free(frame_buffer_input);
    // Module stop
    sp_stop_decode(decoder);
    sp_vio_close(vps);
    // Object release
    sp_release_decoder_module(decoder);
    sp_release_vio_module(vps);
    // File handle close
    fclose(image);
    fclose(file_input);
    return 0;
}