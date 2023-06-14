#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include <sp_vio.h>
#include <sp_sys.h>
#include <sp_display.h>
#include <stdlib.h>
#include <string.h>
#include <argp.h>
#include <unistd.h>
#include <sys/stat.h>

#define clear() printf("\033[H\033[J")

double get_stride(int width, int bit)
{
    double temp = (width * bit / 8.0 / 16.0); // Determining whether alignment is possible
    double fractpart;
    double intpart;
    double result;
    // get fractpart
    fractpart = modf(temp, &intpart);

    if (fractpart > 0)
    {
        // Rounding
        return ceil(temp) * 16;
    }

    return temp * 16;
}

struct arguments
{
    int width;
    int height;
    int bit;
    int count;
};
static char doc[] = "capture sample -- An example of capture yuv/raw";
static struct argp_option options[] = {
    {"width", 'w', "width", 0, "sensor output width"},
    {"height", 'h', "height", 0, "sensor output height"},
    {"bit", 'b', "bit", 0, "the depth of raw,mostly is 10,imx477 is 12"},
    {"count", 'c', "number", 0, "capture number"},
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
    case 'b':
        args->bit = atoi(arg);
        break;
    case 'c':
        args->count = atoi(arg);
        break;
    case ARGP_KEY_END:
    {
        if (state->argc < 8)
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
    int widths[] = {args.width};
    int heights[] = {args.height};
    double stride = get_stride(args.width, args.bit);
    // printf("Stride:%.2f\n",stride);
    int raw_size = (stride * args.height); // raw_size = stride * height, stride = width * bit / 8 (align with 16)
    int yuv_size = FRAME_BUFFER_SIZE(args.width, args.height);
    sp_sensors_parameters parms;
    parms.fps = -1;
    parms.raw_height = args.height;
    parms.raw_width = args.width;
    char ch = 0;
    int is_enter = 0;
    int yuv_count = 0;
    int raw_count = 0;
    char yuv_filename[50];
    char raw_filename[50];
    // init camera
    void *camera = sp_init_vio_module();
    // open camera
    // ret = sp_open_camera(camera, 0, -1, 1, &widths[0], &heights[0]);
    ret = sp_open_camera_v2(camera, 0, -1, 1, &parms, widths, heights);
    sleep(2); // wait for isp stability
    if (ret != 0)
    {
        printf("[Error] sp_open_camera failed!\n");
        goto error1;
    }
    // malloc buffer
    char *raw_data = malloc(raw_size * sizeof(char));
    char *yuv_data = malloc(yuv_size * sizeof(char));

    do
    {
        printf("capture time :%d\n", yuv_count);

        sp_vio_get_yuv(camera, yuv_data, args.width, args.height, 2000);
        sprintf(yuv_filename, "yuv_%d.yuv", yuv_count++);
        FILE *yuv_file = fopen(yuv_filename, "wb");
        fwrite(yuv_data, sizeof(char), yuv_size, yuv_file);
        fflush(yuv_file);

        sp_vio_get_raw(camera, raw_data, args.width, args.height, 2000);
        sprintf(raw_filename, "raw_%d.raw", raw_count++);
        FILE *raw_file = fopen(raw_filename, "wb");
        fwrite(raw_data, sizeof(char), raw_size, raw_file);
        fflush(raw_file);
    } while (--args.count > 0);

    free(raw_data);
    free(yuv_data);
error1:
    sp_vio_close(camera);
    sp_release_vio_module(camera);
    return 0;
}
