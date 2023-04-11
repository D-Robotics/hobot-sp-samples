#include <stdio.h>
#include <stdbool.h>
#include <libavformat/avformat.h>
#include <libavutil/timestamp.h>
#include <sp_codec.h>
#include <sp_display.h>
#include <sp_sys.h>
#include <sp_vio.h>
#include <vio/hb_common_vot.h>
#include <argp.h>
#include <stdatomic.h>
#include <signal.h>
//#define SP_DEBUG

static char doc[] = "decode2display sample -- An example of streaming video decoding to the display";
atomic_bool is_stop;
struct arguments
{
    char *rtsp_url;
    char *transfer_type;
};
static struct argp_option options[] = {
    {"input", 'i', "path", 0, "rtsp url"},
    {"type", 't', "type", 0, "tcp or udp"},
    {0}};
static error_t parse_opt(int key, char *arg, struct argp_state *state)
{
    struct arguments *args = state->input;
    switch (key)
    {
    case 'i':
        args->rtsp_url = arg;
        break;
    case 't':
        args->transfer_type = arg;
        break;
    case ARGP_KEY_END:
    {
        if (state->argc != 5)
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

void open_rtsp(const char *rtsp_url, const char *transfer_type)
{
    unsigned int i;
    int ret;
    int video_st_index = -1;
    int audio_st_index = -1;
    AVFormatContext *ifmt_ctx = NULL;
    AVPacket pkt;
    AVStream *st = NULL;
    char errbuf[64];
    AVDictionary *optionsDict = NULL;
    av_register_all();       // Register all codecs and formats so that they can be used.
    avformat_network_init(); // Initialization of network components
    av_init_packet(&pkt);    // initialize packet.
    pkt.data = NULL;
    pkt.size = 0;
    bool nRestart = false;
    AVStream *pVst;
    uint8_t *buffer_rgb = NULL;
    AVCodecContext *pVideoCodecCtx = NULL;
    AVFrame *pFrame = av_frame_alloc();
    AVFrame *pFrameRGB = av_frame_alloc();
    int got_picture;
    AVCodec *pVideoCodec = NULL;

    av_dict_set(&optionsDict, "stimeout", "3000000", 0);           // if don't setting this property，av_read_frame will run as block mode (ms)
    av_dict_set(&optionsDict, "bufsize", "1024000", 0);            // buffer size
    av_dict_set(&optionsDict, "rtsp_transport", transfer_type, 0); // transfer type,udp will faster but may lost some packet,tcp slower but stable
    if ((ret = avformat_open_input(&ifmt_ctx, rtsp_url, 0, &optionsDict)) < 0)
    { // Open the input file for reading.
        printf("Could not open input file '%s' (error '%s')\n", rtsp_url, av_make_error_string(errbuf, sizeof(errbuf), ret));
        goto EXIT;
    }

    printf("avformat_open_input ok!\n");

    if ((ret = avformat_find_stream_info(ifmt_ctx, NULL)) < 0)
    { // Get information on the input file (number of streams etc.).
        printf("Could not open find stream info (error '%s')\n", av_make_error_string(errbuf, sizeof(errbuf), ret));
        goto EXIT;
    }

    printf("avformat_find_stream_info ok!\n");

    for (i = 0; i < ifmt_ctx->nb_streams; i++)
    { // dump information
        av_dump_format(ifmt_ctx, i, rtsp_url, 0);
    }

    printf("av_dump_format ok!\n");

    for (i = 0; i < ifmt_ctx->nb_streams; i++)
    { // find video stream index
        st = ifmt_ctx->streams[i];
        switch (st->codec->codec_type)
        {
        case AVMEDIA_TYPE_AUDIO:
            audio_st_index = i;
            break;
        case AVMEDIA_TYPE_VIDEO:
            video_st_index = i;
            break;
        default:
            break;
        }
    }

    if (-1 == video_st_index)
    {
        printf("No H.264 video stream in the input file\n");
        goto EXIT;
    }

    if (!nRestart)
    { // getting stream msg
        pVst = ifmt_ctx->streams[video_st_index];
        pVideoCodecCtx = pVst->codec;
        pVideoCodec = avcodec_find_decoder(pVideoCodecCtx->codec_id);
        if (pVideoCodec == NULL)
            return;
        if (avcodec_open2(pVideoCodecCtx, pVideoCodec, NULL) < 0)
            return;
    }
    /*Begin decoder and display*/
    // getting stream height and width
    int rtsp_w = pVideoCodecCtx->width, rtsp_h = pVideoCodecCtx->height;
    int disp_w = 0, disp_h = 0;
    // init module
    void *decoder = sp_init_decoder_module();
    void *display = sp_init_display_module();
    void *vps = sp_init_vio_module();
    // get display resolution
    sp_get_display_resolution(&disp_w, &disp_h);
    printf("rtsp_w:%d,rtsp_h:%d\ndisplay_w:%d,dispaly_h:%d\n", pVideoCodecCtx->width, pVideoCodecCtx->height, disp_w, disp_h);
    // decode setting
    // Setting the stream_file to null means that the decoded source is manually called sp_decoder_set_image func
    ret = sp_start_decode(decoder, "", 0, SP_ENCODER_H264, rtsp_w, rtsp_h);
    // if rtsp resolution doesn't match display's,using vps to scale
    // refering our decoder2display sample
    if (ret != 0)
    {
        printf("decoder start error!\n");
        goto EXIT1;
    }
    ret = sp_start_display(display, 1, disp_w, disp_h);
    if (ret != 0)
    {
        printf("display start error!\n");
        goto EXIT1;
    }
    ret = sp_open_vps(vps, 0, 1, SP_VPS_SCALE, rtsp_w, rtsp_h,
                      &disp_w, &disp_h, NULL, NULL, NULL, NULL, NULL);
    if (ret != 0)
    {
        printf("[Error] sp_open_vps failed!\n");
        goto EXIT1;
    }

    printf("sp_open_vps success!\n");

    // bind decoder and vio(vps)
    ret = sp_module_bind(decoder, SP_MTYPE_DECODER, vps, SP_MTYPE_VIO);
    if (ret)
    {
        printf("[Error] sp_module_bind failed, ret = %d\n", ret);
        goto EXIT1;
    }

    // bind vio(vps) and display
    ret = sp_module_bind(vps, SP_MTYPE_VIO, display, SP_MTYPE_DISPLAY);
    if (ret)
    {
        printf("[Error] sp_module_bind failed, ret = %d\n", ret);
        goto EXIT1;
    }
    /*End*/
    while (!is_stop)
    {
        do
        {
#ifdef SP_DEBUG
            printf("start av_read_frame\n");
#endif
            ret = av_read_frame(ifmt_ctx, &pkt); // read frames
#ifdef SP_DEBUG
            printf("end av_read_frame\n");
#endif
            if (pkt.stream_index == video_st_index)
            {
#ifdef SP_DEBUG
                printf("video_st_index\n");
                printf("pkt.size=%d, pkt.pts=%lld, pkt.data=0x%x.\n", pkt.size, pkt.pts, (unsigned int)pkt.data);
#endif
                sp_decoder_set_image(decoder, pkt.data, 0, pkt.size, 0);//sending stream frame to decoder 
            }
        } while (ret == AVERROR(EAGAIN) && (!is_stop));

        if (ret < 0)
        {
            printf("Could not read frame ---(error '%s')\n", av_make_error_string(errbuf, sizeof(errbuf), ret));
            goto EXIT;
        }
        av_packet_unref(&pkt);
    }
    sp_module_unbind(vps, SP_MTYPE_VIO, display, SP_MTYPE_DISPLAY);
    sp_module_unbind(decoder, SP_MTYPE_DECODER, vps, SP_MTYPE_VIO);
EXIT1:
    if (display != NULL)
    {
        sp_stop_display(display);
        sp_release_display_module(display);
    }
    if (vps != NULL)
    {
        sp_vio_close(vps);
        sp_release_vio_module(vps);
    }
    if (decoder != NULL)
    {
        sp_stop_decode(decoder);
        sp_release_decoder_module(decoder);
    }
EXIT:
    if (NULL != ifmt_ctx)
    {
        avcodec_close(pVideoCodecCtx);
        avformat_close_input(&ifmt_ctx);
        av_free(pFrame);
        av_free(pFrameRGB);
        ifmt_ctx = NULL;
    }

    return;
}
void signal_handler_func(int signum)
{
    printf("\nrecv:%d,Stoping...\n", signum);
    is_stop = 1;
}
int main(int argc, char **argv)
{
    // singal handle,stop program while press ctrl + c
    signal(SIGINT, signal_handler_func);
    // start parse cmdline args...
    // 1.initialize an struct which contains property of rtsp_url and transfer_type
    struct arguments args;
    memset(&args, 0, sizeof(args));
    // 2.parse args
    argp_parse(&argp, argc, argv, 0, 0, &args);
    // 3.call open_rtsp
    open_rtsp(args.rtsp_url, args.transfer_type);
    return 0;
}
