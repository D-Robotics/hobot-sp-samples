#!/usr/bin/env python3

import sys
import os
import signal
import getopt
import numpy as np
import cv2
import colorsys
from time import time
from time import sleep
import threading
from queue import Queue
import multiprocessing
from threading import BoundedSemaphore

from hobot_dnn import pyeasy_dnn as dnn
from hobot_vio import libsrcampy as srcampy

import ctypes
import json 


output_tensors = None

fcos_postprocess_info = None
  

class hbSysMem_t(ctypes.Structure):
    _fields_ = [
        ("phyAddr",ctypes.c_double),
        ("virAddr",ctypes.c_void_p),
        ("memSize",ctypes.c_int)
    ]

class hbDNNQuantiShift_yt(ctypes.Structure):
    _fields_ = [
        ("shiftLen",ctypes.c_int),
        ("shiftData",ctypes.c_char_p)
    ]

class hbDNNQuantiScale_t(ctypes.Structure):
    _fields_ = [
        ("scaleLen",ctypes.c_int),
        ("scaleData",ctypes.POINTER(ctypes.c_float)),
        ("zeroPointLen",ctypes.c_int),
        ("zeroPointData",ctypes.c_char_p)
    ]    

class hbDNNTensorShape_t(ctypes.Structure):
    _fields_ = [
        ("dimensionSize",ctypes.c_int * 8),
        ("numDimensions",ctypes.c_int)
    ]

class hbDNNTensorProperties_t(ctypes.Structure):
    _fields_ = [
        ("validShape",hbDNNTensorShape_t),
        ("alignedShape",hbDNNTensorShape_t),
        ("tensorLayout",ctypes.c_int),
        ("tensorType",ctypes.c_int),
        ("shift",hbDNNQuantiShift_yt),
        ("scale",hbDNNQuantiScale_t),
        ("quantiType",ctypes.c_int),
        ("quantizeAxis", ctypes.c_int),
        ("alignedByteSize",ctypes.c_int),
        ("stride",ctypes.c_int * 8)
    ]

class hbDNNTensor_t(ctypes.Structure):
    _fields_ = [
        ("sysMem",hbSysMem_t * 4),
        ("properties",hbDNNTensorProperties_t)
    ]


class FcosPostProcessInfo_t(ctypes.Structure):
    _fields_ = [
        ("height",ctypes.c_int),
        ("width",ctypes.c_int),
        ("ori_height",ctypes.c_int),
        ("ori_width",ctypes.c_int),
        ("score_threshold",ctypes.c_float),
        ("nms_threshold",ctypes.c_float),
        ("nms_top_k",ctypes.c_int),
        ("is_pad_resize",ctypes.c_int)
    ]


libpostprocess = ctypes.CDLL('/usr/lib/libpostprocess.so') 

get_Postprocess_result = libpostprocess.FcosPostProcess
get_Postprocess_result.argtypes = [ctypes.POINTER(FcosPostProcessInfo_t)]  
get_Postprocess_result.restype = ctypes.c_char_p  

def get_TensorLayout(Layout):
    if Layout == "NCHW":
        return int(2)
    else:
        return int(0)

is_stop=False
def get_display_res():
    if os.path.exists("/usr/bin/get_hdmi_res") == False:
        return 1920, 1080

    import subprocess
    p = subprocess.Popen(["/usr/bin/get_hdmi_res"], stdout=subprocess.PIPE)
    result = p.communicate()
    res = result[0].split(b',')
    res[1] = max(min(int(res[1]), 1920), 0)
    res[0] = max(min(int(res[0]), 1080), 0)
    return int(res[1]), int(res[0])

disp_w, disp_h = get_display_res()

def print_properties(pro):
    print("tensor type:", pro.tensor_type)
    print("data type:", pro.dtype)
    print("layout:", pro.layout)
    print("shape:", pro.shape)


def get_hw(pro):
    if pro.layout == "NCHW":
        return pro.shape[2], pro.shape[3]
    else:
        return pro.shape[1], pro.shape[2]

# detection model class names
def get_classes():
    return np.array([
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    ])

class ParallelExector(object):
    def __init__(self, parallel_num=4):
        self.parallel_num = parallel_num
        if parallel_num != 1:
            self._pool = multiprocessing.Pool(processes=self.parallel_num,
                                              maxtasksperchild=5)
            self.workers = BoundedSemaphore(self.parallel_num)

    def infer(self, output):
        if self.parallel_num == 1:
            run(output)
        else:
            self.workers.acquire()
            self._pool.apply_async(func=run,
                                   args=(output, ),
                                   callback=self.task_done,
                                   error_callback=print)

    def task_done(self, *args, **kwargs):
        """Called once task is done, releases the queue is blocked."""
        self.workers.release()

    def close(self):
        if hasattr(self, "_pool"):
            self._pool.close()
            self._pool.join()


def limit_display_cord(coor):
    coor[0] = max(min(disp_w, coor[0]), 0)
    # min coor is set to 2 not 0, leaving room for string display
    coor[1] = max(min(disp_h, coor[1]), 2)
    coor[2] = max(min(disp_w, coor[2]), 0)
    coor[3] = max(min(disp_h, coor[3]), 0)
    return coor


def run(outputs):
    # Do post process
    strides = [8, 16, 32, 64, 128]
    for i in range(len(strides)):
        if (output_tensors[i].properties.quantiType == 0):
            output_tensors[i].sysMem[0].virAddr = ctypes.cast(outputs[i].ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.c_void_p)
            output_tensors[i + 5].sysMem[0].virAddr = ctypes.cast(outputs[i + 5].ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.c_void_p)
            output_tensors[i + 10].sysMem[0].virAddr = ctypes.cast(outputs[i + 10].ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.c_void_p)
        else:  
            output_tensors[i].sysMem[0].virAddr = ctypes.cast(outputs[i].ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ctypes.c_void_p)
            output_tensors[i + 5].sysMem[0].virAddr = ctypes.cast(outputs[i + 5].ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ctypes.c_void_p)
            output_tensors[i + 10].sysMem[0].virAddr = ctypes.cast(outputs[i + 10].ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ctypes.c_void_p)
        
        libpostprocess.FcosdoProcess(output_tensors[i], output_tensors[i + 5], output_tensors[i + 10], ctypes.pointer(fcos_postprocess_info), i)
        
    result_str = get_Postprocess_result(ctypes.pointer(fcos_postprocess_info))  
    result_str = result_str.decode('utf-8')  
    # print(result_str)

    # draw result
    # 解析JSON字符串  
    data = json.loads(result_str[14:])  

    # 遍历每一个结果  
    for index, result in enumerate(data): 
        bbox = result['bbox']  # 矩形框位置信息  
        score = result['score']  # 得分  
        id = int(result['id'])  # id  
        name = result['name']  # 类别名称 

        coor = limit_display_cord(bbox)
        coor = [round(i) for i in coor]
        # get bbox score
        score = float(score)
        # concat bbox string
        bbox_string = '%s: %.2f' % (name, score)
        bbox_string = bbox_string.encode('gb2312')
        # concat bbox color
        box_color = colors[id]
        color_base = 0xFF000000
        box_color_ARGB = color_base | (box_color[0]) << 16 | (
            box_color[1]) << 8 | (box_color[2])

        # print("{} is in the picture with confidence:{:.4f}, bbox:{}".format(name, score, coor))

        # if new frame come in, need to flush the display buffer.
        # For the meaning of parameters, please refer to the relevant documents of display api
        if index == 0:
            disp.set_graph_rect(coor[0], coor[1], coor[2], coor[3], 3, 1,
                                box_color_ARGB)
            disp.set_graph_word(coor[0], coor[1] - 2, bbox_string, 3, 1,
                                box_color_ARGB)
        else:
            disp.set_graph_rect(coor[0], coor[1], coor[2], coor[3], 3, 0,
                                box_color_ARGB)
            disp.set_graph_word(coor[0], coor[1] - 2, bbox_string, 3, 0,
                                box_color_ARGB)


def get_nalu_pos(byte_stream):
    size = byte_stream.__len__()
    nals = []
    retnals = []

    startCodePrefixShort = b"\x00\x00\x01"

    pos = 0
    while pos < size:
        is4bytes = False
        retpos = byte_stream.find(startCodePrefixShort, pos)
        if retpos == -1:
            break
        if byte_stream[retpos - 1] == 0:
            retpos -= 1
            is4bytes = True
        if is4bytes:
            pos = retpos + 4
        else:
            pos = retpos + 3
        val = hex(byte_stream[pos])
        val = "{:d}".format(byte_stream[pos], 4)
        val = int(val)
        fb = (val >> 7) & 0x1
        nri = (val >> 5) & 0x3
        type = val & 0x1f
        nals.append((pos, is4bytes, fb, nri, type))
    for i in range(0, len(nals) - 1):
        start = nals[i][0]
        if nals[i + 1][1]:
            end = nals[i + 1][0] - 5
        else:
            end = nals[i + 1][0] - 4
        retnals.append((start, end, nals[i][1], nals[i][2], nals[i][3], nals[i][4]))
    start = nals[-1][0]
    end = byte_stream.__len__() - 1
    retnals.append((start, end, nals[-1][1], nals[-1][2], nals[-1][3], nals[-1][4]))
    return retnals

def get_h264_nalu_type(byte_stream):
    nalu_types = []
    nalu_pos = get_nalu_pos(byte_stream)

    for idx, (start, end, is4bytes, fb, nri, type) in enumerate(nalu_pos):
        # print("NAL#%d: %d, %d, %d, %d, %d" % (idx, start, end, fb, nri, type))
        nalu_types.append(type)
    
    return nalu_types

def bytes_to_numpy(image_bytes):
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
    # image_np2 = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    print(type(image_np),image_np.shape)
    return image_np

class DecodeRtspStream(threading.Thread):
    def __init__(self, rtsp_url):
        threading.Thread.__init__(self)
        self.rtsp_url = rtsp_url
        self.is_running = True
        self.frame_queue = Queue(maxsize=2)

    def open(self, dec_chn=0, dec_type=1):
        self.dec_chn = dec_chn
        self.dec_type = dec_type
        cap = cv2.VideoCapture(self.rtsp_url)
        cap.set(cv2.CAP_PROP_FORMAT, -1) # get stream data
        if not cap.isOpened():
            print("fail to open rtsp: {}".format(self.rtsp_url))
            return -1

        print("RTSP stream frame_width:{:.0f}, frame_height:{:.0f}"
                    .format(cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.cap = cap
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # init decode module
        self.dec = srcampy.Decoder()
        ret = self.dec.decode("", dec_chn, dec_type, self.width, self.height)
        print("Decoder(%d, %d) return:%d frame count: %d" %(dec_chn, dec_type, ret[0], ret[1]))

        return ret[0]

    def close(self):
        self.is_running = False
        self.join()
        self.dec.close()
        self.cap.release()

    def run(self):
        global is_stop
        start_time = time()
        image_count = 0
        skip_count = 0
        find_pps_sps = 0
        while not is_stop:
            ret, stream_frame = self.cap.read()
            if not ret:
                self.dec.close()
                self.cap.release()
                ret = self.open(self.dec_chn, self.dec_type)
                if ret != 0:
                    return ret
                start_time = time()
                image_count = 0
                skip_count = 0
                continue
            nalu_types = get_h264_nalu_type(stream_frame.tobytes())
            # print("ret:{}, len{}, type{}, nalu_types{}".format(ret, stream_frame.shape, type(stream_frame), nalu_types))

            # 送入解码的第一帧需要是 pps，sps, 否则解码器会报 "FAILED TO DEC_PIC_HDR" 异常而退出
            if (nalu_types[0] in [1, 5]) and find_pps_sps == 0:
                continue

            # if (nalu_types[0] in [6, 7, 8]):
            #     print("ret:{}, len{}, type{}, nalu_types{}".format(ret, stream_frame.shape, type(stream_frame), nalu_types))

            find_pps_sps = 1
            if stream_frame is not None:
                ret = self.dec.set_img(stream_frame.tobytes(), self.dec_chn) # 发送码流, 先解码数帧图像后再获取
                if ret != 0:
                    return ret
                if skip_count < 8:
                    skip_count += 1
                    image_count = 0
                    continue

                frame = self.dec.get_img() # 获取nv12格式的yuv帧数据

                if frame is not None:
                    # print("self.frame_queue.full(): qsize: ", self.frame_queue.full(), self.frame_queue.qsize())
                    if self.frame_queue.full() == False:
                        self.frame_queue.put(frame)

            finish_time = time()
            image_count += 1
            if finish_time - start_time > 3:
                # print(start_time, finish_time, image_count)
                print("Decode CHAN: {:d} FPS: {:.2f}".format(self.dec_chn, image_count / (finish_time - start_time)))
                start_time = finish_time
                image_count = 0

    def get_frame(self):
        # print("self.frame_queue.empty(): qsize: ", self.frame_queue.empty(), self.frame_queue.qsize())
        if self.frame_queue.empty() == True:
            return None
        return self.frame_queue.get()

class VideoDisplay(threading.Thread):
    def __init__(self, streamer, vps_group):
        threading.Thread.__init__(self)
        self.streamer = streamer
        self.vps_group = vps_group
        self.frame_queue = Queue(maxsize=2)
        self.is_running = True
        # Get HDMI display object
        self.disp = srcampy.Display()
        # For the meaning of parameters, please refer to the relevant documents of HDMI display
        self.disp.display(0, disp_w, disp_h)

        # change disp for bbox display
        self.disp.display(3, disp_w, disp_h)

        # vps start
        self.vps = srcampy.Camera()
        ret = self.vps.open_vps(self.vps_group, 1, self.streamer.width, self.streamer.height, [1920, 512, disp_w], [1080, 512, disp_h])
        print("Camera vps return:%d" % ret)

        # 绑定 vps 和 disp，这样vps的输出不需要软件操作直接可以输出显示，可以提供性能
        srcampy.bind(self.vps, self.disp)

    def close(self):
        self.is_running = False
        self.join()
        print("dis stop success")
        self.disp.close()
        self.vps.close_cam()
        srcampy.unbind(self.vps, self.disp)
       

    def run(self):
        global is_stop
        disp_start_time = time()
        disp_image_count = 0
        while not is_stop:
            frame = self.streamer.get_frame() # 获取nv12格式的yuv帧数据
            if frame is None:
                sleep(0.01)
                continue

            self.vps.set_img(frame)

            # print("self.frame_queue.full(): qsize: ", self.frame_queue.full(), self.frame_queue.qsize())
            if self.frame_queue.full() == False:
                sleep(0.001) # 这一延时不可缺少
                nv12_img = self.vps.get_img(2, 512, 512)
                self.frame_queue.put(nv12_img)

            disp_finish_time = time()
            disp_image_count += 1
            if disp_finish_time - disp_start_time > 3:
                # print(start_time, finish_time, image_count)
                print("Display FPS: {:.2f}".format(disp_image_count / (disp_finish_time - disp_start_time)))
                disp_start_time = disp_finish_time
                disp_image_count = 0
        self.disp.close()

    def get_frame(self):
        # print("self.frame_queue.empty(): qsize: ", self.frame_queue.empty(), self.frame_queue.qsize())
        if self.frame_queue.empty() == True:
            return None
        return self.frame_queue.get()


class AiInference(threading.Thread):
    def __init__(self, video_display, models):
        threading.Thread.__init__(self)
        self.video_display = video_display
        # self.modle_file = modle_file

        self.models = models#dnn.load(self.modle_file)

    def close(self):
        pass


    def run(self):
        # post process parallel executor
        parallel_exe = ParallelExector()
        global is_stop
        ai_start_time = time()
        ai_image_count = 0

        while not is_stop:
            img = self.video_display.get_frame() # 获取nv12格式的yuv帧数据
            if img is None:
                sleep(0.02)
                continue

            # Convert to numpy
            img = np.frombuffer(img, dtype=np.uint8)

            # Forward
            outputs = self.models[0].forward(img)

            output_array = []
            for item in outputs:
                output_array.append(item.buffer)
            parallel_exe.infer(output_array)

            ai_finish_time = time()
            ai_image_count += 1
            if ai_finish_time - ai_start_time > 3:
                print("AI FPS: {:.2f}".format(ai_image_count / (ai_finish_time - ai_start_time)))
                ai_start_time = ai_finish_time
                ai_image_count = 0

    def print_properties(self, pro):
        print("tensor type:", pro.tensor_type)
        print("data type:", pro.dtype)
        print("layout:", pro.layout)
        print("shape:", pro.shape)


    def get_hw(self, pro):
        if pro.layout == "NCHW":
            return pro.shape[2], pro.shape[3]
        else:
            return pro.shape[1], pro.shape[2]
def signal_handler(sig, frame):
    print("Ctrl+C received. Closing app.")
    global is_stop 
    is_stop = True


if __name__ == '__main__':
    # rtsp_urls = ["rtsp://username:passwd@127.0.0.1/1080P_test.h264"]
    rtsp_urls = ["rtsp://127.0.0.1/1080P_test.h264"]
    
    enable_display = 1
    enable_ai_inference = 0
    signal.signal(signal.SIGINT, signal_handler)
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hu:d:a",["rtsp_url="])
    except getopt.GetoptError:
        print('./decode_rtsp_stream.py [-u <rtsp_url>] [-d] [-a]')
        print('./decode_rtsp_stream.py [-u <rtsp_url;rtsp_url2>] [-d] [-a]')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('./decode_rtsp_stream.py [-u <rtsp_url>] [-d] [-a]')
            print('./decode_rtsp_stream.py [-u <rtsp_url;rtsp_url2>] [-d] [-a]')
            sys.exit()
        elif opt in ("-u", "--rtsp_url"):
            rtsp_urls = arg.split(";")
        elif opt in ("-d"):
            enable_display = int(arg)
        elif  opt in ("-a"):
            # 使能ai的时候，hdmi的显示也必须使能，因为算法结果需要绘制到可视化图像上
            enable_display = 1
            enable_ai_inference = 1

    print(rtsp_urls)

    vdec_chan = 0 # 解码器通道
    rtsp_streams = []
    for rtsp_url in rtsp_urls:
        rtsp_stream = DecodeRtspStream(rtsp_url) # 初始化一个实例，并且设置这一路流会显示到输出显示屏上
        ret = rtsp_stream.open(vdec_chan, 1) # 打开rtsp流，并初始化解码器通道0，编码格式为H264
        if ret != 0:
            quit(ret)
        rtsp_stream.start()

        rtsp_streams.append(rtsp_stream)

        vdec_chan += 1 # 下一路打开的解码器通道

    if enable_display == 1:
        # 把解码实例 rtsp_stream 作为显示示例的输入
        video_display = VideoDisplay(rtsp_streams[0], 1)
        video_display.start()

    if enable_ai_inference == 1:
        # setup for box color and box string
        classes = get_classes()
        num_classes = len(classes)
        hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                colors))

        disp = video_display.disp

        models = dnn.load('../models/fcos_512x512_nv12.bin')

        # get model info
        h, w = get_hw(models[0].inputs[0].properties)
        input_shape = (h, w)
        # 打印输出 tensor 的属性
        for output in models[0].outputs:
            print_properties(output.properties)

        # 获取结构体信息
        fcos_postprocess_info = FcosPostProcessInfo_t()
        fcos_postprocess_info.height = h
        fcos_postprocess_info.width = w
        fcos_postprocess_info.ori_height = disp_h
        fcos_postprocess_info.ori_width = disp_w
        fcos_postprocess_info.score_threshold = 0.5 
        fcos_postprocess_info.nms_threshold = 0.6
        fcos_postprocess_info.nms_top_k = 5
        fcos_postprocess_info.is_pad_resize = 0

        output_tensors = (hbDNNTensor_t * len(models[0].outputs))()
        
        for i in range(len(models[0].outputs)):
            output_tensors[i].properties.tensorLayout = get_TensorLayout(models[0].outputs[i].properties.layout)
            # print(output_tensors[i].properties.tensorLayout)
            if (len(models[0].outputs[i].properties.scale_data) == 0):
                output_tensors[i].properties.quantiType = 0
            else:
                output_tensors[i].properties.quantiType = 2  
                scale_data_tmp = models[0].outputs[i].properties.scale_data.reshape(1, 1, 1, models[0].outputs[i].properties.shape[3])  
                output_tensors[i].properties.scale.scaleData = scale_data_tmp.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                
            for j in range(len(models[0].outputs[i].properties.shape)):
                output_tensors[i].properties.validShape.dimensionSize[j] = models[0].outputs[i].properties.shape[j]
                output_tensors[i].properties.alignedShape.dimensionSize[j] = models[0].outputs[i].properties.shape[j]
    
        ai_inference = AiInference(video_display, models)
        ai_inference.start()
