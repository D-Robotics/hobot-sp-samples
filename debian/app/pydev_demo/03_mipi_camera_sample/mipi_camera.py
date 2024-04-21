#!/usr/bin/env python3
import sys
import signal
import os
import numpy as np
import cv2
import colorsys
from time import time,sleep
import multiprocessing
from threading import BoundedSemaphore
import ctypes
import json
# Camera API libs

from hobot_vio import libsrcampy as srcampy
from hobot_dnn import pyeasy_dnn as dnn
import threading

image_counter = None
is_stop=False
output_tensors = None

fcos_postprocess_info = None

def signal_handler(signal, frame):
    sys.exit(0)
    global is_stop
    print("Stopping!\n")
    is_stop=True

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

def signal_handler(signal, frame):
    global is_stop
    print("Stopping!\n")
    is_stop=True


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


def get_hw(pro):
    if pro.layout == "NCHW":
        return pro.shape[2], pro.shape[3]
    else:
        return pro.shape[1], pro.shape[2]

def print_properties(pro):
    print("tensor type:", pro.tensor_type)
    print("data type:", pro.dtype)
    print("layout:", pro.layout)
    print("shape:", pro.shape)


class ParallelExector(object):
    def __init__(self, counter, parallel_num=4):
        global image_counter
        image_counter = counter
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
    global image_counter

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

        print("{} is in the picture with confidence:{:.4f}, bbox:{}".format(name, score, coor))

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

    # fps timer and counter
    with image_counter.get_lock():
        image_counter.value += 1
    if image_counter.value == 100:
        finish_time = time()
        print(
            f"Total time cost for 100 frames: {finish_time - start_time}, fps: {100/(finish_time - start_time)}"
        )


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)

    models = dnn.load('../models/fcos_512x512_nv12.bin')
    print("--- model input properties ---")
    # 打印输入 tensor 的属性
    print_properties(models[0].inputs[0].properties)
    print("--- model output properties ---")
    # 打印输出 tensor 的属性
    for output in models[0].outputs:
        print_properties(output.properties)

    # 获取结构体信息
    fcos_postprocess_info = FcosPostProcessInfo_t()
    fcos_postprocess_info.height = 512
    fcos_postprocess_info.width = 512
    fcos_postprocess_info.ori_height = disp_h
    fcos_postprocess_info.ori_width = disp_w
    fcos_postprocess_info.score_threshold = 0.5
    fcos_postprocess_info.nms_threshold = 0.6
    fcos_postprocess_info.nms_top_k = 5
    fcos_postprocess_info.is_pad_resize = 0

    output_tensors = (hbDNNTensor_t * len(models[0].outputs))()

    for i in range(len(models[0].outputs)):
        output_tensors[i].properties.tensorLayout = get_TensorLayout(models[0].outputs[i].properties.layout)
        #print(output_tensors[i].properties.tensorLayout)
        if (len(models[0].outputs[i].properties.scale_data) == 0):
            output_tensors[i].properties.quantiType = 0
        else:
            output_tensors[i].properties.quantiType = 2
            scale_data_tmp = models[0].outputs[i].properties.scale_data.reshape(1, 1, 1, models[0].outputs[i].properties.shape[3])
            output_tensors[i].properties.scale.scaleData = scale_data_tmp.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        for j in range(len(models[0].outputs[i].properties.shape)):
            output_tensors[i].properties.validShape.dimensionSize[j] = models[0].outputs[i].properties.shape[j]
            output_tensors[i].properties.alignedShape.dimensionSize[j] = models[0].outputs[i].properties.shape[j]


    # Camera API, get camera object
    cam = srcampy.Camera()

    # get model info
    h, w = get_hw(models[0].inputs[0].properties)
    input_shape = (h, w)
    # Open f37 camera
    # For the meaning of parameters, please refer to the relevant documents of camera
    cam.open_cam(0, -1, 30, [w, disp_w], [h, disp_h])

    # Get HDMI display object
    disp = srcampy.Display()
    # For the meaning of parameters, please refer to the relevant documents of HDMI display
    disp.display(0, disp_w, disp_h)

    # bind camera directly to display
    srcampy.bind(cam, disp)

    # change disp for bbox display
    disp.display(3, disp_w, disp_h)

    # setup for box color and box string
    classes = get_classes()
    num_classes = len(classes)
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))

    # fps timer and counter
    start_time = time()
    image_counter = multiprocessing.Value("i", 0)

    # post process parallel executor
    parallel_exe = ParallelExector(image_counter)

    while not is_stop:
        # image_counter += 1
        # Get image data with shape of 512x512 nv12 data from camera
        cam_start_time = time()
        img = cam.get_img(2, 512, 512)
        cam_finish_time = time()

        # Convert to numpy
        buffer_start_time = time()
        img = np.frombuffer(img, dtype=np.uint8)
        buffer_finish_time = time()

        # Forward
        infer_start_time = time()
        outputs = models[0].forward(img)
        infer_finish_time = time()

        output_array = []
        for item in outputs:
            output_array.append(item.buffer)
        parallel_exe.infer(output_array)

    cam.close_cam()
    disp.close()
