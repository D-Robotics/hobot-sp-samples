#!/usr/bin/env python3
import sys, os
import signal
import numpy as np
import cv2
import google.protobuf
import asyncio
import websockets
import x3_pb2
import time
import subprocess

# Camera API libs
from hobot_vio import libsrcampy as srcampy
from hobot_dnn import pyeasy_dnn
fps = 30

import ctypes
import json 

image_counter = None

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

def signal_handler(signal, frame):
    sys.exit(0)

# detection model class names
def get_classes():
    return np.array(["person", "bicycle", "car",
                     "motorcycle", "airplane", "bus",
                     "train", "truck", "boat",
                     "traffic light", "fire hydrant", "stop sign",
                     "parking meter", "bench", "bird",
                     "cat", "dog", "horse",
                     "sheep", "cow", "elephant",
                     "bear", "zebra", "giraffe",
                     "backpack", "umbrella", "handbag",
                     "tie", "suitcase", "frisbee",
                     "skis", "snowboard", "sports ball",
                     "kite", "baseball bat", "baseball glove",
                     "skateboard", "surfboard", "tennis racket",
                     "bottle", "wine glass", "cup",
                     "fork", "knife", "spoon",
                     "bowl", "banana", "apple",
                     "sandwich", "orange", "broccoli",
                     "carrot", "hot dog", "pizza",
                     "donut", "cake", "chair",
                     "couch", "potted plant", "bed",
                     "dining table", "toilet", "tv",
                     "laptop", "mouse", "remote",
                     "keyboard", "cell phone", "microwave",
                     "oven", "toaster", "sink",
                     "refrigerator", "book", "clock",
                     "vase", "scissors", "teddy bear",
                     "hair drier", "toothbrush"])


def bgr2nv12_opencv(image):
    height, width = image.shape[0], image.shape[1]
    area = height * width
    yuv420p = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
    y = yuv420p[:area]
    uv_planar = yuv420p[area:].reshape((2, area // 4))
    uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))

    nv12 = np.zeros_like(yuv420p)
    nv12[:height * width] = y
    nv12[height * width:] = uv_packed
    return nv12


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

def limit_display_cord(coor):
    coor[0] = max(min(1920, coor[0]), 0)
    # min coor is set to 2 not 0, leaving room for string display
    coor[1] = max(min(1080, coor[1]), 2)
    coor[2] = max(min(1920, coor[2]), 0)
    coor[3] = max(min(1080, coor[3]), 0)
    return coor

def serialize(FrameMessage, data):

    if data:
        for result in data: 
            # get class name
            Target = x3_pb2.Target()
            bbox = result['bbox']  # 矩形框位置信息  
            score = result['score']  # 得分  
            id = int(result['id'])  # id  
            name = result['name']  # 类别名称 

            # print(f"bbox: {bbox}, score: {score}, id: {id}, name: {name}")
            bbox = limit_display_cord(bbox)
            Target.type_ = classes[id]
            Box = x3_pb2.Box()
            Box.type_ = classes[id]
            Box.score_ = float(score)

            Box.top_left_.x_ = int(bbox[0])
            Box.top_left_.y_ = int(bbox[1])
            Box.bottom_right_.x_ = int(bbox[2])
            Box.bottom_right_.y_ = int(bbox[3])

            Target.boxes_.append(Box)
            FrameMessage.smart_msg_.targets_.append(Target)

    prot_buf = FrameMessage.SerializeToString()
    return prot_buf

models = pyeasy_dnn.load('../models/fcos_efficientnetb0_512x512_nv12_dq.bin')
input_shape = (512, 512)
cam = srcampy.Camera()
cam.open_cam(0, -1, fps, [512,1920], [512,1080])
enc = srcampy.Encoder()
enc.encode(0, 3, 1920, 1080)
classes = get_classes()
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
fcos_postprocess_info.ori_height = 1080
fcos_postprocess_info.ori_width = 1920
fcos_postprocess_info.score_threshold = 0.5 
fcos_postprocess_info.nms_threshold = 0.6
fcos_postprocess_info.nms_top_k = 500
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


async def web_service(websocket, path):
    while True:
        FrameMessage = x3_pb2.FrameMessage()
        FrameMessage.img_.height_ = 1080
        FrameMessage.img_.width_ = 1920
        FrameMessage.img_.type_ = "JPEG"

        img = cam.get_img(2, 512, 512)
        img = np.frombuffer(img, dtype=np.uint8)
        t0 = time.time()
        outputs = models[0].forward(img)
        t1 = time.time()
        print("forward time is :", (t1 - t0))


        strides = [8, 16, 32, 64, 128]
        for i in range(len(strides)):
            if (output_tensors[i].properties.quantiType == 0):
                output_tensors[i].sysMem[0].virAddr = ctypes.cast(outputs[i].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.c_void_p)
                output_tensors[i + 5].sysMem[0].virAddr = ctypes.cast(outputs[i + 5].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.c_void_p)
                output_tensors[i + 10].sysMem[0].virAddr = ctypes.cast(outputs[i + 10].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.c_void_p)
            else:  
                output_tensors[i].sysMem[0].virAddr = ctypes.cast(outputs[i].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ctypes.c_void_p)
                output_tensors[i + 5].sysMem[0].virAddr = ctypes.cast(outputs[i + 5].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ctypes.c_void_p)
                output_tensors[i + 10].sysMem[0].virAddr = ctypes.cast(outputs[i + 10].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ctypes.c_void_p)

            libpostprocess.FcosdoProcess(output_tensors[i], output_tensors[i + 5], output_tensors[i + 10], fcos_postprocess_info, i)

        result_str = get_Postprocess_result(ctypes.pointer(fcos_postprocess_info))  
        result_str = result_str.decode('utf-8')  
        t2 = time.time()
        print("FcosdoProcess time is :", (t2 - t1))
        # print(result_str)

        # draw result
        # 解析JSON字符串  
        data = json.loads(result_str[14:])  

        origin_image = cam.get_img(2, 1920, 1080)
        enc.encode_file(origin_image)
        FrameMessage.img_.buf_ = enc.get_img()
        FrameMessage.smart_msg_.timestamp_ = int(time.time())

        prot_buf = serialize(FrameMessage, data)

        await websocket.send(prot_buf)

    cam.close_cam()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    start_server = websockets.serve(web_service, "0.0.0.0", 8080)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
