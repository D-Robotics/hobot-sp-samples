#!/usr/bin/env python3
import sys, os
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


def postprocess(model_output,
                model_hw_shape,
                origin_image=None,
                origin_img_shape=None,
                score_threshold=0.5,
                nms_threshold=0.6,
                dump_image=False):
    input_height = model_hw_shape[0]
    input_width = model_hw_shape[1]
    if origin_image is not None:
        origin_image_shape = origin_image.shape[0:2]
    else:
        origin_image_shape = origin_img_shape

    prediction_bbox = decode(outputs=model_output,
                             score_threshold=score_threshold,
                             origin_shape=origin_image_shape,
                             input_size=512)

    prediction_bbox = nms(prediction_bbox, iou_threshold=nms_threshold)

    prediction_bbox = np.array(prediction_bbox)
    topk = min(prediction_bbox.shape[0], 1000)

    if topk != 0:
        idx = np.argpartition(prediction_bbox[..., 4], -topk)[-topk:]
        prediction_bbox = prediction_bbox[idx]

    if dump_image and origin_image is not None:
        draw_bboxs(origin_image, prediction_bbox)
    return prediction_bbox


def decode(outputs, score_threshold, origin_shape, input_size=512):
    def _distance2bbox(points, distance):
        x1 = points[..., 0] - distance[..., 0]
        y1 = points[..., 1] - distance[..., 1]
        x2 = points[..., 0] + distance[..., 2]
        y2 = points[..., 1] + distance[..., 3]
        return np.stack([x1, y1, x2, y2], -1)

    def _scores(cls, ce):
        cls = 1 / (1 + np.exp(-cls))
        ce = 1 / (1 + np.exp(-ce))
        return np.sqrt(ce * cls)

    def _bbox(bbox, stride, origin_shape, input_size):
        h, w = bbox.shape[1:3]
        yv, xv = np.meshgrid(np.arange(h), np.arange(w))
        xy = (np.stack((yv, xv), 2) + 0.5) * stride
        bbox = _distance2bbox(xy, bbox)
        # opencv read, shape[1] is w, shape[0] is h
        scale_w = origin_shape[1] / input_size
        scale_h = origin_shape[0] / input_size
        scale = max(origin_shape[0], origin_shape[1]) / input_size
        # origin img is pad resized
        # bbox = bbox * scale
        # origin img is resized
        bbox = bbox * [scale_w, scale_h, scale_w, scale_h]
        return bbox

    bboxes = list()
    strides = [8, 16, 32, 64, 128]

    for i in range(len(strides)):
        cls = outputs[i].buffer
        bbox = outputs[i + 5].buffer
        ce = outputs[i + 10].buffer
        scores = _scores(cls, ce)

        classes = np.argmax(scores, axis=-1)
        classes = np.reshape(classes, [-1, 1])
        max_score = np.max(scores, axis=-1)
        max_score = np.reshape(max_score, [-1, 1])
        bbox = _bbox(bbox, strides[i], origin_shape, input_size)
        bbox = np.reshape(bbox, [-1, 4])

        pred_bbox = np.concatenate([bbox, max_score, classes], axis=1)

        index = pred_bbox[..., 4] > score_threshold
        pred_bbox = pred_bbox[index]
        bboxes.append(pred_bbox)

    return np.concatenate(bboxes)


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    def bboxes_iou(boxes1, boxes2):
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)
        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * \
                      (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * \
                      (boxes2[..., 3] - boxes2[..., 1])
        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        ious = np.maximum(1.0 * inter_area / union_area,
                          np.finfo(np.float32).eps)

        return ious

    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate(
                [cls_bboxes[:max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0
            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


def serialize(FrameMessage, prediction_bbox):
    if (prediction_bbox.shape[0] > 0):
        for i in range(prediction_bbox.shape[0]):
            # get class name
            Target = x3_pb2.Target()
            id = int(prediction_bbox[i][5])
            Target.type_ = classes[id]
            Box = x3_pb2.Box()
            Box.type_ = classes[id]
            Box.score_ = prediction_bbox[i][4]

            Box.top_left_.x_ = prediction_bbox[i][0]
            Box.top_left_.y_ = prediction_bbox[i][1]
            Box.bottom_right_.x_ = prediction_bbox[i][2]
            Box.bottom_right_.y_ = prediction_bbox[i][3]

            Target.boxes_.append(Box)
            FrameMessage.smart_msg_.targets_.append(Target)
    prot_buf = FrameMessage.SerializeToString()
    return prot_buf


models = pyeasy_dnn.load('../models/fcos_512x512_nv12.bin')
input_shape = (512, 512)
cam = srcampy.Camera()
cam.open_cam(0, -1, fps, [512,1920], [512,1080])
enc = srcampy.Encoder()
enc.encode(0, 3, 1920, 1080)
classes = get_classes()


async def web_service(websocket, path):
    while True:
        FrameMessage = x3_pb2.FrameMessage()
        FrameMessage.img_.height_ = 1080
        FrameMessage.img_.width_ = 1920
        FrameMessage.img_.type_ = "JPEG"

        img = cam.get_img(2, 512, 512)
        img = np.frombuffer(img, dtype=np.uint8)
        outputs = models[0].forward(img)
        # Do post process
        prediction_bbox = postprocess(outputs, input_shape, origin_img_shape=(1080, 1920))
        print(prediction_bbox)

        origin_image = cam.get_img(2, 1920, 1080)
        enc.encode_file(origin_image)
        FrameMessage.img_.buf_ = enc.get_img()
        FrameMessage.smart_msg_.timestamp_ = int(time.time())

        prot_buf = serialize(FrameMessage, prediction_bbox)

        await websocket.send(prot_buf)
    cam.close_cam()


if __name__ == '__main__':
    start_server = websockets.serve(web_service, "0.0.0.0", 8080)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
