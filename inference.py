import tensorflow as tf
import cv2
import os
import numpy as np
from typing import Dict, Optional, Tuple
import random
import math

_COLORS = (
    np.array(
        [
            0.000,
            0.447,
            0.741,
            0.850,
            0.325,
            0.098,
            0.929,
            0.694,
            0.125,
            0.494,
            0.184,
            0.556,
            0.466,
            0.674,
            0.188,
            0.301,
            0.745,
            0.933,
            0.635,
            0.078,
            0.184,
            0.300,
            0.300,
            0.300,
            0.600,
            0.600,
            0.600,
            1.000,
            0.000,
            0.000,
            1.000,
            0.500,
            0.000,
            0.749,
            0.749,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.333,
            0.333,
            0.000,
            0.333,
            0.667,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            0.333,
            0.000,
            0.667,
            0.667,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            1.000,
            0.000,
            0.000,
            0.333,
            0.500,
            0.000,
            0.667,
            0.500,
            0.000,
            1.000,
            0.500,
            0.333,
            0.000,
            0.500,
            0.333,
            0.333,
            0.500,
            0.333,
            0.667,
            0.500,
            0.333,
            1.000,
            0.500,
            0.667,
            0.000,
            0.500,
            0.667,
            0.333,
            0.500,
            0.667,
            0.667,
            0.500,
            0.667,
            1.000,
            0.500,
            1.000,
            0.000,
            0.500,
            1.000,
            0.333,
            0.500,
            1.000,
            0.667,
            0.500,
            1.000,
            1.000,
            0.500,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.333,
            0.333,
            1.000,
            0.333,
            0.667,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.667,
            0.333,
            1.000,
            0.667,
            0.667,
            1.000,
            0.667,
            1.000,
            1.000,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            1.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.143,
            0.143,
            0.143,
            0.286,
            0.286,
            0.286,
            0.429,
            0.429,
            0.429,
            0.571,
            0.571,
            0.571,
            0.714,
            0.714,
            0.714,
            0.857,
            0.857,
            0.857,
            0.000,
            0.447,
            0.741,
            0.314,
            0.717,
            0.741,
            0.50,
            0.5,
            0,
        ]
    )
    .astype(np.float32)
    .reshape(-1, 3)
)

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
              'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
              'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
              'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
              'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
              'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
              'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
              'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
              'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
              'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
              'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
              'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
              'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush']


def get_minimum_dst_shape(
    src_shape: Tuple[int, int],
    dst_shape: Tuple[int, int],
    divisible: Optional[int] = None,
) -> Tuple[int, int]:
    """Calculate minimum dst shape"""
    src_w, src_h = src_shape
    dst_w, dst_h = dst_shape

    if src_w / src_h < dst_w / dst_h:
        ratio = dst_h / src_h
    else:
        ratio = dst_w / src_w

    dst_w = int(ratio * src_w)
    dst_h = int(ratio * src_h)

    if divisible and divisible > 0:
        dst_w = max(divisible, int((dst_w + divisible - 1) // divisible * divisible))
        dst_h = max(divisible, int((dst_h + divisible - 1) // divisible * divisible))
    return dst_w, dst_h

def get_resize_matrix(raw_shape, dst_shape, keep_ratio):
    """
    Get resize matrix for resizing raw img to input size
    :param raw_shape: (width, height) of raw image
    :param dst_shape: (width, height) of input image
    :param keep_ratio: whether keep original ratio
    :return: 3x3 Matrix
    """
    r_w, r_h = raw_shape
    d_w, d_h = dst_shape
    Rs = np.eye(3)
    if keep_ratio:
        C = np.eye(3)
        C[0, 2] = -r_w / 2
        C[1, 2] = -r_h / 2

        if r_w / r_h < d_w / d_h:
            ratio = d_h / r_h
        else:
            ratio = d_w / r_w
        Rs[0, 0] *= ratio
        Rs[1, 1] *= ratio

        T = np.eye(3)
        T[0, 2] = 0.5 * d_w
        T[1, 2] = 0.5 * d_h
        return T @ Rs @ C
    else:
        Rs[0, 0] *= d_w / r_w
        Rs[1, 1] *= d_h / r_h
        return Rs

def generate_grid_center_priors(input_height, input_width, strides):
    center_priors = []
    for stride in strides:
        feat_w = math.ceil(input_width / stride)
        feat_h = math.ceil(input_height / stride)
        for y in range(feat_h):
            for x in range(feat_w):
                center_priors.append([x, y, stride])

    return center_priors


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def activation_function_softmax(src):
    alpha = max(src)
    denominator = 0
    dst = []
    for i in range(len(src)):
        tmp = np.exp(src[i] - alpha)
        dst.append(tmp)
        denominator += tmp
    for i in range(len(src)):
        dst[i] /= denominator
    return dst

def disPred2Bbox(dfl_det, label, score, x, y, stride):
    ct_x = x * stride
    ct_y = y * stride
    dis_pred = []
    reg_max = 7
    for i in range(4):
        dis = 0
        start = i * (reg_max + 1)
        dis_after_sm = activation_function_softmax(dfl_det[start : start + reg_max + 1])
        for j in range(reg_max + 1):
            dis += j * dis_after_sm[j]
        dis *= stride
        dis_pred.append(dis)
    xmin = max(ct_x - dis_pred[0], 0.0)
    ymin = max(ct_y - dis_pred[1], 0.0)
    xmax = min(ct_x + dis_pred[2], 320.0)
    ymax = min(ct_y + dis_pred[3], 320.0)

    return [xmin, ymin, xmax, ymax, score, label]

def decode_infer(pred, center_priors, score_threshold):
    reg_max = 7
    # 模型类别
    num_class = 80
    num_channels = num_class + (reg_max + 1) * 4

    idx = 0
    results = {}
    for x in range(num_class):
        results[x] = []
    pred = pred.flatten()
    for center_prior in center_priors:
        ct_x = center_prior[0]
        ct_y = center_prior[1]
        stride = center_prior[2]
        score = 0.0
        cur_label = 0
        for label in range(num_class):
            if (pred[idx * num_channels + label] > score):
                score = pred[idx * num_channels + label]
                cur_label = label
        if score > score_threshold:
            bbox_pred = pred[idx * num_channels + num_class:]
            results[cur_label].append(disPred2Bbox(bbox_pred, cur_label, score, ct_x, ct_y, stride))
        idx += 1
    return results

def nms(dets, thresh):
    if dets.shape[0] == 0:
        return []
    #x1、y1、x2、y2、以及score赋值
    # （x1、y1）（x2、y2）为box的左上和右下角标
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    #每一个候选框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #order是按照score降序排序的，得到的是排序的本来的索引，不是排完序的原数组
    order = scores.argsort()[::-1]
    # ::-1表示逆序

    temp = []
    while order.size > 0:
        i = order[0]
        temp.append(i)
        #计算当前概率最大矩形框与其他矩形框的相交框的坐标
        # 由于numpy的broadcast机制，得到的是向量
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.minimum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.maximum(y2[i], y2[order[1:]])

        #计算相交框的面积,注意矩形框不相交时w或h算出来会是负数，需要用0代替
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        #计算重叠度IoU
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        #找到重叠度不高于阈值的矩形框索引
        inds = np.where(ovr <= thresh)[0]
        #将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
        order = order[inds + 1]
    return temp

# 图像预处理代码，其中keep_ratio和divisable的取值见模型配置文件nanodet/config/nanodet-plus-m_320.yml中的data-val-keep_ratio
def preprocess(image, input_shape, keep_ratio = True, divisible = 0):
    #image = cv2.imread("000012.jpg")

    height = image.shape[0]  # shape(h,w,c)
    width = image.shape[1]

    dst_shape = input_shape
    if keep_ratio:
        dst_shape = get_minimum_dst_shape(
            (width, height), dst_shape, divisible
        )

    ResizeM = get_resize_matrix((width, height), dst_shape, keep_ratio)
    img = cv2.warpPerspective(image, ResizeM, dsize=tuple(dst_shape))

    img = img.astype(np.float32) / 255
    mean =  [103.53, 116.28, 123.675]
    std =  [57.375, 57.12, 58.395]
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3) / 255
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3) / 255
    img = (img - mean) / std

    W, H = input_shape[1], input_shape[0]
    top = (H - dst_shape[1]) // 2
    bottom = (H - dst_shape[1]) // 2
    if top + bottom + dst_shape[1] < H:
        bottom += 1

    left = (W - dst_shape[0]) // 2
    right = (W - dst_shape[0]) // 2
    if left + right + dst_shape[0] < W:
        right += 1

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value = 0)
    # cv2.imwrite("save.jpg", img)
    img = img.transpose(2, 0, 1)

    effect_roi = [left, top, dst_shape[0], dst_shape[1]]
    return img, effect_roi

# 前向代码
def inference(image, score_threshold = 0.4, nms_threshold = 0.5):
    # 模型输入尺寸
    input_shape = (320, 320)
    # 图像前处理
    img, effect_roi = preprocess(image, input_shape)
    if len(img.shape) == 3:
       img = np.expand_dims(img, 0)
    input_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
    
    # 加载模型，定义输入输出
    interpreter = tf.lite.Interpreter(model_path="models/nanodet_plus_320.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # print(output_details)

    # 设置数据
    interpreter.set_tensor(input_details[0]['index'], input_tensor.numpy())
    # 进行inference
    interpreter.invoke()

    # 获取输出
    dets = interpreter.get_tensor(output_details[0]['index'])
    # print(dets)
    # 后处理
    center_priors = generate_grid_center_priors(input_shape[0], input_shape[1], [8, 16, 32])
    preds = decode_infer(dets, center_priors, score_threshold)
 
    results = []
    # print(preds)
    for key, pred in preds.items():
        idxs = nms(np.array(pred), nms_threshold)
        for idx in idxs:
            #print(pred[idx])
            results.append(pred[idx])

    # bbox结果矫正，由于图像经过处理并且pad过，因此需要映射回原图
    src_w = image.shape[1]
    src_h = image.shape[0]
    dst_w = effect_roi[2]
    dst_h = effect_roi[3]
    width_ratio = float(src_w) / float(dst_w)
    height_ratio = float(src_h) / float(dst_h)
    # print(results)
    for result in results:
        result[0] = int((result[0] - effect_roi[0]) * width_ratio)
        result[1] = int((result[1] - effect_roi[1]) * height_ratio)
        result[2] = int((result[2] - effect_roi[0]) * width_ratio)
        result[3] = int((result[3] - effect_roi[1]) * height_ratio)
    return results

# 画图
def draw(img, all_box):
    for box in all_box:
        x0, y0, x1, y1, score, label = box
        # color = self.cmap(i)[:3]
        color = (_COLORS[label] * 255).astype(np.uint8).tolist()
        text = "{}:{:.1f}%".format(class_names[label], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[label]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font, 0.5, 2)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        cv2.rectangle(
            img,
            (x0, y0 - txt_size[1] - 1),
            (x0 + txt_size[0] + txt_size[1], y0 - 1),
            color,
            -1,
        )
        cv2.putText(img, text, (x0, y0 - 1), font, 0.5, txt_color, thickness=1)
    cv2.imwrite("result.jpg", img)

if __name__ == "__main__":
    image = cv2.imread("000012.jpg")
    results = inference(image)
    draw(image, results)
