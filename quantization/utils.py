from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torchvision.ops import nms as tv_nms


def load_class_names(class_name_file: str) -> List[str]:
    p = Path(class_name_file)
    text = p.read_text(encoding="utf-8")
    l = text.find("[")
    r = text.find("]", l)
    if l < 0 or r < 0:
        raise ValueError(f"{p} 中未找到类别列表（[...]）。")
    names = ast.literal_eval(text[l : r + 1])
    if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
        raise ValueError(f"{p} 的类别列表格式不符合预期。")
    return names


def letterbox(im: np.ndarray, new_shape: Tuple[int, int], color=(114, 114, 114), stride: int = 32) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    # new_shape: (w, h)
    shape = im.shape[:2]  # (h, w)
    w, h = int(new_shape[0]), int(new_shape[1])

    r = min(h / shape[0], w / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # (w, h)
    dw, dh = w - new_unpad[0], h - new_unpad[1]
    dw, dh = dw / 2, dh / 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, (r, r), (dw, dh)


def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def scale_boxes(img1_shape, boxes: torch.Tensor, img0_shape) -> torch.Tensor:
    # img1_shape: (h, w) after letterbox; img0_shape: original (h, w, c)
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2

    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes[:, :4] /= gain
    boxes[:, 0].clamp_(0, img0_shape[1])
    boxes[:, 1].clamp_(0, img0_shape[0])
    boxes[:, 2].clamp_(0, img0_shape[1])
    boxes[:, 3].clamp_(0, img0_shape[0])
    return boxes


def nms_one(
    pred: torch.Tensor,  # (4+nc, n)
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
) -> torch.Tensor:
    # 返回 (n,6): xyxy, conf, cls
    # pred = pred.T  # (n, 4+nc)
    if pred.numel() == 0:
        return pred.new_zeros((0, 6))

    box = xywh2xyxy(pred[:, :4])
    cls = pred[:, 4:]
    conf, j = cls.max(1)
    keep = conf > conf_thres
    box, conf, j = box[keep], conf[keep], j[keep].float()
    if box.numel() == 0:
        return pred.new_zeros((0, 6))

    idx = conf.argsort(descending=True)
    box, conf, j = box[idx], conf[idx], j[idx]

    keep_idx = tv_nms(box, conf, iou_thres)
    keep_idx = keep_idx[:max_det]
    out = torch.cat([box[keep_idx], conf[keep_idx, None], j[keep_idx, None]], dim=1)
    return out


def draw_box(im: np.ndarray, xyxy, label: str, color: Tuple[int, int, int], thickness: int = 2) -> None:
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(im, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
    if not label:
        return
    tf = max(thickness - 1, 1)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, tf)
    outside = y1 - th - 6 >= 0
    y_text = y1 - 4 if outside else y1 + th + 4
    y_box1 = y1 - th - 6 if outside else y1
    y_box2 = y1 if outside else y1 + th + 6
    cv2.rectangle(im, (x1, y_box1), (x1 + tw + 2, y_box2), color, -1, lineType=cv2.LINE_AA)
    cv2.putText(im, label, (x1 + 1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), tf, lineType=cv2.LINE_AA)


def color_bgr(i: int) -> Tuple[int, int, int]:
    base = [
        (56, 56, 255),
        (151, 157, 255),
        (31, 112, 255),
        (29, 178, 255),
        (49, 210, 207),
        (10, 249, 72),
        (23, 204, 146),
        (134, 219, 61),
        (168, 147, 0),
        (187, 99, 44),
        (255, 0, 0),
        (255, 0, 255),
        (255, 255, 0),
    ]
    return base[i % len(base)]


