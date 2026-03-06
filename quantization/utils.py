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


# ================================================src/utils.py文件代码=============================================================
import os
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import numpy as np
import copy
import xml.dom.minidom as minidom
import onnx
import LABELS

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.switch_backend('Agg')

def box_iou(box1, box2):
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    return inter_area/(area1 + area2 - inter_area)

def box_proportion(minbox, maxbox):
    minarea = (minbox[2] - minbox[0]) * (minbox[3] - minbox[1])
    maxarea = (maxbox[2] - maxbox[0]) * (maxbox[3] - maxbox[1])
    
    inter_x1 = max(minbox[0], maxbox[0])
    inter_y1 = max(minbox[1], maxbox[1])
    inter_x2 = min(minbox[2], maxbox[2])
    inter_y2 = min(minbox[3], maxbox[3])

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    if minarea==0 or maxarea==0:
        return 0,0,[0,0,0,0]
    else:
        return inter_area/minarea, inter_area/maxarea, [inter_x1, inter_y1, inter_x2, inter_y2]

def read_xml(img_path, names, names_dataset={}):
    xml_path = 'xml'.join(img_path.replace('/images/','/labels/').rsplit(img_path.split('.')[-1], 1))
    dom_r = minidom.parse(xml_path)
    rot = dom_r.documentElement

    if rot.getElementsByTagName("size"):
        size = rot.getElementsByTagName("size")[0]
        width = int(size.getElementsByTagName('width')[0].firstChild.data)
        height = int(size.getElementsByTagName('height')[0].firstChild.data)
    else:
        print(xml_path,'----------- The xml file have not size attribute -----------')

    rects = []
    for ob in rot.getElementsByTagName("object"):
        cls = ob.getElementsByTagName('name')[0].firstChild.data
        cls = cls.strip()

        if len(names_dataset):
            for m_name in names_dataset:  ## dataset label merge
                if cls in names_dataset[m_name]:
                    cls = m_name
                    break
            
        label = -1   
        if cls in names:
            label = names.index(cls)
        else:
            continue  ## 跳过不在关注列表中的标签

        bndbox_r = ob.getElementsByTagName("bndbox")[0]
        xmin = bndbox_r.getElementsByTagName("xmin")[0].firstChild.data
        ymin = bndbox_r.getElementsByTagName("ymin")[0].firstChild.data
        xmax = bndbox_r.getElementsByTagName("xmax")[0].firstChild.data
        ymax = bndbox_r.getElementsByTagName("ymax")[0].firstChild.data

        xmin = int(xmin) if int(xmin)>0 else 0
        ymin = int(ymin) if int(ymin)>0 else 0
        xmax = int(xmax) if int(xmax)<width else width-1
        ymax = int(ymax) if int(ymax)<height else height-1

        w = xmax - xmin
        h = ymax - ymin
        c_x = xmin + w/2
        c_y = ymin + h/2

        if c_x/width >=1 or c_y/height >=1:
            print(xml_path,'----------- The coordinates exceed the picture boundary -----------')

        if label == -1:
            print(names)
            print(cls, '----------- The label not in classes -----------')
            exit()

        rect = [label, xmin, ymin, xmax, ymax]

        rects.append(rect)

    return rects

def write_xml(image, path, result):
    dom = minidom.getDOMImplementation().createDocument(None,'annotation',None)
    anno = dom.documentElement

    dir_name = os.path.dirname(path).split('/')[-1]
    filename = os.path.basename(path)
    element_f = dom.createElement('folder')
    element_f.appendChild(dom.createTextNode(dir_name))
    anno.appendChild(element_f)
    element_n = dom.createElement('filename')
    element_n.appendChild(dom.createTextNode(filename))
    anno.appendChild(element_n)

    source = dom.createElement('source')
    element_data = dom.createElement('database')
    element_data.appendChild(dom.createTextNode('Unknown'))
    source.appendChild(element_data)
    anno.appendChild(source)

    height, width, depth = image.shape
    size = dom.createElement('size')
    element_w = dom.createElement('width')
    element_w.appendChild(dom.createTextNode(str(width)))
    size.appendChild(element_w)
    element_h = dom.createElement('height')
    element_h.appendChild(dom.createTextNode(str(height)))
    size.appendChild(element_h)
    element_d = dom.createElement('depth')
    element_d.appendChild(dom.createTextNode(str(depth)))
    size.appendChild(element_d)
    anno.appendChild(size)

    segmented = dom.createElement('segmented')
    segmented.appendChild(dom.createTextNode('0'))
    anno.appendChild(segmented)

    for res in result:
        obj = dom.createElement('object')
        element_n = dom.createElement('name')
        element_n.appendChild(dom.createTextNode(res['label']))
        obj.appendChild(element_n)
        element_f = dom.createElement('pose')
        element_f.appendChild(dom.createTextNode('Unspecified'))
        obj.appendChild(element_f)
        element_t = dom.createElement('truncated')
        element_t.appendChild(dom.createTextNode('0'))
        obj.appendChild(element_t)
        element_diff = dom.createElement('difficult')
        element_diff.appendChild(dom.createTextNode('0'))
        obj.appendChild(element_diff)

        bndbox = dom.createElement('bndbox')
        element_xmin = dom.createElement('xmin')
        element_xmin.appendChild(dom.createTextNode(str(res['rect'][0])))
        bndbox.appendChild(element_xmin)
        element_ymin = dom.createElement('ymin')
        element_ymin.appendChild(dom.createTextNode(str(res['rect'][1])))
        bndbox.appendChild(element_ymin)
        element_xmax = dom.createElement('xmax')
        element_xmax.appendChild(dom.createTextNode(str(res['rect'][2])))
        bndbox.appendChild(element_xmax)
        element_ymax = dom.createElement('ymax')
        element_ymax.appendChild(dom.createTextNode(str(res['rect'][3])))
        bndbox.appendChild(element_ymax)
        obj.appendChild(bndbox)
        anno.appendChild(obj)
    
    return dom
    
def filter_label(label, filter_names, names):
    minSize, mediumSize = 30, 100            
    minLable, mediumLabel, largeLabel = [], [], [] 
    
    if len(label) == 0:
        return torch.from_numpy(np.array(label)), torch.from_numpy(np.array(minLable)), torch.from_numpy(np.array(mediumLabel)), torch.from_numpy(np.array(largeLabel))
    
    if len(label[0]) == 5:
        position = 0
        idx = 1
    elif len(label[0]) == 6:
        position = 5
        idx = 0

    if len(filter_names) != 0:
        filter_index = []
        names_list = list(names.values())
        for na in filter_names:
            filter_index.append(names_list.index(na))
            
        i = 0
        for lab in label:
            if lab[position] not in filter_index:
                label = np.delete(label,i,axis=0)
            else:
                i += 1
                
    for lab in label:  
        w = lab[2+idx] - lab[0+idx]
        h = lab[3+idx] - lab[1+idx]
        if w*h < minSize*minSize:
            minLable.append(np.array(lab))
        elif minSize*minSize <= w*h < mediumSize*mediumSize:
            mediumLabel.append(np.array(lab))
        else:
            largeLabel.append(np.array(lab))

    return torch.from_numpy(np.array(label)), torch.from_numpy(np.array(minLable)), torch.from_numpy(np.array(mediumLabel)), torch.from_numpy(np.array(largeLabel))

def merge_label(inputs, names, names_merge, names_merge_dic):
    if len(inputs) == 0:
        return inputs
    
    targets = copy.deepcopy(inputs)
    if len(targets[0]) == 5:
        for target in targets:
            label = names[target[0]]
            for key, value in names_merge.items():
                if label in value:
                    for k, v in names_merge_dic.items():
                        if v == key:
                            target[0] = k
    elif len(targets[0]) == 6:
        for target in targets:
            label = names[int(target[5])]
            for key, value in names_merge.items():
                if label in value:
                    for k, v in names_merge_dic.items():
                        if v == key:
                            target[5] = float(k)
    else:
        print('label size error !')
        exit()
        
    return targets

def np_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = np_iou(labels[:, 1:], detections[:, :4])
    # print(iou, iou.shape)
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct

def get_stats(stats, target, pred, iouv, niou):
    if len(pred) == 0:
        nl = len(target)
        tcls = target[:, 0].tolist() if nl else []  # target class
        if nl:
            stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
    else:
        nl = len(target)
        tcls = target[:, 0].tolist() if nl else []  # target class

        predn = pred.clone()
        if nl:
            correct = process_batch(predn, target, iouv)
        else:
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
        stats.append((correct, pred[:, 4], pred[:, 5], tcls))  # (correct, conf, pcls, tcls)
    
    return stats

def mAP(stats, names_dic):
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy

    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class, nt = ap_per_class(*stats, plot=True, save_dir='./plot', names=names_dic)
        ap25, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map25, map = p.mean(), r.mean(), ap25.mean(), ap.mean()

        mp_weighted = (p * nt).sum() / nt.sum() if nt.sum() > 0 else 0.0
        mr_weighted = (r * nt).sum() / nt.sum() if nt.sum() > 0 else 0.0
        map25_weighted = (ap25 * nt).sum() / nt.sum() if nt.sum() > 0 else 0.0
        map_weighted = (ap * nt).sum() / nt.sum() if nt.sum() > 0 else 0.0

        nt = np.bincount(stats[3].astype(np.int64), minlength=len(names_dic))  # number of targets per class
    else:
        # 没有任何有效统计（例如该尺寸下无目标或无命中），返回全 0，避免上层解包崩溃
        mp = mr = map25 = map = 0.0
        mp_weighted = mr_weighted = map25_weighted = map_weighted = 0.0
        nt = np.zeros(len(names_dic), dtype=np.int64)
        p = np.zeros(0, dtype=np.float32)
        r = np.zeros(0, dtype=np.float32)
        ap25 = np.zeros(0, dtype=np.float32)
        ap = np.zeros(0, dtype=np.float32)
        ap_class = np.zeros(0, dtype=np.int32)

    return mp, mr, map25, map, mp_weighted, mr_weighted, map25_weighted, map_weighted, nt, p, r, ap25, ap, ap_class

def ap_per_class(tp, conf, pred_cls, target_cls, plot=True, save_dir='./plot', names=(), eps=1e-16):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((tp.shape[1], nc, 1000)), np.zeros((tp.shape[1], nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + eps)  # recall curve

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve

            for k in range(tp.shape[1]):
                r[k, ci] = np.interp(-px, -conf[i], recall[:, k], left=0)  # negative x, xp because xp decreases
                p[k, ci] = np.interp(-px, -conf[i], precision[:, k], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    p_mean_iou = np.zeros((nc, tp.shape[1]))
    r_mean_iou = np.zeros((nc, tp.shape[1]))
    for t in (range(tp.shape[1])):
        i = f1[t].mean(0).argmax()
        p_mean_iou[:, t] = p[t, :, i]
        r_mean_iou[:, t] = r[t, :, i]


    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = {i: v for i, v in enumerate(names)}  # to dict
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1[0], Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p[0], Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r[0], Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    p = p_mean_iou.mean(1)
    r = r_mean_iou.mean(1)
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype('int32'), nt

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec

def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close(fig)

def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = py.mean(0)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close(fig)

def onnx_modify_dims(src_path, dst_path, new_input_name, new_output_name, batch, input_h, input_w, onnx_version, ir_version):
    model = onnx.load(src_path)
    onnx.checker.check_model(model)
    print('ir_version:', model.ir_version, 'onnx_version:', model.opset_import[0].version)
    
    ##################################### 修改模型版本 ##########################################
    if onnx_version>0 or ir_version>0:
        model.opset_import[0].version = onnx_version
        model.ir_version = ir_version
        print('ir_version:', model.ir_version, 'onnx_version:', model.opset_import[0].version)

    graph = model.graph
    src_input_name = graph.input[0].name  #外部输入name
    src_output_name = graph.output[0].name #外部输出name
    print('src input name:', src_input_name)
    print('src output name:', src_output_name)
    
    ##################################### 修改输入输出name ##########################################
    if new_input_name:
        for n, node in enumerate(graph.node):
            for k, nd_input in enumerate(node.input):
                if nd_input == src_input_name:
                    graph.node[n].input[k] = new_input_name
        graph.input[0].name = new_input_name
        print('new input name:', new_input_name)
    if new_output_name:
        for n, node in enumerate(graph.node):
            for k, nd_output in enumerate(node.output):
                if nd_output == src_output_name:
                    graph.node[n].output[k] = new_output_name
        graph.output[0].name = new_output_name
        print('new output name:', new_output_name)
    
    ##################################### 修改输入输出shape ##########################################
    for inp in graph.input:
        if inp.name == new_input_name:
            shape = []
            for item in inp.type.tensor_type.shape.dim:
                dim = item.dim_value if item.dim_value!=0 else item.dim_param
                shape.append(dim)
            print('src input shape:', shape)
            
            if input_h==-1 or input_w ==-1:
                new_shape = [batch, shape[1], shape[2], shape[3]]
            else:
                new_shape = [batch, shape[1], input_h, input_w]
            print('new input shape:', new_shape)
            new_inp = onnx.helper.make_tensor_value_info(new_input_name, onnx.TensorProto.FLOAT, new_shape)
            graph.input.remove(inp)
            graph.input.extend([new_inp])

    for oup in graph.output:
        if oup.name == new_output_name:
            shape = []
            for item in oup.type.tensor_type.shape.dim:
                dim = item.dim_value if item.dim_value!=0 else item.dim_param
                shape.append(dim)
            print('src output shape:', shape)
            
            new_shape = shape
            new_shape[0] = batch
            print('new output shape:', new_shape)
            new_oup = onnx.helper.make_tensor_value_info(new_output_name, onnx.TensorProto.FLOAT, new_shape)
            graph.output.remove(oup)
            graph.output.extend([new_oup])

    onnx.save(model, dst_path)

def compute_map_results(Stats, minStats, mediumStats, largeStats,
                 Stats_mg, minStats_mg, mediumStats_mg, largeStats_mg,
                 names_dic, names_dic_mg, data_process, output_excel="map.xlsx"):
    """
    计算和输出mAP结果
    
    参数:
        Stats, minStats, mediumStats, largeStats: 主标签的统计数据
        Stats_mg, minStats_mg, mediumStats_mg, largeStats_mg: 合并标签的统计数据
        names_dic: 标签字典
        names_dic_mg: 合并标签字典
        data_process: 数据处理对象
        output_excel: 输出Excel文件名
    """
    from prettytable import PrettyTable
    import pandas

    # 创建结果表格
    table = PrettyTable()
    table.field_names = ["classes", "images", "objects", "mean_p", "mean_r", "map25", "map25-70", "mean_p_weighted", "mean_r_weighted", "map25-70_weighted"]
    seen = len(data_process.image_path_list())
    rows_data = []
    
    # 处理原始标签结果
    mp, mr, map25, map, mp_weighted, mr_weighted, map25_weighted, map_weighted, nt, p, r, ap25, ap, ap_class = mAP(Stats, names_dic)
    if len(names_dic) > 1 and len(Stats):
        for i, c in enumerate(ap_class):
            table.add_row([names_dic[c], seen, nt[c], round(p[i],4), round(r[i],4), round(ap25[i],4), round(ap[i],4), None, None, None])
            rows_data.append([names_dic[c], seen, nt[c], round(p[i],4), round(r[i],4), round(ap25[i],4), round(ap[i],4), None, None, None]) 
    
    table.add_row(["-------", "-------", "-------", "-------", "-------", "-------", "-------", "-------", "-------", "-------"])  
    
    # 添加总体结果
    table.add_row(["All", seen, nt.sum(), round(mp,4), round(mr,4), round(map25,4), round(map,4), round(mp_weighted,4), round(mr_weighted,4), round(map_weighted,4)])
    rows_data.append(["All", seen, nt.sum(), round(mp,4), round(mr,4), round(map25,4), round(map,4), round(mp_weighted,4), round(mr_weighted,4), round(map_weighted,4)])
    
    # 添加小目标结果
    mp, mr, map25, map, mp_weighted, mr_weighted, map25_weighted, map_weighted, nt, _, _, _, _, _ = mAP(minStats, names_dic)
    table.add_row(["minSize", seen, nt.sum(), round(mp,4), round(mr,4), round(map25,4), round(map,4), round(mp_weighted,4), round(mr_weighted,4), round(map_weighted,4)])
    # table.add_row(["minSize_weighted", seen, nt.sum(), round(mp_weighted,4), round(mr_weighted,4), round(map25_weighted,4), round(map_weighted,4)])
    rows_data.append(["minSize", seen, nt.sum(), round(mp,4), round(mr,4), round(map25,4), round(map,4), round(mp_weighted,4), round(mr_weighted,4), round(map_weighted,4)])
    # rows_data.append(["minSize_weighted", seen, nt.sum(), round(mp_weighted,4), round(mr_weighted,4), round(map25_weighted,4), round(map_weighted,4)])
    
    # 添加中目标结果
    mp, mr, map25, map, mp_weighted, mr_weighted, map25_weighted, map_weighted, nt, _, _, _, _, _ = mAP(mediumStats, names_dic)
    table.add_row(["mediumSize", seen, nt.sum(), round(mp,4), round(mr,4), round(map25,4), round(map,4), round(mp_weighted,4), round(mr_weighted,4), round(map_weighted,4)])
    # table.add_row(["mediumSize_weighted", seen, nt.sum(), round(mp_weighted,4), round(mr_weighted,4), round(map25_weighted,4), round(map_weighted,4)])
    rows_data.append(["mediumSize", seen, nt.sum(), round(mp,4), round(mr,4), round(map25,4), round(map,4), round(mp_weighted,4), round(mr_weighted,4), round(map_weighted,4)])
    # rows_data.append(["mediumSize_weighted", seen, nt.sum(), round(mp_weighted,4), round(mr_weighted,4), round(map25_weighted,4), round(map_weighted,4)])

    
    # 添加大目标结果
    mp, mr, map25, map, mp_weighted, mr_weighted, map25_weighted, map_weighted, nt, _, _, _, _, _ = mAP(largeStats, names_dic)
    table.add_row(["largeSize", seen, nt.sum(), round(mp,4), round(mr,4), round(map25,4), round(map,4), round(mp_weighted,4), round(mr_weighted,4), round(map_weighted,4)])
    # table.add_row(["largeSize_weighted", seen, nt.sum(), round(mp_weighted,4), round(mr_weighted,4), round(map25_weighted,4), round(map_weighted,4)])
    rows_data.append(["largeSize", seen, nt.sum(), round(mp,4), round(mr,4), round(map25,4), round(map,4), round(mp_weighted,4), round(mr_weighted,4), round(map_weighted,4)])
    # rows_data.append(["largeSize_weighted", seen, nt.sum(), round(mp_weighted,4), round(mr_weighted,4), round(map25_weighted,4), round(map_weighted,4)])
    
    table.add_row(["-------", "-------", "-------", "-------", "-------", "-------", "-------", "-------", "-------", "-------"])

    # 添加合并标签总体结果
    mp, mr, map25, map, mp_weighted, mr_weighted, map25_weighted, map_weighted, nt, p, r, ap25, ap, ap_class = mAP(Stats_mg, names_dic_mg) 
    if len(names_dic_mg) > 1 and len(Stats_mg):
        for i, c in enumerate(ap_class):
            table.add_row([names_dic_mg[c], seen, nt[c], round(p[i],4), round(r[i],4), round(ap25[i],4), round(ap[i],4), None, None, None])
            rows_data.append([names_dic_mg[c], seen, nt[c], round(p[i],4), round(r[i],4), round(ap25[i],4), round(ap[i],4), None, None, None]) 


    table.add_row(["-------", "-------", "-------", "-------", "-------", "-------", "-------", "-------", "-------", "-------"]) 
    table.add_row(["All_mg", seen, nt.sum(), round(mp,4), round(mr,4), round(map25,4), round(map,4), round(mp_weighted,4), round(mr_weighted,4), round(map_weighted,4)])
    # table.add_row(["All_mg_weighted", seen, nt.sum(), round(mp_weighted,4), round(mr_weighted,4), round(map25_weighted,4), round(map_weighted,4)])
    rows_data.append(["All_mg", seen, nt.sum(), round(mp,4), round(mr,4), round(map25,4), round(map,4), round(mp_weighted,4), round(mr_weighted,4), round(map_weighted,4)])
    # rows_data.append(["All_mg_weighted", seen, nt.sum(), round(mp_weighted,4), round(mr_weighted,4), round(map25_weighted,4), round(map_weighted,4)])
    
    # 添加合并标签小目标结果
    mp, mr, map25, map, mp_weighted, mr_weighted, map25_weighted, map_weighted, nt, _, _, _, _, _ = mAP(minStats_mg, names_dic_mg)
    table.add_row(["minSize_mg", seen, nt.sum(), round(mp,4), round(mr,4), round(map25,4), round(map,4), round(mp_weighted,4), round(mr_weighted,4), round(map_weighted,4)])
    # table.add_row(["minSize_mg_weighted", seen, nt.sum(), round(mp_weighted,4), round(mr_weighted,4), round(map25_weighted,4), round(map_weighted,4)])
    rows_data.append(["minSize_mg", seen, nt.sum(), round(mp,4), round(mr,4), round(map25,4), round(map,4), round(mp_weighted,4), round(mr_weighted,4), round(map_weighted,4)])
    # rows_data.append(["minSize_mg_weighted", seen, nt.sum(), round(mp_weighted,4), round(mr_weighted,4), round(map25_weighted,4), round(map_weighted,4)])
    
    # 添加合并标签中目标结果
    mp, mr, map25, map, mp_weighted, mr_weighted, map25_weighted, map_weighted, nt, _, _, _, _, _ = mAP(mediumStats_mg, names_dic_mg)
    table.add_row(["mediumSize_mg", seen, nt.sum(), round(mp,4), round(mr,4), round(map25,4), round(map,4), round(mp_weighted,4), round(mr_weighted,4), round(map_weighted,4)])
    # table.add_row(["mediumSize_mg_weighted", seen, nt.sum(), round(mp_weighted,4), round(mr_weighted,4), round(map25_weighted,4), round(map_weighted,4)])
    rows_data.append(["mediumSize_mg", seen, nt.sum(), round(mp,4), round(mr,4), round(map25,4), round(map,4), round(mp_weighted,4), round(mr_weighted,4), round(map_weighted,4)])
    # rows_data.append(["mediumSize_mg_weighted", seen, nt.sum(), round(mp_weighted,4), round(mr_weighted,4), round(map25_weighted,4), round(map_weighted,4)])
    
    # 添加合并标签大目标结果
    mp, mr, map25, map, mp_weighted, mr_weighted, map25_weighted, map_weighted, nt, _, _, _, _, _ = mAP(largeStats_mg, names_dic_mg)
    table.add_row(["largeSize_mg", seen, nt.sum(), round(mp,4), round(mr,4), round(map25,4), round(map,4), round(mp_weighted,4), round(mr_weighted,4), round(map_weighted,4)])
    # table.add_row(["largeSize_mg_weighted", seen, nt.sum(), round(mp_weighted,4), round(mr_weighted,4), round(map25_weighted,4), round(map_weighted,4)])
    rows_data.append(["largeSize_mg", seen, nt.sum(), round(mp,4), round(mr,4), round(map25,4), round(map,4), round(mp_weighted,4), round(mr_weighted,4), round(map_weighted,4)])
    # rows_data.append(["largeSize_mg_weighted", seen, nt.sum(), round(mp_weighted,4), round(mr_weighted,4), round(map25_weighted,4), round(map_weighted,4)])
    
    # 输出结果到Excel
    df = pandas.DataFrame(rows_data, columns=["classes", "images", "objects", "mean_p", "mean_r", "map25", "map25-70", "mean_p_weighted", "mean_r_weighted", "map25-70_weighted"])
    df.to_excel(output_excel, index=False)
    
    # 输出表格
    print(table)
    
    # 返回主要指标
    return {
        "map25": map25,
        "map": map,
        "mean_precision": mp,
        "mean_recall": mr
    }

def read_infer_txt(filepath, thin_coarse_index,is_model_label_thin,ignore_labels):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    Pred,Pred_mg = [],[]
    for line in lines:
        line = line.split()
        if line[0] == '':
            continue
        line = [eval(i) for i in line]
        rect = [int(line[1]), int(line[2]), int((line[3])),int((line[4]))]
        label = int(line[0])
        conf = float(line[5])
        
        if label not in ignore_labels :
            if is_model_label_thin:
                Pred.append([rect[0], rect[1], rect[2], rect[3], conf, label])
                label_mg = thin_coarse_index[label] if label in thin_coarse_index else -1
                if label_mg!=-1:
                    Pred_mg.append([rect[0], rect[1], rect[2], rect[3], conf, label_mg])
            else:
                Pred.append([rect[0], rect[1], rect[2], rect[3], conf, label])
                Pred_mg.append([rect[0], rect[1], rect[2], rect[3], conf, label])
    
    return Pred,Pred_mg  ## xmin,ymin,xmax,ymax,conf,label




def read_gt_xml(xml_path, label_index, thin_coarse,thin_coarse_index, is_model_label_thin=True):
    """gt xml 里的标签细标签，转换为模型训练时的标签索引,支持模型标签为细标签或粗标签两种情况
       is_model_label_thin: True-模型标签是细标签 False-模型标签是粗标签
    """
    dom_r = minidom.parse(xml_path)
    rot = dom_r.documentElement

    if rot.getElementsByTagName("size"):
        size = rot.getElementsByTagName("size")[0]
        width = int(size.getElementsByTagName('width')[0].firstChild.data)
        height = int(size.getElementsByTagName('height')[0].firstChild.data)
    else:
        print(xml_path,'----------- The xml file have not size attribute -----------')

    rects = []
    rects_mg = []
    for ob in rot.getElementsByTagName("object"):
        cls = ob.getElementsByTagName('name')[0].firstChild.data
        cls = cls.strip()
                
        label = -1
        label_mg = -1

        if is_model_label_thin:   
            if cls in label_index:
                label = label_index[cls]
                label_mg = thin_coarse_index[label] if label in thin_coarse_index else -1       
        else:
            if cls in thin_coarse:
                coarse_cls = thin_coarse[cls]
                if coarse_cls in label_index:
                    label = label_index[coarse_cls]
                    label_mg = label
       
        bndbox_r = ob.getElementsByTagName("bndbox")[0]
        xmin = bndbox_r.getElementsByTagName("xmin")[0].firstChild.data
        ymin = bndbox_r.getElementsByTagName("ymin")[0].firstChild.data
        xmax = bndbox_r.getElementsByTagName("xmax")[0].firstChild.data
        ymax = bndbox_r.getElementsByTagName("ymax")[0].firstChild.data

        xmin = int(xmin) if int(xmin)>0 else 0
        ymin = int(ymin) if int(ymin)>0 else 0
        xmax = int(xmax) if int(xmax)<width else width-1
        ymax = int(ymax) if int(ymax)<height else height-1

        w = xmax - xmin
        h = ymax - ymin
        c_x = xmin + w/2
        c_y = ymin + h/2

        if c_x/width >=1 or c_y/height >=1:
            print(xml_path,'----------- The coordinates exceed the picture boundary -----------')

        if label != -1 and label_mg != -1:
            rect = [label, xmin, ymin, xmax, ymax]
            rects.append(rect)
            rect_mg = [label_mg, xmin, ymin, xmax, ymax]
            rects_mg.append(rect_mg)
            # rects_mg.append(rect_mg)
            # exit()
        else:
            # print(cls, '----------- The label not in classes -----------')
            pass



    return rects,rects_mg


def mAP_new(stats, names_dic):
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy

    if len(stats) and stats[0].any():
        tp, fp, p, r,mp,mr, f1, ap, ap_class, nt = ap_per_class_new(*stats, plot=False, save_dir='./plot', names=names_dic)
        ### p(iou:0.5) r(iou:0.5), mp(iou:0.25:0.95), mr(iou:0.25:0.95)
        ap25, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95

        map25, map = ap25.mean(), ap.mean()
        mp_weighted = (mp * nt).sum() / nt.sum() if nt.sum() > 0 else 0.0
        mr_weighted = (mr * nt).sum() / nt.sum() if nt.sum() > 0 else 0.0
        map25_weighted = (ap25 * nt).sum() / nt.sum() if nt.sum() > 0 else 0.0
        map_weighted = (ap * nt).sum() / nt.sum() if nt.sum() > 0 else 0.0
        nt = np.bincount(stats[3].astype(np.int64), minlength=len(names_dic))  # number of targets per class
    else:
        nt = torch.zeros(1)
        print('WARNING: No detections, no mAP. Check your data and model.')
        exit(0)

    return p,r,mp, mr, map25, map ,mp_weighted,mr_weighted,map25_weighted,map_weighted

def ap_per_class_new(tp, conf, pred_cls, target_cls, plot=True, save_dir='./plot', names=(), eps=1e-16):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)  ##从大到小排序
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)  ##真实的类可能漏
    nc = unique_classes.shape[0]  # number of classes, number of detections
    # nc = len(names) # number of classes, number of detections
    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((tp.shape[1], nc, 1000)), np.zeros((tp.shape[1], nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + eps)  # recall curve

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve

            for k in range(tp.shape[1]):
                r[k, ci] = np.interp(-px, -conf[i], recall[:, k], left=0)  # negative x, xp because xp decreases
                p[k, ci] = np.interp(-px, -conf[i], precision[:, k], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    p_mean_iou = np.zeros((nc, tp.shape[1]))
    r_mean_iou = np.zeros((nc, tp.shape[1]))
    for t in (range(tp.shape[1])):
        i = f1[t].mean(0).argmax()
        # bests_conf.append(conf[len(conf)-1-i]) 
        # p_mean_iou[:, t] = p[t, :, i]
        # r_mean_iou[:, t] = r[t, :, i]
        p_mean_iou[:, t] = p[t, :, 0]
        r_mean_iou[:, t] = r[t, :, 0]


    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = {i: v for i, v in enumerate(names)}  # to dict
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1[0], Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p[0], Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r[0], Path(save_dir) / 'R_curve.png', names, ylabel='Recall')
    p = p_mean_iou[:,0]
    r = r_mean_iou[:,0]
    mp = p_mean_iou.mean(1)
    mr = r_mean_iou.mean(1)
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r,mp,mr, f1, ap, unique_classes.astype('int32'), nt


def merge_label_new(thin_coarse_index,Pred,is_model_label_thin):
    Pred_mg = []
    if is_model_label_thin:
        for pred in Pred:
            label = int(pred[5])
            new_label = thin_coarse_index[label] if label in thin_coarse_index else -1
            if new_label != -1:
                pred[5] = new_label
                Pred_mg.append(pred)
    else:
        Pred_mg = Pred.copy()
    return Pred_mg

