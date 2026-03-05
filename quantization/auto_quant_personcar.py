#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PersonCarAnimal ONNX model accuracy-based auto quantization with AMCT.

Workflow:
1. create_quant_config
2. accuracy_based_auto_calibration

Evaluator design:
- calibration(): use /workspace/datasets/person_car_animal-1101
- evaluate(): compute detection mAP (IoU 0.25:0.7) on /workspace/AlgoServerScript/datasets/person_car
"""

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import torch
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

import amct_onnx as amct
from amct_onnx.common.auto_calibration import AutoCalibrationEvaluatorBase
from utils import letterbox, nms_one, scale_boxes  # noqa: E402


ALGO_SERVER_ROOT = "/workspace/AlgoServerScript"
if ALGO_SERVER_ROOT not in sys.path:
    sys.path.insert(0, ALGO_SERVER_ROOT)

import LABELS  # noqa: E402
from src import utils  # noqa: E402


def patch_amct_auto_calibration_helper():
    """
    Patch AMCT auto calibration helper for multi-batch dump files.

    AMCT 8.5.0 may collect all dump files across batches for one layer and
    treat them as multiple inputs, which can cause:
    IndexError: list assignment index out of range
    """
    try:
        from amct_onnx.utils.auto_calibration_helper import AutoCalibrationHelper
    except Exception:
        return

    if getattr(AutoCalibrationHelper, "_fm_file_patch_applied", False):
        return

    original_find_fm_file_path = AutoCalibrationHelper.find_fm_file_path

    def _patched_find_fm_file_path(self, layer_name):
        layer_prefix = layer_name.replace("/", "_")
        pattern = re.compile(
            rf"^{re.escape(layer_prefix)}_act_calibration_layer_dump(\d+)_(\d+)\.bin$"
        )
        log_dir = os.path.realpath(self.amct_log_dir)

        # Keep one dump file per input index, prefer the smallest batch index.
        per_input_file = {}
        for file_name in os.listdir(log_dir):
            match = pattern.match(file_name)
            if not match:
                continue
            input_idx = int(match.group(1))
            batch_idx = int(match.group(2))
            file_path = os.path.join(log_dir, file_name)
            if input_idx not in per_input_file or batch_idx < per_input_file[input_idx][0]:
                per_input_file[input_idx] = (batch_idx, file_path)

        if per_input_file:
            return [per_input_file[idx][1] for idx in sorted(per_input_file.keys())]

        # Fallback to AMCT original logic.
        return original_find_fm_file_path(self, layer_name)

    AutoCalibrationHelper.find_fm_file_path = _patched_find_fm_file_path
    AutoCalibrationHelper._fm_file_patch_applied = True


def progress_bar(iterable, total: int, desc: str):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=False)


def collect_images(root: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP", ".TIFF"}
    root_path = Path(root)
    if not root_path.exists():
        return []
    images = [str(p) for p in root_path.rglob("*") if p.suffix in exts]
    images.sort()
    return images


def create_session(model_file: str) -> ort.InferenceSession:
    providers = ["CUDAExecutionProvider","CPUExecutionProvider"]
    amct_so = getattr(amct, "AMCT_SO", None)
    if amct_so is None:
        return ort.InferenceSession(model_file, providers=providers)

    try:
        return ort.InferenceSession(model_file, amct_so, providers=providers)
    except TypeError:
        return ort.InferenceSession(model_file, sess_options=amct_so, providers=providers)


def preprocess_image(im0_bgr: np.ndarray, input_wh: Tuple[int, int], stride: int) -> np.ndarray:
    im, _, _ = letterbox(im0_bgr, input_wh, stride=stride)
    im = im.transpose((2, 0, 1))[::-1]  # BGR -> RGB, HWC -> CHW
    return np.ascontiguousarray(im, dtype=np.float32) / 255.0


def normalize_output(output_value: np.ndarray) -> np.ndarray:
    out = np.asarray(output_value)
    if out.ndim == 4 and out.shape[1] == 1:
        out = out[:, 0]
    elif out.ndim == 2:
        out = out[None, ...]
    if out.ndim != 3:
        raise RuntimeError(f"Unexpected model output shape: {out.shape}")
    return out


class PersonCarAutoCalibrationEvaluator(AutoCalibrationEvaluatorBase):
    def __init__(
        self,
        calibration_dir: str,
        eval_data_dir: str,
        batch_num: int,
        batch_size: int,
        calibration_iters: int,
        calibration_samples: int,
        expected_metric_loss: float,
        input_width: int,
        input_height: int,
        conf_thres: float,
        iou_thres: float,
        max_det: int,
        eval_max_images: int,
    ):
        super().__init__()
        self.batch_num = int(batch_num)
        self.batch_size = int(batch_size)
        self.calibration_iters = int(calibration_iters)
        self.calibration_samples = int(calibration_samples)
        self.expected_metric_loss = float(expected_metric_loss)
        self.input_wh = (int(input_width), int(input_height))
        self.input_hw = (int(input_height), int(input_width))
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.max_det = int(max_det)
        self.eval_max_images = int(eval_max_images)
        self.stride = 32

        self.names = LABELS.PersonCarAnimal
        self.names_merge = LABELS.PersonCarAnimal_merge
        self.names_dataset = LABELS.PersonCarAnimal_dataset_merge
        self.names_dic = {k: v for k, v in enumerate(self.names)}
        self.names_dic_mg = {k: v for k, v in enumerate(self.names_merge.keys())}

        self.filter_names: List[str] = []
        self.filter_names_mg: List[str] = []
        self.iouv = torch.linspace(0.25, 0.7, 10)
        self.niou = self.iouv.numel()

        self.calibration_images = collect_images(calibration_dir)
        eval_images_dir = Path(eval_data_dir) / "images"
        self.eval_images = collect_images(str(eval_images_dir if eval_images_dir.exists() else eval_data_dir))
        if self.eval_max_images > 0:
            self.eval_images = self.eval_images[: self.eval_max_images]

    @staticmethod
    def _run_batch(session: ort.InferenceSession, batch_input: np.ndarray) -> np.ndarray:
        input_name = session.get_inputs()[0].name
        output_value = session.run(None, {input_name: batch_input})[0]
        return normalize_output(output_value)

    def calibration(self, model_file: str):
        if not self.calibration_images:
            raise RuntimeError("Calibration images are not found.")

        if self.calibration_samples > 0:
            iterations = int(math.ceil(float(self.calibration_samples) / float(self.batch_size)))
        elif self.calibration_iters > 0:
            iterations = self.calibration_iters
        else:
            iterations = self.batch_num
        iterations = max(iterations, self.batch_num)

        session = create_session(model_file)
        image_count = len(self.calibration_images)
        for idx in progress_bar(range(iterations), total=iterations, desc="校准进度"):
            start = (idx * self.batch_size) % image_count
            batch_paths = [self.calibration_images[(start + k) % image_count] for k in range(self.batch_size)]
            batch_data = []
            for img_path in batch_paths:
                im0 = cv2.imread(img_path)
                if im0 is None:
                    raise RuntimeError(f"Failed to read calibration image: {img_path}")
                batch_data.append(preprocess_image(im0, self.input_wh, self.stride))
            batch_input = np.stack(batch_data, axis=0)
            _ = self._run_batch(session, batch_input)

    def evaluate(self, model_file: str):
        if not self.eval_images:
            raise RuntimeError("No evaluation images found.")

        session = create_session(model_file)

        stats = []
        stats_mg = []
        min_stats = []
        medium_stats = []
        large_stats = []
        min_stats_mg = []
        medium_stats_mg = []
        large_stats_mg = []

        batch_inputs = []
        batch_img_paths = []
        batch_raw_images = []

        def flush_batch():
            nonlocal batch_inputs, batch_img_paths, batch_raw_images
            nonlocal stats, stats_mg, min_stats, medium_stats, large_stats
            nonlocal min_stats_mg, medium_stats_mg, large_stats_mg

            if not batch_inputs:
                return

            batch_input = np.stack(batch_inputs, axis=0)
            output_np = self._run_batch(session, batch_input)

            for b in range(output_np.shape[0]):
                img_path = batch_img_paths[b]
                im0 = batch_raw_images[b]

                target_list = utils.read_xml(img_path, self.names, names_dataset=self.names_dataset)
                target, min_target, medium_target, large_target = utils.filter_label(
                    target_list, self.filter_names, self.names_dic
                )
                target_mg = utils.merge_label(target_list, self.names, self.names_merge, self.names_dic_mg)
                target_mg, min_target_mg, medium_target_mg, large_target_mg = utils.filter_label(
                    target_mg, self.filter_names_mg, self.names_dic_mg
                )

                pred_raw = torch.from_numpy(output_np[b]).float()
                det = nms_one(
                    pred_raw,
                    conf_thres=self.conf_thres,
                    iou_thres=self.iou_thres,
                    max_det=self.max_det,
                )

                if len(det):
                    det[:, :4] = scale_boxes(self.input_hw, det[:, :4], im0.shape).round()
                    pred = det.cpu().numpy()
                else:
                    pred = np.array([])

                pred, min_pred, medium_pred, large_pred = utils.filter_label(pred, self.filter_names, self.names_dic)
                pred_mg = utils.merge_label(pred, self.names, self.names_merge, self.names_dic_mg)
                pred_mg, min_pred_mg, medium_pred_mg, large_pred_mg = utils.filter_label(
                    pred_mg, self.filter_names_mg, self.names_dic_mg
                )

                stats = utils.get_stats(stats, target, pred, self.iouv, self.niou)
                min_stats = utils.get_stats(min_stats, min_target, min_pred, self.iouv, self.niou)
                medium_stats = utils.get_stats(medium_stats, medium_target, medium_pred, self.iouv, self.niou)
                large_stats = utils.get_stats(large_stats, large_target, large_pred, self.iouv, self.niou)

                stats_mg = utils.get_stats(stats_mg, target_mg, pred_mg, self.iouv, self.niou)
                min_stats_mg = utils.get_stats(min_stats_mg, min_target_mg, min_pred_mg, self.iouv, self.niou)
                medium_stats_mg = utils.get_stats(
                    medium_stats_mg, medium_target_mg, medium_pred_mg, self.iouv, self.niou
                )
                large_stats_mg = utils.get_stats(large_stats_mg, large_target_mg, large_pred_mg, self.iouv, self.niou)

            batch_inputs = []
            batch_img_paths = []
            batch_raw_images = []

        for img_path in progress_bar(self.eval_images, total=len(self.eval_images), desc="评估进度"):
            im0 = cv2.imread(img_path)
            if im0 is None:
                continue
            batch_inputs.append(preprocess_image(im0, self.input_wh, self.stride))
            batch_img_paths.append(img_path)
            batch_raw_images.append(im0)

            if len(batch_inputs) >= self.batch_size:
                flush_batch()

        flush_batch()

        # Use map25-70 as the evaluate metric.
        _, _, _, map_2570, _, _, _, _, _, _, _, _, _, _ = utils.mAP(stats, self.names_dic)
        return float(map_2570)

    def metric_eval(self, original_metric: float, new_metric: float):
        loss = float(original_metric - new_metric)
        return (loss <= self.expected_metric_loss), loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AMCT ONNX auto quantization for PersonCarAnimal model")
    parser.add_argument(
        "--model",
        default="/workspace/models/PersonCarAnimal_od-v3-x-bestp-d4-416-768_20251203.onnx",
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--calibration-dir",
        default="/workspace/datasets/person_car_animal-1101",
        help="Calibration image directory",
    )
    parser.add_argument(
        "--eval-data-dir",
        default="/workspace/AlgoServerScript/datasets/person_car",
        help="Evaluation dataset root (contains images/ and labels/)",
    )
    parser.add_argument(
        "--output-dir",
        default="/workspace/quantization/out/auto_quant_personcar_result",
        help="Output directory",
    )
    parser.add_argument("--batch-num", type=int, default=4, help="Calibration batch number")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for calibration/evaluation")
    parser.add_argument(
        "--calib-iters",
        type=int,
        default=0,
        help="Actual calibration forward iterations; 0 means use batch-num",
    )
    parser.add_argument(
        "--calib-samples",
        type=int,
        default=1101,
        help="Target calibration image count; if >0, overrides calib-iters",
    )
    parser.add_argument(
        "--expected-metric-loss",
        type=float,
        default=0.001,
        help="Allowed mAP loss (absolute value)",
    )
    parser.add_argument("--input-width", type=int, default=768, help="Model input width")
    parser.add_argument("--input-height", type=int, default=416, help="Model input height")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Score threshold for NMS")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--max-det", type=int, default=300, help="Max detections per image")
    parser.add_argument(
        "--eval-max-images",
        type=int,
        default=0,
        help="Max eval images. Set 0 to evaluate all images.",
    )
    parser.add_argument("--strategy", default="BinarySearch", help="Auto calibration strategy")
    parser.add_argument("--sensitivity", default="CosineSimilarity", help="Layer sensitivity method")
    parser.add_argument("--skip-layers", default="", help="Comma-separated layer names to skip quantization")

    activation_group = parser.add_mutually_exclusive_group()
    activation_group.add_argument("--activation-offset", action="store_true", help="Enable activation offset")
    activation_group.add_argument("--no-activation-offset", action="store_true", help="Disable activation offset")

    args = parser.parse_args()
    if not args.activation_offset and not args.no_activation_offset:
        args.activation_offset = True
    elif args.no_activation_offset:
        args.activation_offset = False
    return args


def main():
    args = parse_args()
    patch_amct_auto_calibration_helper()

    if args.batch_num <= 0:
        raise RuntimeError(f"batch-num must be > 0, got {args.batch_num}")
    if args.batch_size <= 0:
        raise RuntimeError(f"batch-size must be > 0, got {args.batch_size}")
    if args.calib_iters < 0:
        raise RuntimeError(f"calib-iters must be >= 0, got {args.calib_iters}")
    if args.calib_samples < 0:
        raise RuntimeError(f"calib-samples must be >= 0, got {args.calib_samples}")

    model_file = os.path.realpath(args.model)
    if not os.path.isfile(model_file):
        raise RuntimeError(f"Model not found: {model_file}")

    calibration_dir = os.path.realpath(args.calibration_dir)
    if not os.path.isdir(calibration_dir):
        raise RuntimeError(f"Calibration directory not found: {calibration_dir}")

    eval_data_dir = os.path.realpath(args.eval_data_dir)
    if not os.path.isdir(eval_data_dir):
        raise RuntimeError(f"Evaluation data directory not found: {eval_data_dir}")

    output_dir = os.path.realpath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    config_file = os.path.join(output_dir, "config.json")
    record_file = os.path.join(output_dir, "scale_offset_record.txt")
    save_prefix = os.path.join(output_dir, "personcar_model")
    skip_layers = [s.strip() for s in args.skip_layers.split(",") if s.strip()]

    amct.create_quant_config(
        config_file=config_file,
        model_file=model_file,
        skip_layers=skip_layers,
        batch_num=args.batch_num,
        activation_offset=args.activation_offset,
    )

    evaluator = PersonCarAutoCalibrationEvaluator(
        calibration_dir=calibration_dir,
        eval_data_dir=eval_data_dir,
        batch_num=args.batch_num,
        batch_size=args.batch_size,
        calibration_iters=args.calib_iters,
        calibration_samples=args.calib_samples,
        expected_metric_loss=args.expected_metric_loss,
        input_width=args.input_width,
        input_height=args.input_height,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        max_det=args.max_det,
        eval_max_images=args.eval_max_images,
    )

    from incremental_strategy import IncrementalStrategy # 导入自定义增量策略

    # 调用AMCT量化
    # 使用增量策略（适合精度差一点点的场景）
    # step_ratio=0.05 表示每次还原5%的层，在速度和精度之间取得平衡
    incremental_strategy = IncrementalStrategy(step_ratio=0.2, min_step=1)

    amct.accuracy_based_auto_calibration(
        model_file=model_file,
        model_evaluator=evaluator,
        config_file=config_file,
        record_file=record_file,
        save_dir=save_prefix,
        # strategy=args.strategy,
        strategy=incremental_strategy,
        sensitivity=args.sensitivity,
    )

    print("Auto quantization finished.")
    print(f"Output directory: {output_dir}")
    print(f"Config file: {config_file}")
    print(f"Record file: {record_file}")
    print(f"Fake quant model: {save_prefix}_fake_quant_model.onnx")
    print(f"Deploy model: {save_prefix}_deploy_model.onnx")


if __name__ == "__main__":
    main()
