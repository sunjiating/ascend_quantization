#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

import amct_onnx as amct
from amct_onnx.common.auto_calibration import AutoCalibrationEvaluatorBase


ALGO_SERVER_ROOT = "/workspace/AlgoServerScript"
if ALGO_SERVER_ROOT not in sys.path:
    sys.path.insert(0, ALGO_SERVER_ROOT)

import LABELS  # noqa: E402
from src import utils  # noqa: E402


def patch_amct_auto_calibration_helper():
    try:
        from amct_onnx.utils.auto_calibration_helper import AutoCalibrationHelper
    except Exception:
        return

    if getattr(AutoCalibrationHelper, "_fm_file_patch_applied", False):
        return

    original_find_fm_file_path = AutoCalibrationHelper.find_fm_file_path
    original_generate_single_model = AutoCalibrationHelper.generate_single_model

    def _patched_find_fm_file_path(self, layer_name):
        layer_prefix = layer_name.replace("/", "_")
        pattern = re.compile(rf"^{re.escape(layer_prefix)}_act_calibration_layer_dump(\d+)_(\d+)\.bin$")
        log_dir = os.path.realpath(self.amct_log_dir)

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

        return original_find_fm_file_path(self, layer_name)

    def _patched_generate_single_model(self, layer_name, input_file_list):
        # Safety net for AMCT versions where duplicate dump files cause
        # input_file_list length > real node inputs.
        try:
            graph = self.original_graph
            object_node = None
            for node in graph.nodes:
                if node.name == layer_name:
                    object_node = node
                    break
            if object_node is not None and hasattr(object_node, "input"):
                real_inputs = len(object_node.input)
                if real_inputs > 0 and len(input_file_list) > real_inputs:
                    input_file_list = input_file_list[:real_inputs]
        except Exception:
            pass
        return original_generate_single_model(self, layer_name, input_file_list)

    AutoCalibrationHelper.find_fm_file_path = _patched_find_fm_file_path
    AutoCalibrationHelper.generate_single_model = _patched_generate_single_model
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


def collect_eval_images_recursive(eval_data_dir: str) -> List[str]:
    # Support recursive dataset layout like:
    # smoke_phone/{正样本,负样本,...}/images/*.jpg
    all_images = collect_images(eval_data_dir)
    normalized = [p.replace("\\", "/") for p in all_images]
    images_with_standard_path = [p for p in normalized if "/images/" in p]
    if images_with_standard_path:
        return images_with_standard_path
    return normalized


def create_session(model_file: str) -> ort.InferenceSession:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    amct_so = getattr(amct, "AMCT_SO", None)
    if amct_so is None:
        return ort.InferenceSession(model_file, providers=providers)
    try:
        return ort.InferenceSession(model_file, amct_so, providers=providers)
    except TypeError:
        return ort.InferenceSession(model_file, sess_options=amct_so, providers=providers)


def normalize_output(output_value: np.ndarray) -> np.ndarray:
    out = np.asarray(output_value)
    if out.ndim == 4 and out.shape[1] == 1:
        out = out[:, 0]
    elif out.ndim == 2:
        out = out[None, ...]
    if out.ndim != 3:
        raise RuntimeError(f"Unexpected model output shape: {out.shape}")
    return out


def load_preprocessed_calibration(npy_file: str) -> np.ndarray:
    if not os.path.isfile(npy_file):
        raise RuntimeError(f"Calibration npy not found: {npy_file}")
    data = np.load(npy_file)
    if data.ndim == 3:
        data = data[None, ...]
    if data.ndim != 4:
        raise RuntimeError(f"Calibration npy must be 4D, got {data.shape}")

    # Accept NHWC and convert to NCHW.
    if data.shape[-1] == 3 and data.shape[1] != 3:
        data = np.transpose(data, (0, 3, 1, 2))

    if data.shape[1] != 3:
        raise RuntimeError(f"Calibration npy channel dim must be 3, got {data.shape}")
    return np.ascontiguousarray(data, dtype=np.float32)


def letterbox_resize(image: Image.Image, new_shape: Tuple[int, int], fill_color=(0, 0, 0)) -> Image.Image:
    # new_shape: (h, w)
    orig_w, orig_h = image.size
    new_h, new_w = int(new_shape[0]), int(new_shape[1])

    scale = min(new_w / orig_w, new_h / orig_h)
    resize_w, resize_h = int(orig_w * scale), int(orig_h * scale)
    pad_w, pad_h = new_w - resize_w, new_h - resize_h
    pad_left, pad_top = pad_w // 2, pad_h // 2

    image = image.resize((resize_w, resize_h), Image.BILINEAR)
    new_image = Image.new("RGB", (new_w, new_h), fill_color)
    new_image.paste(image, (pad_left, pad_top))
    return new_image


def preprocess_image(im0_bgr: np.ndarray, input_wh: Tuple[int, int]) -> np.ndarray:
    # Align with /workspace/onnx_infer.py preprocessing.
    im_pil = Image.fromarray(cv2.cvtColor(im0_bgr, cv2.COLOR_BGR2RGB))
    resized = letterbox_resize(im_pil, new_shape=(input_wh[1], input_wh[0]))
    tensor = T.ToTensor()(resized)
    return np.ascontiguousarray(tensor.numpy(), dtype=np.float32)


def denormalize_bbox_to_original(
    bbox_pred: torch.Tensor,
    orig_size: torch.Tensor,
    input_size: Tuple[int, int],
) -> torch.Tensor:
    # input_size: (h, w)
    bs, _, _ = bbox_pred.shape
    device = bbox_pred.device
    input_h, input_w = int(input_size[0]), int(input_size[1])

    scale = torch.tensor([input_w, input_h, input_w, input_h], device=device).view(1, 1, 4)
    bbox_scaled = bbox_pred * scale

    bbox_orig = torch.zeros_like(bbox_scaled)
    if bs != orig_size.shape[0] and orig_size.shape[0] == 1:
        orig_size = orig_size.repeat(bs, 1)

    for i in range(bs):
        orig_w, orig_h = orig_size[i]
        r = min(input_w / orig_w, input_h / orig_h)
        new_w, new_h = int(orig_w * r), int(orig_h * r)
        pad_w = (input_w - new_w) / 2
        pad_h = (input_h - new_h) / 2

        x1 = (bbox_scaled[i, :, 0] - pad_w) / r
        y1 = (bbox_scaled[i, :, 1] - pad_h) / r
        x2 = (bbox_scaled[i, :, 2] - pad_w) / r
        y2 = (bbox_scaled[i, :, 3] - pad_h) / r
        bbox_orig[i] = torch.stack([x1, y1, x2, y2], dim=-1)

    return bbox_orig


def postprocess_output(
    output_per_image: np.ndarray,
    orig_hw: Tuple[int, int],
    input_wh: Tuple[int, int],
    score_thresh: float,
) -> np.ndarray:
    # Align with /workspace/onnx_infer.py postprocessing.
    boxes = torch.from_numpy(output_per_image[:, :4]).float().unsqueeze(0)
    logits = torch.from_numpy(output_per_image[:, 4:]).float().unsqueeze(0)

    bbox_pred = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
    orig_h, orig_w = int(orig_hw[0]), int(orig_hw[1])
    orig_size = torch.tensor([[orig_w, orig_h]], dtype=torch.float32)
    bbox_pred = denormalize_bbox_to_original(
        bbox_pred,
        orig_size,
        input_size=(int(input_wh[1]), int(input_wh[0])),
    )

    scores = F.sigmoid(logits.squeeze(0))
    scores, labels = torch.topk(scores, 1, dim=-1)
    scores = scores.squeeze(-1)
    labels = labels.squeeze(-1)
    boxes_xyxy = bbox_pred.squeeze(0)

    keep = scores >= float(score_thresh)
    if not keep.any():
        return np.zeros((0, 6), dtype=np.float32)

    boxes_xyxy = boxes_xyxy[keep]
    scores = scores[keep]
    labels = labels[keep]

    boxes_xyxy[:, 0::2] = boxes_xyxy[:, 0::2].clamp_(0, orig_w)
    boxes_xyxy[:, 1::2] = boxes_xyxy[:, 1::2].clamp_(0, orig_h)

    pred = torch.cat(
        [boxes_xyxy, scores.unsqueeze(1), labels.float().unsqueeze(1)],
        dim=1,
    )
    return pred.detach().cpu().numpy().astype(np.float32)


class SmokeAutoCalibrationEvaluator(AutoCalibrationEvaluatorBase):
    def __init__(
        self,
        calibration_npy: str,
        eval_data_dir: str,
        batch_num: int,
        batch_size: int,
        calibration_iters: int,
        calibration_samples: int,
        expected_metric_loss: float,
        input_width: int,
        input_height: int,
        conf_thres: float,
        eval_max_images: int,
    ):
        super().__init__()
        self.batch_num = int(batch_num)
        self.batch_size = int(batch_size)
        self.calibration_iters = int(calibration_iters)
        self.calibration_samples = int(calibration_samples)
        self.expected_metric_loss = float(expected_metric_loss)
        self.input_wh = (int(input_width), int(input_height))
        self.conf_thres = float(conf_thres)
        self.eval_max_images = int(eval_max_images)

        self.names = LABELS.smokephone
        self.names_merge = LABELS.smokephone_merge
        self.names_dataset = {}
        self.names_dic = {k: v for k, v in enumerate(self.names)}
        self.names_dic_mg = {k: v for k, v in enumerate(self.names_merge.keys())}
        self.filter_names: List[str] = []
        self.filter_names_mg: List[str] = []
        self.iouv = torch.linspace(0.05, 0.5, 10)
        self.niou = self.iouv.numel()

        self.calibration_data = load_preprocessed_calibration(calibration_npy)
        if self.calibration_data.shape[2] != self.input_wh[1] or self.calibration_data.shape[3] != self.input_wh[0]:
            raise RuntimeError(
                f"Calibration npy shape mismatch: got {self.calibration_data.shape}, "
                f"expected N,3,{self.input_wh[1]},{self.input_wh[0]}"
            )

        eval_images_dir = Path(eval_data_dir) / "images"
        self.eval_images = collect_eval_images_recursive(
            str(eval_images_dir if eval_images_dir.exists() else eval_data_dir)
        )
        if self.eval_max_images > 0:
            self.eval_images = self.eval_images[: self.eval_max_images]

    @staticmethod
    def _run_batch(session: ort.InferenceSession, batch_input: np.ndarray) -> np.ndarray:
        input_name = session.get_inputs()[0].name
        output_value = session.run(None, {input_name: batch_input})[0]
        return normalize_output(output_value)

    def calibration(self, model_file: str):
        if self.calibration_data.size == 0:
            raise RuntimeError("Calibration data is empty.")

        if self.calibration_samples > 0:
            iterations = int(math.ceil(float(self.calibration_samples) / float(self.batch_size)))
        elif self.calibration_iters > 0:
            iterations = self.calibration_iters
        else:
            iterations = self.batch_num
        iterations = max(iterations, self.batch_num)

        session = create_session(model_file)
        sample_count = int(self.calibration_data.shape[0])
        for idx in progress_bar(range(iterations), total=iterations, desc="校准进度"):
            start = (idx * self.batch_size) % sample_count
            batch_indices = [(start + k) % sample_count for k in range(self.batch_size)]
            batch_input = self.calibration_data[batch_indices]
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

                pred = postprocess_output(
                    output_per_image=output_np[b],
                    orig_hw=(im0.shape[0], im0.shape[1]),
                    input_wh=self.input_wh,
                    score_thresh=self.conf_thres,
                )
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
            batch_inputs.append(preprocess_image(im0, self.input_wh))
            batch_img_paths.append(img_path)
            batch_raw_images.append(im0)

            if len(batch_inputs) >= self.batch_size:
                flush_batch()

        flush_batch()

        if not stats and not stats_mg:
            return 0.0

        map_weighted = 0.0
        _, _, _, _, _, _, _, map_weighted, _, _, _, _, _, _ = utils.mAP(stats_mg, self.names_dic_mg)

        # Prefer merged metric for smoke dataset evaluation.
        return float(map_weighted)

    def metric_eval(self, original_metric: float, new_metric: float):
        loss = float(original_metric - new_metric)
        return (loss <= self.expected_metric_loss), loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AMCT ONNX auto quantization for SmokePhone model")
    parser.add_argument(
        "--model",
        default="/workspace/models/SmokePhone_od-v5-1-x-best-d4-416-224-opset16_20251104_no_layernorm.onnx",
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--calibration-npy",
        default="/workspace/smoke_phone-416-768.npy",
        help="Preprocessed calibration data .npy (NCHW/NHWC)",
    )
    parser.add_argument(
        "--eval-data-dir",
        default="/workspace/AlgoServerScript/datasets/smoke_phone",
        help="Evaluation dataset root (contains images/ and labels/)",
    )
    parser.add_argument(
        "--output-dir",
        default="/workspace/quantization/out/auto_quant_smoke_result_2",
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
        default=10,
        help="Target calibration image count; if >0, overrides calib-iters",
    )
    parser.add_argument(
        "--expected-metric-loss",
        type=float,
        default=0.5,
        help="Allowed mAP loss (absolute value)",
    )
    parser.add_argument("--input-width", type=int, default=224, help="Model input width")
    parser.add_argument("--input-height", type=int, default=416, help="Model input height")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="Score threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="Reserved for compatibility")
    parser.add_argument("--max-det", type=int, default=300, help="Reserved for compatibility")
    parser.add_argument(
        "--eval-max-images",
        type=int,
        default=10,
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

    calibration_npy = os.path.realpath(args.calibration_npy)
    if not os.path.isfile(calibration_npy):
        raise RuntimeError(f"Calibration npy not found: {calibration_npy}")

    eval_data_dir = os.path.realpath(args.eval_data_dir)
    if not os.path.isdir(eval_data_dir):
        raise RuntimeError(f"Evaluation data directory not found: {eval_data_dir}")

    output_dir = os.path.realpath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    config_file = os.path.join(output_dir, "config.json")
    record_file = os.path.join(output_dir, "scale_offset_record.txt")
    save_prefix = os.path.join(output_dir, "smoke_model")
    skip_layers = [s.strip() for s in args.skip_layers.split(",") if s.strip()]

    amct.create_quant_config(
        config_file=config_file,
        model_file=model_file,
        skip_layers=skip_layers,
        batch_num=args.batch_num,
        activation_offset=args.activation_offset,
    )

    evaluator = SmokeAutoCalibrationEvaluator(
        calibration_npy=calibration_npy,
        eval_data_dir=eval_data_dir,
        batch_num=args.batch_num,
        batch_size=args.batch_size,
        calibration_iters=args.calib_iters,
        calibration_samples=args.calib_samples,
        expected_metric_loss=args.expected_metric_loss,
        input_width=args.input_width,
        input_height=args.input_height,
        conf_thres=args.conf_thres,
        eval_max_images=args.eval_max_images,
    )

    from incremental_strategy import IncrementalStrategy

    incremental_strategy = IncrementalStrategy(step_ratio=0.2, min_step=1)

    amct.accuracy_based_auto_calibration(
        model_file=model_file,
        model_evaluator=evaluator,
        config_file=config_file,
        record_file=record_file,
        save_dir=save_prefix,
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
