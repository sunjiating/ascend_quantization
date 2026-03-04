#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import os
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort

import amct_onnx as amct
from utils import letterbox


def collect_images(root: str) -> List[str]:
    exts = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
        ".tiff",
        ".JPG",
        ".JPEG",
        ".PNG",
        ".BMP",
        ".WEBP",
        ".TIFF",
    }
    root_path = Path(root)
    if not root_path.exists():
        return []
    images = [str(p) for p in root_path.rglob("*") if p.suffix in exts]
    images.sort()
    return images


def create_session(model_file: str) -> ort.InferenceSession:
    providers = []
    available = set(ort.get_available_providers())
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    if not providers:
        providers = None

    amct_so = getattr(amct, "AMCT_SO", None)
    if amct_so is None:
        return ort.InferenceSession(model_file, providers=providers)
    try:
        return ort.InferenceSession(model_file, amct_so, providers=providers)
    except TypeError:
        return ort.InferenceSession(model_file, sess_options=amct_so, providers=providers)


def preprocess_image(path: Path, imgsz: Sequence[int], stride: int) -> np.ndarray:
    im = cv2.imread(str(path))
    if im is None:
        raise FileNotFoundError(f"failed to read image: {path}")
    im, _, _ = letterbox(im, (int(imgsz[0]), int(imgsz[1])), stride=stride)
    im = im.transpose((2, 0, 1))[::-1]  # BGR->RGB, HWC->CHW
    return np.ascontiguousarray(im, dtype=np.float32) / 255.0


def normalize_output(output_value: np.ndarray) -> np.ndarray:
    out = np.asarray(output_value)
    if out.ndim == 4 and out.shape[1] == 1:
        out = out[:, 0]
    elif out.ndim == 2:
        out = out[None, ...]
    if out.ndim != 3:
        raise RuntimeError(f"unexpected model output shape: {out.shape}")
    return out


def onnx_forward(
    onnx_model: str,
    calibration_images: Sequence[str],
    batch_size: int,
    iterations: int,
    input_wh: Tuple[int, int],
    stride: int = 32,
) -> None:
    if not calibration_images:
        raise RuntimeError("no calibration images found")

    session = create_session(onnx_model)
    input_name = session.get_inputs()[0].name
    image_count = len(calibration_images)

    for idx in range(iterations):
        start = (idx * batch_size) % image_count
        batch_paths = [calibration_images[(start + j) % image_count] for j in range(batch_size)]
        batch_data = [preprocess_image(Path(p), input_wh, stride) for p in batch_paths]
        batch_input = np.stack(batch_data, axis=0)
        output_value = session.run(None, {input_name: batch_input})[0]
        _ = normalize_output(output_value)
        print(f"[Calibration] iteration {idx + 1}/{iterations} finished.")


def resolve_calibration_iterations(
    batch_num: int,
    batch_size: int,
    calib_iters: int,
    calib_samples: int,
) -> int:
    if calib_samples > 0:
        resolved = int(math.ceil(float(calib_samples) / float(batch_size)))
    elif calib_iters > 0:
        resolved = int(calib_iters)
    else:
        resolved = int(batch_num)
    return max(int(batch_num), resolved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AMCT ONNX manual quantization for PersonCarAnimal model"
    )
    parser.add_argument(
        "--model",
        default="/workspace/models/PersonCarAnimal_od-v3-x-bestp-d4-416-768_20251203.onnx",
        help="path to ONNX model",
    )
    parser.add_argument(
        "--calibration-dir",
        default="/workspace/datasets/person_car_animal-1101",
        help="calibration image directory",
    )
    parser.add_argument(
        "--output-dir",
        default="/workspace/quantization/out/manual_quant_perscar_result_2",
        help="output directory",
    )
    parser.add_argument("--batch-num", type=int, default=8, help="AMCT required minimum calibration batches")
    parser.add_argument("--batch-size", type=int, default=8, help="batch size for calibration")
    parser.add_argument(
        "--calib-iters",
        type=int,
        default=0,
        help="actual calibration forward iterations; 0 means use batch-num",
    )
    parser.add_argument(
        "--calib-samples",
        type=int,
        default=1101,
        help="target calibration image count; if >0, overrides calib-iters (iterations=ceil(samples/batch-size))",
    )
    parser.add_argument("--input-width", type=int, default=768, help="model input width")
    parser.add_argument("--input-height", type=int, default=416, help="model input height")
    parser.add_argument("--skip-layers", default="", help="comma-separated layer names to skip")
    parser.add_argument("--nuq", action="store_true", help="enable non-uniform quantization mode")
    parser.add_argument("--nuq-config", default="", help="path to NUQ config file")

    activation_group = parser.add_mutually_exclusive_group()
    activation_group.add_argument("--activation-offset", action="store_true", help="enable activation offset")
    activation_group.add_argument("--no-activation-offset", action="store_true", help="disable activation offset")

    args = parser.parse_args()
    if not args.activation_offset and not args.no_activation_offset:
        args.activation_offset = True
    elif args.no_activation_offset:
        args.activation_offset = False
    return args


def main() -> None:
    args = parse_args()

    model_file = os.path.realpath(args.model)
    if not os.path.isfile(model_file):
        raise RuntimeError(f"model not found: {model_file}")

    calibration_dir = os.path.realpath(args.calibration_dir)
    if not os.path.isdir(calibration_dir):
        raise RuntimeError(f"calibration directory not found: {calibration_dir}")

    output_dir = os.path.realpath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    batch_num = int(args.batch_num)
    batch_size = int(args.batch_size)
    calib_iters = int(args.calib_iters)
    calib_samples = int(args.calib_samples)
    input_wh = (int(args.input_width), int(args.input_height))
    skip_layers = [s.strip() for s in args.skip_layers.split(",") if s.strip()]
    if batch_num <= 0:
        raise RuntimeError(f"batch-num must be > 0, got {batch_num}")
    if batch_size <= 0:
        raise RuntimeError(f"batch-size must be > 0, got {batch_size}")
    if calib_iters < 0:
        raise RuntimeError(f"calib-iters must be >= 0, got {calib_iters}")
    if calib_samples < 0:
        raise RuntimeError(f"calib-samples must be >= 0, got {calib_samples}")
    if input_wh[0] <= 0 or input_wh[1] <= 0:
        raise RuntimeError(f"input size must be > 0, got {input_wh}")

    calibration_images = collect_images(calibration_dir)
    if not calibration_images:
        raise RuntimeError(f"no calibration images found under {calibration_dir}")
    calibration_iterations = resolve_calibration_iterations(
        batch_num=batch_num,
        batch_size=batch_size,
        calib_iters=calib_iters,
        calib_samples=calib_samples,
    )
    used_samples = calibration_iterations * batch_size

    config_json_file = os.path.join(output_dir, "config.json")
    record_file = os.path.join(output_dir, "record.txt")
    modified_model = os.path.join(output_dir, "modified_model.onnx")
    save_prefix = os.path.join(output_dir, Path(model_file).stem)

    config_defination = None
    if args.nuq:
        if not args.nuq_config:
            raise RuntimeError("NUQ mode requires --nuq-config")
        config_defination = os.path.realpath(args.nuq_config)
        if not os.path.isfile(config_defination):
            raise RuntimeError(f"nuq config not found: {config_defination}")

    print("[INFO] create quant config ...")
    amct.create_quant_config(
        config_file=config_json_file,
        model_file=model_file,
        skip_layers=skip_layers,
        batch_num=batch_num,
        activation_offset=args.activation_offset,
        config_defination=config_defination,
    )

    print("[INFO] quantize model (phase1) ...")
    amct.quantize_model(
        config_file=config_json_file,
        model_file=model_file,
        modified_onnx_file=modified_model,
        record_file=record_file,
    )

    print("[INFO] run calibration forward on modified model ...")
    onnx_forward(
        onnx_model=modified_model,
        calibration_images=calibration_images,
        batch_size=batch_size,
        iterations=calibration_iterations,
        input_wh=input_wh,
        stride=32,
    )

    print("[INFO] save fake-quant/deploy models (phase3) ...")
    amct.save_model(modified_model, record_file, save_prefix)

    print("[INFO] manual quantization finished.")
    print(f"Output directory: {output_dir}")
    print(f"Config file: {config_json_file}")
    print(f"Record file: {record_file}")
    print(f"Modified model: {modified_model}")
    print(f"Fake quant model: {save_prefix}_fake_quant_model.onnx")
    print(f"Deploy model: {save_prefix}_deploy_model.onnx")
    print(f"Skip layers count: {len(skip_layers)}")
    print(f"AMCT batch_num: {batch_num}")
    print(f"Calibration iterations: {calibration_iterations}")
    print(f"Calibration samples used (with repeat): {used_samples}")
    print(f"Unique calibration images available: {len(calibration_images)}")


if __name__ == "__main__":
    main()
