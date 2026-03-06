#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper


def get_attr(node: onnx.NodeProto, name: str, default):
    for attr in node.attribute:
        if attr.name == name:
            return helper.get_attribute_value(attr)
    return default


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]", "_", name)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "layernorm"


def build_rank_map(model: onnx.ModelProto) -> Dict[str, int]:
    rank_map: Dict[str, int] = {}

    def collect(value_infos: List[onnx.ValueInfoProto]) -> None:
        for vi in value_infos:
            t = vi.type.tensor_type
            if not t.HasField("shape"):
                continue
            rank_map[vi.name] = len(t.shape.dim)

    collect(list(model.graph.input))
    collect(list(model.graph.value_info))
    collect(list(model.graph.output))
    return rank_map


def resolve_axes(axis: int, rank: Optional[int]) -> List[int]:
    if rank is None:
        if axis == -1:
            return [-1]
        raise RuntimeError(
            f"cannot expand LayerNormalization when input rank is unknown and axis={axis} (only axis=-1 supported)"
        )

    if axis < 0:
        axis = rank + axis
    if axis < 0 or axis >= rank:
        raise RuntimeError(f"invalid axis={axis} for input rank={rank}")
    return list(range(axis, rank))


def expand_layernorm(model: onnx.ModelProto) -> int:
    graph = model.graph
    rank_map = build_rank_map(model)
    new_nodes: List[onnx.NodeProto] = []
    replaced = 0

    for idx, node in enumerate(graph.node):
        if node.domain not in ("", "ai.onnx") or node.op_type != "LayerNormalization":
            new_nodes.append(node)
            continue

        replaced += 1
        axis = int(get_attr(node, "axis", -1))
        epsilon = float(get_attr(node, "epsilon", 1e-5))
        x = node.input[0]
        scale = node.input[1] if len(node.input) > 1 and node.input[1] else ""
        bias = node.input[2] if len(node.input) > 2 and node.input[2] else ""
        out = node.output[0]

        axes = resolve_axes(axis, rank_map.get(x))
        base = sanitize_name(node.name or f"layernorm_{idx}")
        tag = f"{base}_{idx}"

        mean = f"{tag}_mean"
        centered = f"{tag}_centered"
        sq = f"{tag}_sq"
        var = f"{tag}_var"
        eps_const = f"{tag}_eps_const"
        var_eps = f"{tag}_var_eps"
        std = f"{tag}_std"
        norm = f"{tag}_norm"

        eps_tensor = helper.make_tensor(
            name=eps_const,
            data_type=onnx.TensorProto.FLOAT,
            dims=[],
            vals=[epsilon],
        )
        graph.initializer.append(eps_tensor)

        new_nodes.append(
            helper.make_node(
                "ReduceMean",
                inputs=[x],
                outputs=[mean],
                name=f"{tag}_ReduceMean0",
                axes=axes,
                keepdims=1,
            )
        )
        new_nodes.append(helper.make_node("Sub", [x, mean], [centered], name=f"{tag}_Sub"))
        new_nodes.append(helper.make_node("Mul", [centered, centered], [sq], name=f"{tag}_MulSq"))
        new_nodes.append(
            helper.make_node(
                "ReduceMean",
                inputs=[sq],
                outputs=[var],
                name=f"{tag}_ReduceMean1",
                axes=axes,
                keepdims=1,
            )
        )
        new_nodes.append(helper.make_node("Add", [var, eps_const], [var_eps], name=f"{tag}_AddEps"))
        new_nodes.append(helper.make_node("Sqrt", [var_eps], [std], name=f"{tag}_Sqrt"))
        new_nodes.append(helper.make_node("Div", [centered, std], [norm], name=f"{tag}_Div"))

        current = norm
        if scale:
            scaled = out if not bias else f"{tag}_scaled"
            new_nodes.append(helper.make_node("Mul", [current, scale], [scaled], name=f"{tag}_MulScale"))
            current = scaled
        if bias:
            new_nodes.append(helper.make_node("Add", [current, bias], [out], name=f"{tag}_AddBias"))
        elif current != out:
            new_nodes.append(helper.make_node("Identity", [current], [out], name=f"{tag}_Identity"))

    del graph.node[:]
    graph.node.extend(new_nodes)
    return replaced


def compare_outputs(orig_path: str, new_path: str, input_shape: List[int]) -> None:
    try:
        sess_orig = ort.InferenceSession(orig_path, providers=["CPUExecutionProvider"])
        sess_new = ort.InferenceSession(new_path, providers=["CPUExecutionProvider"])
    except Exception as exc:
        print(f"[WARN] skip ORT verification: {exc}")
        return

    in_name = sess_orig.get_inputs()[0].name
    rng = np.random.default_rng(0)
    sample = rng.standard_normal(size=input_shape).astype(np.float32)
    out0 = sess_orig.run(None, {in_name: sample})[0]
    out1 = sess_new.run(None, {in_name: sample})[0]
    max_abs = float(np.max(np.abs(out0 - out1)))
    denom = np.maximum(np.abs(out0), 1e-6)
    max_rel = float(np.max(np.abs(out0 - out1) / denom))
    print(f"[VERIFY] max_abs_diff={max_abs:.8f}, max_rel_diff={max_rel:.8f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite ONNX LayerNormalization into basic ops.")
    parser.add_argument("--input", required=True, help="input ONNX path")
    parser.add_argument("--output", required=True, help="output ONNX path")
    parser.add_argument(
        "--verify-shape",
        default="",
        help="optional verification input shape, e.g. 1,3,416,224",
    )
    parser.add_argument(
        "--strict-check",
        action="store_true",
        help="fail if onnx.checker.check_model fails (default: warn and continue)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = str(Path(args.input).resolve())
    out_path = str(Path(args.output).resolve())

    model = onnx.load(in_path)
    replaced = expand_layernorm(model)
    try:
        onnx.checker.check_model(model)
    except Exception as exc:
        if args.strict_check:
            raise
        print(f"[WARN] onnx.checker failed, continue anyway: {exc}")
    onnx.save(model, out_path)
    print(f"[INFO] rewritten model saved: {out_path}")
    print(f"[INFO] LayerNormalization replaced: {replaced}")

    if args.verify_shape:
        shape = [int(v.strip()) for v in args.verify_shape.split(",") if v.strip()]
        compare_outputs(in_path, out_path, shape)


if __name__ == "__main__":
    main()
