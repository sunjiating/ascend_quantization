#!/bin/bash
TARGET_FILE="/workspace/quantization/out/manual_quant_smoke_result3/SmokePhone_od-v5-1-x-best-d4-416-224-opset16_20251104_no_layernorm_deploy_model.onnx"
OUTPUT_FILE="/workspace/SmokePhone_od-v5-1-x-best-d4-int8-416-224-opset16_20251104"

INPUT_SHAPE="images:4,3,416,224"

atc \
    --precision_mode_v2="cube_fp16in_fp32out"   \
    --host_env_cpu="aarch64" --framework=5 --log=info   --input_format=NCHW   \
    --insert_op_conf="/workspace/objectdetectV3_768x416_YUV2BGR.cfg"   \
    --model="$TARGET_FILE"   \
    --output="$OUTPUT_FILE"   \
    --soc_version="Ascend310P3"   \
    --input_shape="$INPUT_SHAPE"

chmod -R 777 $OUTPUT_FILE.om