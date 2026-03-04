
# algoserver模型转换
atc \
    --precision_mode_v2="cube_fp16in_fp32out" \
    --host_env_cpu="aarch64" --framework=5 --log=info --input_format=NCHW \
    --insert_op_conf="/workspace/objectdetectV3_768x416_YUV2BGR.cfg"  \
    --model="/workspace/quantization/out/auto_quant_personcar_result/personcar_model_deploy_model.onnx" \
    --output="/workspace/PersonCarAnimal_od-v3-x-bestp-d4-int8-416-768_20251203" \
    --soc_version="Ascend310P3" --input_shape="input:4,3,416,768"

# 自测模型转换
atc \
    --precision_mode_v2="cube_fp16in_fp32out" \
    --model=/workspace/quantization/out/auto_quant_personcar_result/personcar_model_deploy_model.onnx \
    --framework=5 --host_env_cpu="aarch64" \
    --output=/workspace/quantization/PersonCarAnimal_od-v3-x-bestp-d4-int8-416-768_20251203 \
    --soc_version=Ascend310P3 \
    --input_format=NCHW  \
    --input_shape="input:1,3,416,768"  \
    --log=info