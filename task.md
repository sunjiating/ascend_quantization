1.ascend_quantization官方案例代码/workspace/quantization/official_sample.py
2.校准数据集/workspace/datasets/person_car_animal-1101
3.评测数据集/workspace/AlgoServerScript/datasets/person_car
4.需要量化的模型/workspace/models/PersonCarAnimal_od-v3-x-bestp-d4-416-768_20251203.onnx
5.evaluate可参照/workspace/AlgoServerScript/algo_server.py的map指标实现
6.amct量化工具介绍、用法、api说明文档/workspace/CANN社区版 8.5.0 AMCT模型压缩工具用户指南 01.md（内容较大，若有需要读取对应部分内容即可）

参照ascend_quantization官方案例，实现对/workspace/models/PersonCarAnimal_od-v3-x-bestp-d4-416-768_20251203.onnx模型的量化。
生成的代码放在/workspace/quantization目录。
生成说明文档readme.md.
用中文回复。

# algoserver模型转换
atc \
    --precision_mode_v2="cube_fp16in_fp32out" \
    --host_env_cpu="aarch64" --framework=5 --log=info --input_format=NCHW \
    --insert_op_conf="/workspace/objectdetectV3_768x416_YUV2BGR.cfg"  \
    --model="/workspace/quantization/out/auto_quant_personcar_result/personcar_model_deploy_model.onnx" \
    --output="/workspace/quantization/PersonCarAnimal_od-v3-x-bestp-d4-int8-416-768_20251203" \
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