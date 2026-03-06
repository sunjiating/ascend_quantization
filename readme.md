# Ascend ONNX 模型量化手册

本仓库提供 PersonCarAnimal 模型在 AMCT ONNX 上的两种量化流程：

1. 手工量化：`quantization/manual_quant_perscar.py`
2. 自动精度量化：`quantization/auto_quant_personcar.py`

详细说明见：

- 手工量化文档：`quantization/read_mual.md`
- 自动量化文档：`quantization/read_auto.md`

## 1. 环境准备

优先使用已准备好的镜像：

- `quay.io/ascend/cann:8.2.rc1-310p-ubuntu22.04-py3.11`
记得要将AlgoServerScript的代码挂载到容器/workspace/AlgoServerScript目录下  
具体docker运行配置参照docker-compose.yml

若需自行搭建，可参考如下步骤（按需执行）：

```bash
# 1) 创建容器（示例）
docker run -itd --name xs_atlas_quatization \
  -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY \
  -v /data/xs/workspace/ml/atlas_quatization:/home/osmagic \
  -v /etc/timezone:/etc/timezone \
  -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility \
  -e LANG=zh_CN.UTF-8 -e LC_ALL=zh_CN.UTF-8 \
  --privileged=true --shm-size 64G --gpus all \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  192.168.2.78:1443/algolib/atlas:8.0.0.alpha001

# 2) Python 环境
apt update && apt upgrade -y
apt install -y wget libgl1 libglib2.0-0
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
~/miniconda3/bin/conda init bash
conda config --set auto_activate_base false
conda create -n atlas python=3.10.0 -y

# 3) 进入环境并安装依赖
conda activate atlas
pip install -r /workspace/requirements.txt
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# 4) 安装 amct_onnx（示例路径）
cd /home/osmagic/amct/amct_onnx
pip install amct_onnx-0.19.0-py3-none-linux_x86_64.whl
cd amct_onnx_op && python3 setup.py build
```

## 2. 快速开始

### 2.1 手工量化
- 介绍  
手动量化实现简单，速度快，但无法自动搜索某些敏感节点去排除量化
```bash
python3 /workspace/quantization/manual_quant_perscar.py \
  --model /workspace/models/PersonCarAnimal_od-v3-x-bestp-d4-416-768_20251203.onnx \
  --calibration-dir /workspace/datasets/person_car_animal-1101 \
  --output-dir /workspace/quantization/out/manual_quant_perscar_result \
  --batch-num 1 \
  --batch-size 4 \
  --calib-samples 1101 \
  --input-width 768 \
  --input-height 416
```

- 重要参数说明  
--calib-samples：需要使用的校准数据个数  
--nuq：非均匀量化  
--skip-layers：逗号分隔的“跳过量化节点名”，如["/model.1/conv/Conv", "/model.2/conv/Conv"]

- 运行后生成文件  
-config.json 量化配置文件，包含各个节点的量化配置  
-xxxx_deploy_model.onnx 可部署的量化模型文件，经过ATC转换工具转换后可部署到昇腾AI处理器。  
-xxxx_fake_quant_model.onnx 精度仿真模型文件,可以在ONNX执行框架ONNX Runtime进行精度仿真。

- 注意：  
1.换模型量化直接修改预处理函数即可  
2.模型精度不理想，skip-layers参数回退节点，或修改量化算法：https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/devaids/amct/atlasamct_16_0146.html

PersonCarAnimal_od-v3-x-bestp-d4-416-768_20251203模型回退节点:
skip_layers = [
    "/model.42/cv3.0/cv3.0.0/conv/Conv", 
    "/model.42/cv3.1/cv3.1.1/conv/Conv",
    "/model.42/cv3.1/cv3.1.0/conv/Conv",
    "/model.42/cv3.0/cv3.0.1/conv/Conv",
    "/model.42/cv2.0/cv2.0.0/conv/Conv",
    "/model.42/cv3.2/cv3.2.0/conv/Conv",
    "/model.42/cv2.1/cv2.1.0/conv/Conv",
    "/model.42/cv2.0/cv2.0.1/conv/Conv",
    "/model.42/cv3.2/cv3.2.1/conv/Conv",
    "/model.42/cv2.1/cv2.1.1/conv/Conv",
    "/model.42/dfl/conv/Conv",
    "/model.42/cv2.2/cv2.2.0/conv/Conv",
    "/model.42/cv2.2/cv2.2.1/conv/Conv",
    "/model.42/cv2.0/cv2.0.2/Conv",
    "/model.42/cv2.1/cv2.1.2/Conv",
    "/model.42/cv2.2/cv2.2.2/Conv",
    "/model.42/cv3.1/cv3.1.2/Conv",
    "/model.42/cv3.0/cv3.0.2/Conv",
    "/model.42/cv3.2/cv3.2.2/Conv"
    ]

### 2.2 自动精度量化
- 介绍  
可自动搜索某些敏感节点，但我测试时，发现其精度分析回退并非完全准确，若有经验可直接使用手动量化skip-layers跳过量化节点
```bash
python3 /workspace/quantization/auto_quant_personcar.py \
  --model /workspace/models/PersonCarAnimal_od-v3-x-bestp-d4-416-768_20251203.onnx \
  --calibration-dir /workspace/datasets/person_car_animal-1101 \
  --eval-data-dir /workspace/AlgoServerScript/datasets/person_car \
  --output-dir /workspace/quantization/out/auto_quant_personcar_result \
  --batch-num 4 \
  --batch-size 8 \
  --calib-samples 1101 \
  --expected-metric-loss 0.005 \
  --eval-max-images 0
```
- 重要参数说明  
--eval-data-dir:评估数据集  
--eval-max-images：评估数据个数 0为全部使用  
--calib-samples：需要使用的校准数据个数  
--expected-metric-loss：可接受的精度mAP损失

- 运行后生成文件  
-config.json 量化配置文件，包含各个节点的量化配置  
-xxxx_deploy_model.onnx 可部署的量化模型文件，经过ATC转换工具转换后可部署到昇腾AI处理器。  
-xxxx_fake_quant_model.onnx 精度仿真模型文件,可以在ONNX执行框架ONNX Runtime进行精度仿真。  
-accuracy_based_auto_calibration_final_config.json 经多次迭代回退后最终量化配置文件  
-accuracy_based_auto_calibration_ranking_information.json 敏感层从低到高排序

- 注意：  
换模型量化时，注意修改预处理函数！  
评估精度评估指标直接使用的algoserver中map_2570，若要修改，可在evaluate中修改选择algoserver其它评估指标

## 3. 输出文件

量化完成后，核心产物通常包括：

- `config.json`：量化配置
- `*_fake_quant_model.onnx`：用于 ONNX Runtime 精度验证
- `*_deploy_model.onnx`：用于后续 ATC 转换部署
- 自动量化额外日志（例如 AMCT 过程文件、灵敏度/回退信息）

## 4. ATC 转换示例

```bash
# 在algoserver上部署的模型转换（部署场景）
atc \
  --precision_mode_v2="cube_fp16in_fp32out" \
  --host_env_cpu="aarch64" --framework=5 --log=info \
  --input_format=NCHW \
  --insert_op_conf="/workspace/objectdetectV3_768x416_YUV2BGR.cfg" \
  --model="/workspace/quantization/out/manual_quant_smoke_result2/SmokePhone_od-v5-1-x-best-d4-416-224-opset16_20251104_no_layernorm_deploy_model.onnx" \
  --output="/workspace/SmokePhone_od-v5-1-x-best-d4-int8-416-224-opset16_20251104" \
  --soc_version="Ascend310P3" \
  --input_shape="images:4,3,416,224"

# 自动量化模型（自测场景）
atc \
  --precision_mode_v2="cube_fp16in_fp32out" \
  --framework=5 --host_env_cpu="aarch64" --log=info \
  --input_format=NCHW \
  --model="/workspace/quantization/out/manual_quant_smoke_result/SmokePhone_od-v5-1-x-best-d4-416-224-opset16_20251104_deploy_model.onnx" \
  --output="/workspace/quantization/SmokePhone_od-v5-1-x-best-d4-416-224-opset16_20251104" \
  --soc_version="Ascend310P3" \
  --input_shape="input:1,3,416,768"
```

# 注意！
对于SmokePhone_od-v5-1-x-best-d4-416-224模型（包含layernorm算子），自动精度量化失效
首先执行python /workspace/quantization/rewrite_layernorm_onnx.py将layernorm算子拆分成基础算子
然后执行python /workspace/quantization/manual_quant_smoke.py量化
最后执行bash /workspace/atc.sh模型转换为om