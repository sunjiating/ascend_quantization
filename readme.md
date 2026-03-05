# ascend ONNX模型量化手册
目录
1.环境搭建
2.模型量化
-手动精度量化
-自动精度量化
## 环境搭建（不推荐，可直接用搭建好的：quay.io/ascend/cann:8.2.rc1-310p-ubuntu22.04-py3.11镜像）
基础镜像：192.168.2.78:1443/algolib/atlas:8.0.0.alpha001
1. 创建 Docker 容器
docker run -itd --name xs_atlas_quatization \
    -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY \
    -v /data/xs/workspace/ml/atlas_quatization:/home/osmagic \
    -v /etc/timezone:/etc/timezone \
    -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility \
    -e LANG=zh_CN.UTF-8 \
    -e LC_ALL=zh_CN.UTF-8 \
    --privileged=true \
    --shm-size 64G \
    --gpus all \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    192.168.2.78:1443/algolib/atlas:8.0.0.alpha001

2. 安装 Miniconda 和虚拟环境
apt update
apt upgrade -y
apt install wget

# 下载并安装 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh

# 初始化 conda
~/miniconda3/bin/conda init bash
conda config --set auto_activate false

# 创建虚拟环境
conda create -n atlas python=3.10.0

3. 安装 CUDA
# 下载 CUDA 11.8.0
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sh cuda_11.8.0_520.61.05_linux.run

# 配置环境变量
cat >> ~/.bashrc << 'EOF'
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-11.8
EOF

source ~/.bashrc

4. 安装 cuDNN
1.访问 NVIDIA cuDNN Archive 下载 cuDNN 8.6.0 对应的 deb 包（可能需要注册账号）


2.安装 cuDNN：

dpkg -i cudnn-local-repo-ubuntu2004-8.6.0.163_1.0-1_amd64.deb
cp /var/cudnn-local-repo-ubuntu2004-8.6.0.163/cudnn-local-B0FE0A41-keyring.gpg /usr/share/keyrings/
apt update
apt -y install libcudnn8 libcudnn8-dev
5. 安装 Python 库与 AMCT
# 安装基础依赖
pip install -r requirements.txt

# 安装 AMCT ONNX
cd /home/osmagic/amct/amct_onnx
pip install amct_onnx-0.19.0-py3-none-linux_x86_64.whl

# 编译自定义算子
cd amct_onnx_op
python3 setup.py build

# 安装 PyTorch（CUDA 11.8 版本）
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# 安装系统依赖
apt install libgl1 libglib2.0-0

##  模型量化
1.手动量化
介绍：手动量化实现简单，速度快，但无法自动搜索某些敏感节点去排除量化
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
重要参数说明
--calib-samples：需要使用的校准数据个数
--nuq：非均匀量化
--skip-layers：逗号分隔的“跳过量化节点名”，如["/model.1/conv/Conv", "/model.2/conv/Conv"]

运行后生成文件
-config.json 量化配置文件，包含各个节点的量化配置
-xxxx_deploy_model.onnx 可部署的量化模型文件，经过ATC转换工具转换后可部署到昇腾AI处理器。
-xxxx_fake_quant_model.onnx 精度仿真模型文件,可以在ONNX执行框架ONNX Runtime进行精度仿真。

注意：
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


2.自动精度量化
介绍：可自动搜索某些敏感节点，但我测试时，发现其精度分析回退并非完全准确，若有经验可直接使用手动量化skip-layers跳过量化节点
```bash
python3 /workspace/quantization/auto_quant_personcar.py \
  --model /workspace/models/PersonCarAnimal_od-v3-x-bestp-d4-416-768_20251203.onnx \
  --calibration-dir /workspace/datasets/person_car_animal-1101 \
  --eval-data-dir /workspace/AlgoServerScript/datasets/person_car \
  --output-dir /workspace/quantization/out/auto_quant_personcar_result \
  --batch-num 8 \
  --batch-size 4 \
  --calib-samples 1101 \
  --expected-metric-loss 0.005 \
  --eval-max-images 0
```
重要参数说明
--eval-data-dir:评估数据集
--eval-max-images：评估数据个数 0为全部使用
--calib-samples：需要使用的校准数据个数
--expected-metric-loss：可接受的精度mAP损失

运行后生成文件
-config.json 量化配置文件，包含各个节点的量化配置
-xxxx_deploy_model.onnx 可部署的量化模型文件，经过ATC转换工具转换后可部署到昇腾AI处理器。
-xxxx_fake_quant_model.onnx 精度仿真模型文件,可以在ONNX执行框架ONNX Runtime进行精度仿真。
-accuracy_based_auto_calibration_final_config.json 经多次迭代回退后最终量化配置文件
-accuracy_based_auto_calibration_ranking_information.json 敏感层从低到高排序

注意：
换模型量化时，注意修改预处理函数！
评估精度评估指标直接使用的algoserver中map_2570，若要修改，可在evaluate中修改选择algoserver其它评估指标



# 模型转换
# 在algoserver上部署的模型转换
atc \
    --precision_mode_v2="cube_fp16in_fp32out" \
    --host_env_cpu="aarch64" --framework=5 --log=info --input_format=NCHW \
    --insert_op_conf="/workspace/objectdetectV3_768x416_YUV2BGR.cfg"  \
    --model="/workspace/quantization/out/manual_quant_perscar_result/PersonCarAnimal_od-v3-x-bestp-d4-416-768_20251203_deploy_model.onnx" \
    --output="/workspace/PersonCarAnimal_od-v3-x-bestp-d4-int8-4-416-768_20251203" \
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