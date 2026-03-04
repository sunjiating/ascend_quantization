# PersonCarAnimal ONNX 量化说明

## 1. 目标

基于 AMCT ONNX 官方“基于精度的自动量化”流程，对以下模型进行量化：

- 待量化模型：`/workspace/models/PersonCarAnimal_od-v3-x-bestp-d4-416-768_20251203.onnx`
- 校准数据集：`/workspace/datasets/person_car_animal-1101`
- 评测数据集：`/workspace/AlgoServerScript/datasets/person_car`

量化脚本：`/workspace/quantization/auto_quant_personcar.py`

## 2. 实现说明

脚本实现遵循文档接口流程：

1. `amct.create_quant_config(...)`
2. `amct.accuracy_based_auto_calibration(...)`

并实现了 `AutoCalibrationEvaluatorBase` 的三个回调函数：

- `calibration(model_file)`：使用校准集做前向推理，保证推理 batch 数不小于 `batch_num`
- `evaluate(model_file)`：参考 `AlgoServerScript/algo_server.py` 的人车 mAP 流程，计算 `mAP@0.25:0.7`
- `metric_eval(original_metric, new_metric)`：以 mAP 损失是否小于阈值判定量化是否达标

## 3. 依赖要求

需要可用的 Python3 环境，并安装以下依赖（至少）：

- `amct_onnx`
- `onnxruntime`
- `opencv-python`
- `numpy`
- `torch`
- `tqdm`（用于显示校准/评估进度条）

另外脚本会复用以下仓库代码：

- `AlgoServerScript/src/utils.py`
- `AlgoServerScript/vision.py`
- `AlgoServerScript/LABELS.py`

## 4. 运行方式

进入工作目录后执行：

```bash
python3 /workspace/quantization/auto_quant_personcar.py
```

常用参数示例（调大校准数据量、全量评测）：

```bash
python3 /workspace/quantization/auto_quant_personcar.py \
  --batch-num 8 \
  --batch-size 4 \
  --expected-metric-loss 0.005 \
  --eval-max-images 0
```

## 5. 关键参数说明

- `--batch-num`：校准 batch 数，实际校准图片数约为 `batch-num * batch-size`
- `--batch-size`：校准和评测推理 batch 大小
- `--expected-metric-loss`：允许的 mAP 下降阈值（绝对值）
- `--eval-max-images`：评测图片上限，`0` 表示评测全部图片
- `--activation-offset / --no-activation-offset`：是否启用 activation offset
- `--skip-layers`：逗号分隔的跳过量化层名

## 6. 输出文件

默认输出目录：

- `/workspace/quantization/auto_quant_personcar_result`

主要文件：

- `config.json`：自动生成的量化配置
- `scale_offset_record.txt`：量化因子记录文件
- `personcar_model_fake_quant_model.onnx`：可在 ONNX Runtime 上验证精度的仿真模型
- `personcar_model_deploy_model.onnx`：部署模型（用于后续 ATC 转换）

## 7. 注意事项

- 若校准图片不足（小于 `batch-num * batch-size`），脚本会报错退出。
- 自动量化会多次调用评测流程，若耗时较长可先减小 `--eval-max-images` 做快速验证，再使用全量评测。
- 若出现 AMCT 自定义算子相关报错，请先确认 AMCT 环境变量与安装版本匹配。
