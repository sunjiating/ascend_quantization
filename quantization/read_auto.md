# PersonCarAnimal 自动精度量化说明（AMCT ONNX）

## 1. 目标与文件

- 脚本：`/workspace/quantization/auto_quant_personcar.py`
- 待量化模型（默认）：`/workspace/models/PersonCarAnimal_od-v3-x-bestp-d4-416-768_20251203.onnx`
- 校准集（默认）：`/workspace/datasets/person_car_animal-1101`
- 评测集（默认）：`/workspace/AlgoServerScript/datasets/person_car`
- 输出目录（默认）：`/workspace/quantization/out/auto_quant_personcar_result`

## 2. 流程说明

脚本基于 AMCT 自动量化流程：

1. `amct.create_quant_config(...)`
2. `amct.accuracy_based_auto_calibration(...)`

评估器 `PersonCarAutoCalibrationEvaluator` 实现：

- `calibration(model_file)`：使用校准集做前向，保证校准迭代数不小于 `batch_num`
- `evaluate(model_file)`：计算 `mAP@0.25:0.7`（`map_2570`）
- `metric_eval(original_metric, new_metric)`：以 `expected_metric_loss` 判断精度是否达标

## 3. 依赖要求

至少需要：

- `amct_onnx`
- `onnxruntime`（建议具备 CUDA provider）
- `opencv-python`
- `numpy`
- `torch`
- `tqdm`（可选，提供进度条）

并依赖 AlgoServerScript 中的评测代码：

- `/workspace/AlgoServerScript/src/utils.py`
- `/workspace/AlgoServerScript/LABELS.py`

## 4. 运行方式

### 4.1 最小示例

```bash
python3 /workspace/quantization/auto_quant_personcar.py
```

### 4.2 常用示例

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

## 5. 关键参数

- `--batch-num`：AMCT 最小校准批次数要求（默认 `4`）
- `--batch-size`：校准和评估 batch 大小（默认 `8`）
- `--calib-samples`：目标校准样本数，`>0` 时优先级高于 `--calib-iters`
- `--calib-iters`：实际校准迭代数，默认 `0`（由 `batch-num`/`calib-samples` 决定）
- `--expected-metric-loss`：允许的 mAP 绝对损失（默认 `0.001`）
- `--eval-max-images`：评估图像数上限，`0` 表示全量
- `--conf-thres` / `--iou-thres` / `--max-det`：评估阶段 NMS 参数
- `--activation-offset` / `--no-activation-offset`：控制 activation offset（默认启用）
- `--skip-layers`：逗号分隔层名，指定跳过量化层
- `--strategy` / `--sensitivity`：自动量化策略参数（当前脚本内部默认使用 `IncrementalStrategy(step_ratio=0.2)`）

## 6. 输出文件

默认输出目录下主要产物：

- `config.json`：量化配置
- `scale_offset_record.txt`：量化 scale/offset 记录
- `personcar_model_fake_quant_model.onnx`：仿真量化模型
- `personcar_model_deploy_model.onnx`：部署模型

自动量化过程中还可能生成 AMCT 分析文件（灵敏度排序、回退过程信息等），文件名以实际 AMCT 版本输出为准。

## 7. 常见问题

- `Calibration directory not found`：校准目录路径错误或未挂载。
- `Evaluation data directory not found`：评估目录错误，或不包含可读取图片。
- 评估很慢：先用 `--eval-max-images 200` 快速验证，再切回全量评估。
- 精度下降超阈值：适当放宽 `--expected-metric-loss`，或增加 `--skip-layers` 做手动回退。
