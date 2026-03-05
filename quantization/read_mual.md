# PersonCarAnimal 手工量化说明（AMCT ONNX）

## 1. 目标与文件

- 待量化模型：`/workspace/models/PersonCarAnimal_od-v3-x-bestp-d4-416-768_20251203.onnx`
- 校准数据集：`/workspace/datasets/person_car_animal-1101`
- 手工量化脚本：`/workspace/quantization/manual_quant_perscar.py`
- 默认输出目录：`/workspace/quantization/out/manual_quant_perscar_result`

脚本已按 AMCT 手工量化流程实现：

1. `amct.create_quant_config(...)`
2. `amct.quantize_model(...)`
3. 对 `modified_model.onnx` 执行校准前向（迭代数由 `calib_samples/calib_iters/batch_num` 决定）
4. `amct.save_model(...)` 导出 `fake_quant` 和 `deploy` 模型

## 2. 运行方式

最小可运行示例（快速验证）：

```bash
python3 /workspace/quantization/manual_quant_perscar.py \
  --batch-num 1 \
  --batch-size 4 \
  --calib-samples 800
```

常用完整示例：

```bash
python3 /workspace/quantization/manual_quant_perscar.py \
  --model /workspace/models/PersonCarAnimal_od-v3-x-bestp-d4-416-768_20251203.onnx \
  --calibration-dir /workspace/datasets/person_car_animal-1101 \
  --output-dir /workspace/quantization/out/manual_quant_perscar_result \
  --batch-num 1 \
  --batch-size 4 \
  --calib-iters 200 \
  --input-width 768 \
  --input-height 416
```

## 3. 关键参数

- `--batch-num`：AMCT 要求的最小校准批次数（不等于总校准样本）
- `--batch-size`：每个批次图片数
- `--calib-iters`：实际校准前向迭代数（`0` 表示进入“全量图片”默认模式）
- `--calib-samples`：目标校准样本数（优先级高于 `calib-iters`）
- `--activation-offset / --no-activation-offset`：是否启用 activation offset
- `--skip-layers`：逗号分隔的“跳过量化节点名”
- `--nuq --nuq-config <file>`：启用非均匀量化并指定配置文件

## 4. skip_layers 使用方法

### 4.1 不跳过任何层

默认就是空列表（`skip_layers = []`），对应命令行为不传 `--skip-layers`：

```bash
python3 /workspace/quantization/manual_quant_perscar.py
```

### 4.2 指定跳过量化节点

通过 `--skip-layers` 传入节点名，多个节点用逗号分隔：

```bash
python3 /workspace/quantization/manual_quant_perscar.py \
  --skip-layers "/model.1/conv/Conv,/model.2/conv/Conv"
```

脚本内部会解析为：

```python
skip_layers = ["/model.1/conv/Conv", "/model.2/conv/Conv"]
```

### 4.3 如何查看可用节点名

可用以下命令打印 ONNX 图里的节点名（前 200 个）：

```bash
python3 - <<'PY'
import onnx
m = onnx.load('/workspace/models/PersonCarAnimal_od-v3-x-bestp-d4-416-768_20251203.onnx')
for i, n in enumerate(m.graph.node[:200]):
    print(i, n.name, n.op_type)
PY
```

当前模型中可见的节点名示例：

- `/model.1/conv/Conv`
- `/model.2/conv/Conv`
- `/model.3/cv1/conv/Conv`

把这些名称按需放入 `--skip-layers` 即可跳过对应节点量化。

## 5. 关于校准数据量

当前脚本已将 `batch_num` 与校准数据量解耦。实际校准迭代数规则：

1. 若设置了 `--calib-samples (>0)`，则优先按该样本数执行校准
2. 否则若设置了 `--calib-iters (>0)`，则样本数为 `calib_iters * batch_size`
3. 否则默认使用“校准集全部图片跑一遍”
4. 最终保证批次数 `>= batch_num`（满足 AMCT 最小要求，不足时会循环复用图片）

总校准样本数以日志里打印的 `Calibration samples used` 为准。

## 6. 输出文件

运行完成后，默认会生成：

- `config.json`：量化配置
- `record.txt`：scale/offset 记录
- `modified_model.onnx`：量化中间模型
- `<prefix>_fake_quant_model.onnx`：仿真量化模型（用于 ONNX 精度验证）
- `<prefix>_deploy_model.onnx`：部署模型（用于后续 ATC）
