# PersonCarAnimal 手工量化说明（AMCT ONNX）

## 1. 目标与文件

- 待量化模型：`/workspace/models/PersonCarAnimal_od-v3-x-bestp-d4-416-768_20251203.onnx`
- 校准数据集：`/workspace/datasets/person_car_animal-1101`
- 手工量化脚本：`/workspace/quantization/manual_quant_perscar.py`
- 默认输出目录：`/workspace/quantization/out/manual_quant_perscar_result`

脚本已按 AMCT 手工量化流程实现：

1. `amct.create_quant_config(...)`
2. `amct.quantize_model(...)`
3. 对 `modified_model.onnx` 执行校准前向（`batch_num` 轮）
4. `amct.save_model(...)` 导出 `fake_quant` 和 `deploy` 模型

## 2. 运行方式

最小可运行示例（快速验证）：

```bash
python3 /workspace/quantization/manual_quant_perscar.py \
  --batch-num 1 \
  --batch-size 1
```

常用完整示例：

```bash
python3 /workspace/quantization/manual_quant_perscar.py \
  --model /workspace/models/PersonCarAnimal_od-v3-x-bestp-d4-416-768_20251203.onnx \
  --calibration-dir /workspace/datasets/person_car_animal-1101 \
  --output-dir /workspace/quantization/out/manual_quant_perscar_result \
  --batch-num 8 \
  --batch-size 16 \
  --input-width 768 \
  --input-height 416
```

## 3. 关键参数

- `--batch-num`：校准批次数（AMCT 要求校准前向次数 `>= batch_num`）
- `--batch-size`：每个批次图片数
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

## 5. 输出文件

运行完成后，默认会生成：

- `config.json`：量化配置
- `record.txt`：scale/offset 记录
- `modified_model.onnx`：量化中间模型
- `<prefix>_fake_quant_model.onnx`：仿真量化模型（用于 ONNX 精度验证）
- `<prefix>_deploy_model.onnx`：部署模型（用于后续 ATC）
