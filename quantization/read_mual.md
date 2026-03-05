# PersonCarAnimal 手工量化说明（AMCT ONNX）

## 1. 目标与文件

- 脚本：`/workspace/quantization/manual_quant_perscar.py`
- 待量化模型（默认）：`/workspace/models/PersonCarAnimal_od-v3-x-bestp-d4-416-768_20251203.onnx`
- 校准数据集（默认）：`/workspace/datasets/person_car_animal-1101`
- 输出目录（默认）：`/workspace/quantization/out/manual_quant_perscar_result`

## 2. 量化流程

脚本按 AMCT 手工量化三阶段执行：

1. `amct.create_quant_config(...)`
2. `amct.quantize_model(...)` 生成 `modified_model.onnx`
3. 对 `modified_model.onnx` 做校准前向后执行 `amct.save_model(...)`

## 3. 运行方式

### 3.1 快速示例

```bash
python3 /workspace/quantization/manual_quant_perscar.py \
  --batch-num 1 \
  --batch-size 4 \
  --calib-samples 800
```

### 3.2 常用完整示例

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

## 4. 关键参数

- `--batch-num`：AMCT 要求的最小校准批次数（默认 `8`）
- `--batch-size`：每批图片数（默认 `8`）
- `--calib-samples`：目标校准样本数，`>0` 时优先于 `--calib-iters`（默认 `1101`）
- `--calib-iters`：实际校准迭代数（默认 `0`）
- `--input-width / --input-height`：模型输入尺寸（默认 `768x416`）
- `--activation-offset` / `--no-activation-offset`：是否启用 activation offset（默认启用）
- `--nuq` + `--nuq-config`：非均匀量化
- `--skip-layers`：逗号分隔的跳过量化层名

## 5. 校准迭代规则

当前脚本中校准迭代数 `iterations` 计算逻辑：

1. 若 `calib-samples > 0`，`iterations = ceil(calib-samples / batch-size)`
2. 否则若 `calib-iters > 0`，`iterations = calib-iters`
3. 否则 `iterations = batch-num`
4. 最终 `iterations = max(iterations, batch-num)`

实际参与前向样本数约为 `iterations * batch-size`。

## 6. skip-layers 说明

可通过命令行传入：

```bash
python3 /workspace/quantization/manual_quant_perscar.py \
  --skip-layers "/model.1/conv/Conv,/model.2/conv/Conv"
```

查看节点名示例：

```bash
python3 - <<'PY'
import onnx
m = onnx.load('/workspace/models/PersonCarAnimal_od-v3-x-bestp-d4-416-768_20251203.onnx')
for i, n in enumerate(m.graph.node[:200]):
    print(i, n.name, n.op_type)
PY
```

注意：当前脚本内部对 `skip_layers` 存在固定赋值逻辑，若你希望完全由命令行控制，需要同步修改脚本代码中的 `skip_layers` 定义。

## 7. 输出文件

默认输出目录下主要文件：

- `config.json`：量化配置
- `record.txt`：scale/offset 记录
- `modified_model.onnx`：量化中间模型
- `<模型名>_fake_quant_model.onnx`：仿真量化模型
- `<模型名>_deploy_model.onnx`：部署模型（供 ATC 转换）

## 8. 常见问题

- `no calibration images found`：校准目录为空或图片格式不支持。
- `batch-size must be > 0` / `batch-num must be > 0`：参数非法。
- 精度不理想：逐步增加回退层，或改用自动精度量化流程。
