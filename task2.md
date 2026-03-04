1.ascend_quantization手工量化官方案例代码/workspace/quantization/zz_mual_official_sample.py
2.校准数据集/workspace/datasets/person_car_animal-1101
3.评测数据集/workspace/AlgoServerScript/datasets/person_car
4.需要量化的模型/workspace/models/PersonCarAnimal_od-v3-x-bestp-d4-416-768_20251203.onnx
5.amct手工量化工具介绍、用法、api说明文档/workspace/CANN社区版 8.5.0 AMCT模型压缩工具用户指南 01.md（内容较大，若有需要读取对应部分内容即可）

参照ascend_quantization手工量化官方案例，实现对/workspace/models/PersonCarAnimal_od-v3-x-bestp-d4-416-768_20251203.onnx模型的手工量化/workspace/quantization/manual_quant_perscar.py代码完善可运行。
生成说明文档mual.md，另外包含如何skip_layers = []参数使用，指定跳过量化某些节点
用中文回复。