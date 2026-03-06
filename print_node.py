import json
import fnmatch

pattern = "/model/decoder/decoder*"
config = "/workspace/quantization/out/manual_quant_smoke_result/config.json"

with open(config, "r") as f:
    data = json.load(f)

for key in data.keys():
    if fnmatch.fnmatch(key, pattern):
        # 生成带引号和逗号的列表
        result = [f'"{key}"' for key in data.keys() if fnmatch.fnmatch(key, pattern)]

# 打印结果
for item in result:
    print(f"{item},")