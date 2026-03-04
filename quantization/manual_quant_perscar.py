import os

import amct_onnx as amct

if __name__ == "__main__":
    save_path = '/workspace/quantization/out/manual_quant_personcar_result'
    config_file = os.path.join(save_path, 'config.yaml')
    ori_model = '/workspace/models/PersonCarAnimal_od-v3-x-bestp-d4-416-768_20251203.onnx'
    skip_layers = []
    batch_num = 1
    amct.create_quant_config(config_file=config_file,
                model_file=ori_model,
                skip_layers=skip_layers,
                batch_num=batch_num)
    
    record_file = os.path.join(save_path, 'record.txt')
    modified_model = os.path.join(save_path, 'modified_model.onnx')
    amct.quantize_model(config_file=config_file,
                        model_file=ori_model,
                        modified_onnx_file=modified_model,
                        record_file=record_file)
    
    quant_model_path = os.path.join(save_path, 'user_model')
    amct.save_model(modified_onnx_file=modified_model,
                    record_file=record_file,
                    save_path=quant_model_path)


