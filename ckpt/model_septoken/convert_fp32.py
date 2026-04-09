# Copyright (c) 2026 Siriusquirrel
# Part of the SongGeneration-v2-Large-16GB-Fork

import torch
from safetensors.torch import load_file, save_file

def convert_safetensors_to_fp32(input_path, output_path):
    print(f"Loading {input_path} (via mmap)...")
    weights = load_file(input_path, device="cpu")

    converted_weights = {}
    converted_count = 0

    print("Converting F16 tensors to F32...")
    for key, tensor in weights.items():
        # Alles was Half-Precision (FP16) ist, in Full-Precision (FP32) wandeln
        if tensor.dtype == torch.float16:
            converted_weights[key] = tensor.float()
            converted_count += 1
        else:
            converted_weights[key] = tensor

    print(f"Saving {output_path} ({converted_count} tensors adjusted)...")
    save_file(converted_weights, output_path)

convert_safetensors_to_fp32("model_septoken.safetensors", "model_septoken_fp32.safetensors")
