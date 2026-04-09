# Copyright (c) 2026 Siriusquirrel
# Part of the SongGeneration-v2-Large-16GB-Fork

import torch
import os

def convert_to_fp16(input_path, output_path):
    print(f"Loading {input_path} via mmap...")
    weights = torch.load(input_path, map_location="cpu", mmap=True, weights_only=True)

    actual_weights = weights.get('state_dict', weights) if isinstance(weights, dict) else weights
    converted_weights = {}
    converted_count = 0

    print("Scanning tensors...")
    for key, tensor in actual_weights.items():
        if isinstance(tensor, torch.Tensor):
            # Gezielte Konvertierung von FP32 und BF16 zu FP16
            if (tensor.dtype == torch.float32) or (tensor.dtype == torch.bfloat16):
                converted_weights[key] = tensor.half()
                converted_count += 1
            else:
                converted_weights[key] = tensor
        else:
            converted_weights[key] = tensor

    print(f"Saving {output_path} ({converted_count} tensors converted to FP16)...")
    torch.save(converted_weights, output_path)

# Beispielaufruf
convert_to_fp16("model_v2_large_orig.pt", "model_v2_large_fp16.pt")
