# Copyright (c) 2026 Siriusquirrel
# Part of the SongGeneration-v2-Large-16GB-Fork

import torch
import gc

output_file = "model_v2_large_fp16_new_data_structure.pt"

with torch.no_grad():
    checkpoint = torch.load('model_v2_large_fp16.pt', map_location='cpu', mmap=True, weights_only=True)
prefix = 'audiolm.'
state_dict = {}
for k, v in checkpoint.items():
    if k.startswith(prefix):
        state_dict[k[len(prefix):]] = v
del checkpoint
gc.collect()
# In-Place Fusing (Gate + Up zu Fused verschmelzen)
# Wir nutzen eine Liste der Keys, um während der Iteration löschen zu können
for k in list(state_dict.keys()):
    if ".model.embed_tokens.weight" in k:
        state_dict.pop(k)
    if k == "out_norm.weight" or k == "out_norm.bias":
        state_dict.pop(k)
    if "gate_proj.weight" in k:
        up_key = k.replace("gate_proj.weight", "up_proj.weight")
        fused_key = k.replace("gate_proj.weight", "fused_gate_up_proj.weight")
        # Die beiden alten Tensoren zusammenfügen
        # .pop() löscht sie direkt aus dem Dict und gibt den Tensor zurück
        gate_w = state_dict.pop(k)
        up_w = state_dict.pop(up_key)
        state_dict[fused_key] = torch.cat([gate_w, up_w], dim=0)
    if "q_proj.weight" in k:
        k_key = k.replace("q_proj.weight", "k_proj.weight")
        v_key = k.replace("q_proj.weight", "v_proj.weight")
        fused_key = k.replace("q_proj.weight", "fused_qkv_proj.weight")
        # Wichtig: Reihenfolge muss Q, K, V sein!
        q_w = state_dict.pop(k)
        k_w = state_dict.pop(k_key)
        v_w = state_dict.pop(v_key)
        state_dict[fused_key] = torch.cat([q_w, k_w, v_w], dim=0)
        # Falls Bias vorhanden ist (config.attention_bias=True)
        q_b_key = k.replace("weight", "bias")
        if q_b_key in state_dict:
            k_b_key = q_b_key.replace("q_proj", "k_proj")
            v_b_key = q_b_key.replace("q_proj", "v_proj")
            fused_b_key = q_b_key.replace("q_proj", "fused_qkv_proj")
            q_b = state_dict.pop(q_b_key)
            k_b = state_dict.pop(k_b_key)
            v_b = state_dict.pop(v_b_key)
            state_dict[fused_b_key] = torch.cat([q_b, k_b, v_b], dim=0)

torch.save(state_dict, output_file)
