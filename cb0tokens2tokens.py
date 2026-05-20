# Copyright (c) 2026 Siriusquirrel
# Part of the SongGeneration-v2-Large-16GB-Fork

from codeclm.models.lmlevo_cb12 import LmModel_cb12, get_lm_model_cb12
from omegaconf import OmegaConf
from pathlib import Path
import argparse
import torch
import numpy as np
import random
import gc
import io
import zstandard as zstd

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Wichtig für exakte Reproduzierbarkeit auf der GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main(args):
#    set_seed()

    input_dir = Path(f"./out/{args.batch}")
    if not input_dir.exists():
        print(f"Directory {input_dir} does not exist.")
        return

    zst_files = list(input_dir.glob("*.pt.zst"))
    if not zst_files:
        print(f"No .pt.zst files found in {input_dir}.")
        return
    print(f"{len(zst_files)} files found. Processing...")
    dctx = zstd.ZstdDecompressor()

    print("Initialising Audio Transformer…")
    cfg = OmegaConf.load('ckpt/songgeneration/config.yaml')
    with torch.device("meta"):
        audiolm = get_lm_model_cb12(cfg)
    with torch.no_grad():
        checkpoint = torch.load('ckpt/songgeneration/model_st_v2_large_fp16_new_data_structure.pt', map_location='cpu', mmap=True, weights_only=True)
    audiolm.load_state_dict(checkpoint, strict=True, assign=True)
    del checkpoint
    gc.collect()

    for name, buf in audiolm.named_buffers():
        if buf.is_meta:
            print(f"!!! CRITICAL: Buffer '{name}' is still a meta-tensor!")

    for name, param in audiolm.named_parameters():
        if param.is_meta:
            print(f"!!! CRITICAL: Parameter '{name}' is still a meta-tensor!")

    audiolm.cuda().eval()
    gc.collect()
    torch.cuda.empty_cache()

    for file in zst_files:
        print(f"Processing song: {file.name}")

        out_file = input_dir / file.with_suffix('').name
        if out_file.exists():
            print(f"Skipping because {out_file} already exists")
            continue

        compressed_data = file.read_bytes()
        decompressed_data = dctx.decompress(compressed_data)
        cb0_dict = torch.load(io.BytesIO(decompressed_data), weights_only=False)
        out_dict = audiolm.generate(cb0_dict)
        tokens = out_dict["tokens"]
        actual_len = tokens.shape[-1]
        last_tokens = tokens[:, -5:]
        print(f"Saving tokens for {file.name} to {out_file} ...")
        print(f"Tokens generated: {actual_len}")
        print(f"Last 5 tokens: {last_tokens}")
        torch.save(out_dict, out_file)
        del tokens, compressed_data, decompressed_data, cb0_dict, out_dict
        torch.cuda.empty_cache()
        gc.collect()

def parse_args():
    parser = argparse.ArgumentParser(description='Script generating CB12 tokens from CB0')
    parser.add_argument('--batch', type=str, required=True,
                      help='Basename of batch to generate tokens for')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
