# Copyright (c) 2026 Siriusquirrel
# Part of the SongGeneration-v2-Large-16GB-Fork

from codeclm.models.lmlevo_cb0 import LmModel_cb0, get_lm_model_cb0
from omegaconf import OmegaConf
from pathlib import Path
import argparse
import torch
import numpy as np
import random
import gc
import zstandard as zstd
import io

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

    print("Loading conditions file…")
    path = Path("./out") / f"{args.batch}.cond.pt"
    if not path.exists():
        raise FileNotFoundError(f"Conditions file not found: {path}")
    conditions = torch.load(path, map_location="cpu", weights_only=False)

    print("Initialising Audio Transformer…")
    cfg = OmegaConf.load('ckpt/songgeneration/config.yaml')
    max_duration = int(cfg.lyric_processor.max_dur * cfg.audio_tokenizer_frame_rate)
    with torch.device("meta"):
        audiolm = get_lm_model_cb0(cfg, version = 'v2')
    with torch.no_grad():
        checkpoint = torch.load('ckpt/songgeneration/model_ht_v2_large_fp16_new_data_structure.pt', map_location='cpu', mmap=True, weights_only=True)
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

    out_path = Path(f"./out/{args.batch}")
    target_len = cfg.lyric_processor.prompt_len * cfg.audio_tokenizer_frame_rate
    for item in conditions:
        print(f"Processing song id: {item['idx']}")

        out_file = out_path / f"{item['idx']}.pt.zst"
        if out_file.exists():
            print(f"Skipping because {out_file} already exists")
            continue

        pmt_wav   = item['pmt_wav']
        vocal_wav = item['vocal_wav']
        bgm_wav   = item['bgm_wav']
        input_embeds = torch.cat([pmt_wav, vocal_wav, bgm_wav], dim=1).long()
        curr_len = input_embeds.shape[-1]
        if curr_len < target_len:
            padding = torch.full((1, 3, target_len - curr_len), 16385).long()
            input_embeds = torch.cat([input_embeds, padding], dim=-1)
        input_embeds = input_embeds.cuda()

        temp, top_k, top_p, cfg_coef, record_window = 0.8, 200, 0.9, 1.5, 50
        if "parameters" in item:
            params_list = [p.strip() for p in item["parameters"].split(',')]
            for p in params_list:
                if ":" in p:
                    key, val = p.split(":")
                    if   key == "temp": temp = float(val)
                    elif key == "top_k": top_k = int(val)
                    elif key == "top_p": top_p = float(val)
                    elif key == "cfg_coef": cfg_coef = float(val)
                    elif key == "record_window": record_window = int(val)
        print(f"Generating token file… (temp={temp}, top_k={top_k}, top_p={top_p}, cfg_coef={cfg_coef}, record_window={record_window})")

        with torch.inference_mode():
            out_dict = audiolm.generate(
                texts=[item["gt_lyric"]],
                descriptions=[item.get('descriptions', '')],
                audio_qt_embs=input_embeds,
                max_gen_len=max_duration,
                temp=temp,
                top_k=top_k,
                top_p=top_p,
                cfg_coef=cfg_coef,
                record_window=record_window
            )

        tokens = out_dict["tokens"]
        actual_len = tokens.shape[-1]
        last_tokens = tokens[-5:]

        has_raw_wavs = item['raw_wavs']
        out_dict['raw_wavs'] = has_raw_wavs
        if has_raw_wavs == True:
            raw_vocals = item['raw_vocal_wav']
            raw_bgm    = item['raw_bgm_wav']
            out_dict['raw_vocal_wav'] = raw_vocals
            out_dict['raw_bgm_wav']   = raw_bgm

        print(f"Saving tokens for {item['idx']} to {out_file} ...")
        print(f"Tokens generated: {actual_len} of {max_duration}")
        print(f"Last 5 tokens: {last_tokens}")
        buffer = io.BytesIO()
        torch.save(out_dict, buffer)
        cctx = zstd.ZstdCompressor(level=3)
        compressed_data = cctx.compress(buffer.getvalue())
        out_file.write_bytes(compressed_data)
        del tokens, input_embeds, out_dict, buffer, compressed_data
        torch.cuda.empty_cache()
        gc.collect()

def parse_args():
    parser = argparse.ArgumentParser(description='Script generating CB0 tokens from conditions')
    parser.add_argument('--batch', type=str, required=True,
                      help='Basename of batch to generate tokens for')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
