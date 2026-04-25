# Copyright (c) 2026 Siriusquirrel
# Part of the SongGeneration-v2-Large-16GB-Fork

import argparse
from codeclm.models.lm_levo import LmModel, get_lm_model
from codeclm.models.llama.modeling_llama import LlamaRotaryEmbedding
from omegaconf import OmegaConf
from pathlib import Path
import torch
import gc

# ---------------------------------------------------------
# Argumente
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Convert conditions to tokens")
    parser.add_argument("--conditions", type=str, default="out/batch_conditions.pt",  help="Path to input conditions file")
    parser.add_argument("--out",   type=str, default="out/batch_tokens.pt", help="Path for output token file")
    return parser.parse_args()

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    args = parse_args()

    print("Loading conditions file…")
    path = Path(args.conditions)
    if not path.exists():
        raise FileNotFoundError(f"Conditions file not found: {path}")
    conditions = torch.load(path, map_location="cpu", weights_only=False)

    print("Initialising Audio Transformer…")
    cfg = OmegaConf.load('ckpt/songgeneration/config.yaml')
    max_duration = cfg.lyric_processor.max_dur
    with torch.device("meta"):
        audiolm = get_lm_model(cfg, version = 'v2')
    with torch.no_grad():
        checkpoint = torch.load('ckpt/songgeneration/model_v2_large_fp16_new_data_structure.pt', map_location='cpu', mmap=True, weights_only=True)
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

    target_len = cfg.lyric_processor.prompt_len * cfg.audio_tokenizer_frame_rate
    for item in conditions:
        print(f"Processing song id: {item['idx']}")

        out_file = Path(args.out) / f"{item['idx']}_tokens.pt"
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
            tokens = audiolm.generate(
                texts=[item["gt_lyric"]],
                descriptions=[item.get('descriptions', '')],
                audio_qt_embs=input_embeds,
                max_gen_len=int(max_duration * 25),
                temp=temp,
                top_k=top_k,
                top_p=top_p,
                cfg_coef=cfg_coef,
                record_window=record_window
            )

        actual_len = tokens.shape[-1]
        last_tokens = tokens[0, :, -5:]
        ckpt = {}
        has_raw_wavs = item['raw_wavs']
        ckpt['raw_wavs'] = has_raw_wavs
        if has_raw_wavs == True:
            raw_vocals = item['raw_vocal_wav']
            raw_bgm    = item['raw_bgm_wav']
            ckpt['raw_vocal_wav'] = raw_vocals
            ckpt['raw_bgm_wav']   = raw_bgm
        ckpt['tokens'] = tokens.cpu().clone()
        print(f"Saving tokens for {item['idx']} to {out_file} ...")
        print(f"Tokens generated: {actual_len} of {int(max_duration * 25)}")
        print(f"Last 5 tokens: {last_tokens}")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, out_file)
        del tokens, input_embeds
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
