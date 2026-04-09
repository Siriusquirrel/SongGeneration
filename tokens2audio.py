# Copyright (c) 2026 Siriusquirrel
# Part of the SongGeneration-v2-Large-16GB-Fork

import torch
import sound
from pathlib import Path
from omegaconf import OmegaConf
from codeclm.tokenizer.audio_tokenizer import Flow1dVAESeparate

def main():
    cfg = OmegaConf.load('ckpt/songgeneration/config.yaml')

    ckpt_str = cfg.audio_tokenizer_checkpoint_sep
    print(f"Initialising decoder {ckpt_str}...")

    _, _, ckpt_str = ckpt_str.partition('_')
    tokenizer = Flow1dVAESeparate(ckpt_str, cfg)

    directory = Path("./out/batch_tokens.pt/")
    for tokens_path in directory.glob("*.pt"):
        out_wav = tokens_path.with_suffix(".wav")
        if out_wav.exists():
            print(f"Skipping {tokens_path} because {out_wav} already exists in {directory}")
            continue
        ckpt = torch.load(tokens_path, map_location="cpu", weights_only=False)
        has_raw_wavs = ckpt['raw_wavs']
        tokens = ckpt['tokens'].cuda()
        codes_vocal = tokens[:, 1:2, :]  # Shape: [B, 1, T]
        codes_bgm   = tokens[:, 2:3, :]  # Shape: [B, 1, T]
        print(f"Decoding {tokens.shape[-1]} tokens to audio...")
        if has_raw_wavs:
            raw_vocals = ckpt['raw_vocal_wav']
            raw_bgm    = ckpt['raw_bgm_wav']
            ap_voc, _  = sound.load_audio(raw_vocals)
            ap_bgm, _  = sound.load_audio(raw_bgm)
            print(f"...using {raw_vocals} and {raw_bgm}")
        else:
            ap_voc, ap_bgm = None, None
        with torch.inference_mode():
            audio_out = tokenizer.decode((codes_vocal, codes_bgm), prompt_vocal=ap_voc, prompt_bgm=ap_bgm, chunked=True, chunk_size=128)
        sound.save_audio(out_wav, audio_out)
        print(f"Audio file saved: {out_wav}")

if __name__ == "__main__":
    OmegaConf.register_new_resolver("eval", lambda x: eval(x))
    main()
