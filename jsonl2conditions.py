# Copyright (c) 2026 Siriusquirrel
# Part of the SongGeneration-v2-Large-16GB-Fork

import argparse
from codeclm.tokenizer.audio_tokenizer import Flow1dVAE1rvq, Flow1dVAESeparate
from third_party.demucs.models.pretrained import get_model_from_yaml
from third_party.demucs.models.apply import apply_model
import json
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path
import re
import sound
import time
import torch
import torchaudio

auto_prompt_type = ['Pop', 'Latin', 'Rock', 'Electronic', 'Metal', 'Country', 'R&B/Soul', 'Ballad', 'Jazz', 'World', 'Hip-Hop', 'Funk', 'Soundtrack','Auto']

def check_language_by_text(text):
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    english_pattern = re.compile(r'[a-zA-Z]')
    chinese_count = len(re.findall(chinese_pattern, text))
    english_count = len(re.findall(english_pattern, text))
    chinese_ratio = chinese_count / len(text)
    english_ratio = english_count / len(text)
    if chinese_ratio >= 0.2:
        return "zh"
    elif english_ratio >= 0.5:
        return "en"
    else:
        return "en"

def generate(args):
    input_jsonl = args.input_jsonl
    save_dir = args.save_dir
    cfg = OmegaConf.load('ckpt/songgeneration/config.yaml')
    max_duration = cfg.lyric_processor.max_dur
    gen_type = args.generate_type
    auto_prompt = torch.load('tools/new_auto_prompt.pt')
    demucs_model = get_model_from_yaml('ckpt/htdemucs/htdemucs.yaml', 'ckpt/htdemucs/htdemucs.pth').eval().cuda()
    _, _, ckpt_name = cfg.audio_tokenizer_checkpoint.partition('_')
    audio_tokenizer = Flow1dVAE1rvq(path=ckpt_name, cfg=cfg).eval().cuda()
    _, _, ckpt_name_sep = cfg.audio_tokenizer_checkpoint_sep.partition('_')
    seperate_tokenizer = Flow1dVAESeparate(path=ckpt_name_sep, cfg=cfg).eval().cuda()

    with open(input_jsonl, "r") as fp:
        lines = fp.readlines()

    new_items = []
    for line in lines:
        item = json.loads(line)
        item = {k.lower(): v for k, v in item.items()}
        item["idx"] = f"{item['idx']}"
        desc_text = item.get('descriptions', item.get('description', None))
        item.pop('description', None)
        if desc_text is not None:
            desc_text = desc_text.lower()
#            if gen_type == 'bgm':
#                desc_text = '[Musicality-very-high]' + ', ' + '[Pure-Music]' + ', ' + desc_text
#            else:
#                desc_text = '[Musicality-very-high]' + ', ' + desc_text
            item['descriptions'] = desc_text
        target_wav_name = f"{save_dir}/{item['idx']}.wav"
        item["wavout_path"] = target_wav_name
        item["raw_wavs"] = False
        if "prompt_audio_path" in item:
            audiofile = Path(input_jsonl).parent / item['prompt_audio_path']
            assert audiofile.exists(), f"prompt_audio_path {audiofile} not found"
            assert 'auto_prompt_audio_type' not in item, f"auto_prompt_audio_type and prompt_audio_path cannot be used together"
            assert audiofile.suffix.lower() == '.wav', f"Only wav files supported as audio prompts"
            item['raw_pmt_wav']   = audiofile
            item['raw_vocal_wav'] = audiofile.with_name(audiofile.stem + '_vocal.wav')
            item['raw_bgm_wav']   = audiofile.with_name(audiofile.stem + '_bgm.wav')
            item["raw_wavs"] = True
            a, sr = sound.load_audio(audiofile)
            if (sr != 48000):
                print(f"Resampling {audiofile} to 48000 Hz")
                a = torchaudio.functional.resample(a, sr, 48000)
            target_pmt_samples = 48000 * 10 # 10 seconds 48kHz
            a = a[:, :target_pmt_samples]
            if a.shape[0] == 1:
                a = a.repeat(2, 1)
            pmt_audio=a.to("cuda")
            if not Path(item['raw_vocal_wav']).exists():
                print(f"Separating vocals from bgm for {audiofile}...")
                with torch.no_grad():
                    # [B, sources, C, T]
                    sources = apply_model(demucs_model, pmt_audio.unsqueeze(0), device="cuda", shifts=1, split=True, overlap=0.25, progress=False)[0]
                    # Index 3 = Vocals
                    vocal_audio = sources[3]
                    # BGM = Original - Vocals
                    bgm_audio = pmt_audio - vocal_audio

                sound.save_audio(item['raw_vocal_wav'], vocal_audio)
                sound.save_audio(item['raw_bgm_wav'], bgm_audio)
                print(f"Success!\n -> {item['raw_vocal_wav']}\n -> {item['raw_bgm_wav']}")
            else:
                vocal_audio, _ = sound.load_audio(item['raw_vocal_wav'])
                bgm_audio, _   = sound.load_audio(item['raw_bgm_wav'])

            with torch.no_grad():
                # Erzeugt [B, 1, T] - Layer 0
                pmt_wav, _ = audio_tokenizer.encode(pmt_audio.cuda())
                # Erzeugt codes_vocal [B, 1, T] und codes_bgm [B, 1, T]
                vocal_wav, bgm_wav = seperate_tokenizer.encode(vocal_audio.cuda(), bgm_audio.cuda())

            melody_is_wav = False
        elif "auto_prompt_audio_type" in item:
            audio_type = item["auto_prompt_audio_type"]
            assert audio_type in auto_prompt_type, f"auto_prompt_audio_type {item['auto_prompt_audio_type']} not found"
            lang = check_language_by_text(item['gt_lyric'])
#            prompt_idx = np.random.randint(0, len(auto_prompt[item["auto_prompt_audio_type"]][lang]))
            prompt_idx = auto_prompt_type.index(audio_type)
            prompt_token = auto_prompt[audio_type][lang][prompt_idx]
            pmt_wav = prompt_token[:,[0],:]
            vocal_wav = prompt_token[:,[1],:]
            bgm_wav = prompt_token[:,[2],:]
            melody_is_wav = False
        else:
            # Fall: melody_is_wav = True (Rein Melodie-basierte Führung) nicht implementiert
            target_len = 250 # 10s @ 25fps
            if "melody_audio_path" in item and Path(item['melody_audio_path']).exists():
                with torch.no_grad():
                    melody_audio, sr = sound.load_audio(item['melody_audio_path'])
                    # Layer 0: Melodie-Tokens extrahieren
                    pmt_wav, _ = audio_tokenizer.encode(melody_audio.cuda())
                    # Padding/Truncate auf exakt target_len (Sicherheits-Check für den Transformer)
                    if pmt_wav.shape[-1] < target_len:
                        pad = torch.full((1, 1, target_len - pmt_wav.shape[-1]), 16385, device='cuda').long()
                        pmt_wav = torch.cat([pmt_wav, pad], dim=-1)
                    else:
                        pmt_wav = pmt_wav[..., :target_len]
            else:
                # Falls gar kein Audio-Input da ist: Komplett leerer Anker mit Spezial-Token 16385
                pmt_wav = torch.full((1, 1, target_len), 16385, device='cuda').long()

            # Layer 1 & 2 (Vocals/BGM) werden hier mit Stille (16385) initialisiert,
            # da keine Stems für eine reine Melodie-Führung vorliegen.
            vocal_wav = torch.full((1, 1, target_len), 16385).long()
            bgm_wav = torch.full((1, 1, target_len), 16385).long()
            melody_is_wav = True

        # Hier werden direkt die Tokenstreams gespeichert
        item['pmt_wav'] = pmt_wav
        item['vocal_wav'] = vocal_wav
        item['bgm_wav'] = bgm_wav
        item['melody_is_wav'] = melody_is_wav
        new_items.append(item)

    # --- ABSPEICHERN DER GESAMTEN LISTE ---
    conditions_file = Path(save_dir) / "batch_conditions.pt"
    torch.save(new_items, conditions_file)
    print(f"{len(new_items)} song conditions written to {conditions_file}")

def parse_args():
    parser = argparse.ArgumentParser(description='Conditions Generation Script')

    parser.add_argument('--input_jsonl', type=str, required=True,
                      help='Path to input JSONL file containing generation tasks')
    parser.add_argument('--save_dir', type=str, default="./out/",
                      help='Directory to save generated condition files (default: "./out/")')
    parser.add_argument('--generate_type', type=str, default='mixed',
                      help='Type of generation: "vocal" or "bgm" or "separate" or "mixed" (default: "mixed")')
    return parser.parse_args()

if __name__ == "__main__":
    np.random.seed(int(time.time()))

    args = parse_args()
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        reserved = torch.cuda.memory_reserved(device)
        total = torch.cuda.get_device_properties(device).total_memory
        res_mem = (total - reserved) / 1024 / 1024 / 1024
        print(f"reserved memory: {res_mem}GB")
        generate(args)
    else:
        print("CUDA is not available")
        exit()
