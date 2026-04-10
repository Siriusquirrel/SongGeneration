# Original work Copyright (c) Tencent AI Lab
# Refactoring and modifications Copyright (c) 2026 Siriusquirrel
#
# Part of the SongGeneration-v2-Large-16GB-Fork
# Modifications: Completely rewrote sound2code and code2sound

import torch
from model_septoken import PromptCondAudioDiffusion
import math
from sat_1dvae_large import get_model
from safetensors.torch import load_file


class Tango:
    def __init__(self, \
        model_path, \
        cfg,
        layer_vocal=7,\
        layer_bgm=3,\
        device="cuda:0"):
        
        self.sample_rate = 48000
        self.device = device

        self.vae = get_model(cfg.vae_config, cfg.vae_model)
        self.vae = self.vae.to(device).eval()
        self.layer_vocal=layer_vocal
        self.layer_bgm=layer_bgm
        self.MAX_DURATION = cfg.lyric_processor.max_dur
        self.eos_id = cfg.lm.code_size - 1
        self.model = PromptCondAudioDiffusion()
        if model_path.endswith(".safetensors"):
            main_weights = load_file(model_path)
        else:
            main_weights = torch.load(model_path, map_location="cpu", mmap=True, weights_only=True)
        self.model.load_state_dict(main_weights, strict=False, assign=True)
        self.model.to(device)
        print("Successfully loaded checkpoint from:", model_path)
        self.model.eval()
        self.model.init_device_dtype(torch.device(device), torch.float32)

    @torch.inference_mode()
    def sound2code(self, orig_vocal, orig_bgm, batch_size=8):
        # 1. Preprocessing & Layout-Fix
        v_proc = self.preprocess_audio(orig_vocal.to(self.device)).squeeze(0)
        b_proc = self.preprocess_audio(orig_bgm.to(self.device)).squeeze(0)
        
        # Die tatsächliche Länge der Eingabe-Samples bestimmen
        actual_samples = min(v_proc.shape[-1], b_proc.shape[-1])
        # Die exakt benötigte Tokenlänge (z.B. 251 für 10s, 1501 für 60s)
        output_len = int(actual_samples / self.sample_rate * 25) + 1
        
        min_samples = int(40 * self.sample_rate)

        # 2. Internes Padding für die 40s-Modell-Fenster
        num_chunks = max(1, (actual_samples + min_samples - 1) // min_samples)
        total_target = num_chunks * min_samples
        
        # Nur für die Inferenz loopen (damit das Modell "Futter" hat)
        repeats = (total_target + actual_samples - 1) // actual_samples
        vocal = torch.tile(v_proc[:, :actual_samples], (1, repeats))[:, :total_target]
        bgm = torch.tile(b_proc[:, :actual_samples], (1, repeats))[:, :total_target]

        # 3. Batching & Inferenz
        v_input = vocal.reshape(num_chunks, 2, min_samples)
        b_input = bgm.reshape(num_chunks, 2, min_samples)

        v_list, b_list = [], []
        for i in range(0, num_chunks, batch_size):
            [cv, cb], _, _ = self.model.fetch_codes_batch(
                v_input[i : i + batch_size],
                b_input[i : i + batch_size],
                additional_feats=[],
                layer_vocal=self.layer_vocal,
                layer_bgm=self.layer_bgm
            )
            v_list.append(cv)
            b_list.append(cb)

        # 4. Rekonstruktion MIT PRÄZISEM SLICING
        # Das entfernt den Looping-Überhang und liefert exakt output_len zurück
        res_v = torch.cat(v_list, dim=0).permute(1, 0, 2).reshape(1, -1)[:, :output_len].unsqueeze(0)
        res_b = torch.cat(b_list, dim=0).permute(1, 0, 2).reshape(1, -1)[:, :output_len].unsqueeze(0)

        return res_v, res_b
    

    @torch.inference_mode()
    def code2sound(self, codes, prompt_vocal=None, prompt_bgm=None, duration=40, guidance_scale=1.5, num_steps=20, disable_progress=False, chunked=True, chunk_size=128):
        # 1. Setup & Ziel-Definition
        codes_vocal, codes_bgm = [c.to(self.device, non_blocking=True) for c in codes]

        # Fenster-Parameter (Tokens @ 25Hz)
        min_frames = duration * 25
        hop_frames = (min_frames // 4) * 3
        ovlp_frames = min_frames - hop_frames

        # Fenster-Parameter (Audio-Samples & Ziel-Dauer)
        # Basiert auf der reinen Eingabe vor dem Prompt-Handling
        target_len = int(codes_vocal.shape[-1] * self.sample_rate / 25)
        eos_find = (codes_vocal[0, 0] >= self.eos_id).nonzero(as_tuple=False)
        if eos_find.any():
            first_eos = eos_find.min().item()
            target_len = int(first_eos * self.sample_rate / 25)
        ovlp_samples = int(ovlp_frames * self.sample_rate / 25)

        # In-Context Initialisierung
        first_latent = torch.randn(codes_vocal.shape[0], min_frames, 64, device=self.device)
        first_latent_length = 0

        # 2. Prompt Handling (In-Context)
        if isinstance(prompt_vocal, torch.Tensor) and isinstance(prompt_bgm, torch.Tensor):
            p_v, p_b = prompt_vocal.to(self.device), prompt_bgm.to(self.device)
            if p_v.ndim == 3: p_v, p_b = p_v[0], p_b[0]
            elif p_v.ndim == 1: p_v, p_b = p_v.unsqueeze(0).repeat(2, 1), p_b.unsqueeze(0).repeat(2, 1)
            elif p_v.ndim == 2 and p_v.shape[0] == 1: p_v, p_b = p_v.repeat(2, 1), p_b.repeat(2, 1)

            s_10, s_20, s_30 = int(10*self.sample_rate), int(20*self.sample_rate), int(30*self.sample_rate)
            if p_v.shape[-1] < s_30:
                p_v, p_b = p_v[:, :s_10], p_b[:, :s_10]
            else:
                p_v, p_b = p_v[:, s_20:s_30], p_b[:, s_20:s_30]

            # VAE Encoding & Prompt-Tokens anhängen
            true_latent = self.vae.encode_audio(p_v + p_b).permute(0, 2, 1)
            print("true_latent shape:", true_latent.shape)
            first_latent[:, :true_latent.shape[1], :] = true_latent
            first_latent_length = true_latent.shape[1]

            f_v_code, f_b_code = self.sound2code(p_v, p_b)
            print("f_v_code:", f_v_code.shape)
            print("f_b_code:", f_b_code.shape)
            codes_vocal = torch.cat([f_v_code, codes_vocal], dim=-1)
            codes_bgm = torch.cat([f_b_code, codes_bgm], dim=-1)

        # 3. Robustes Padding auf Chunk-Raster (Exakt wie Original-Logik)
        curr_len = codes_vocal.shape[-1]
        needed_len = max(curr_len, min_frames)
        if (needed_len - ovlp_frames) % hop_frames > 0:
            needed_len = math.ceil((needed_len - ovlp_frames) / hop_frames) * hop_frames + ovlp_frames
        if curr_len < needed_len:
            repeats = (needed_len + curr_len - 1) // curr_len
            codes_vocal = torch.tile(codes_vocal, (1, 1, repeats))[:, :, :needed_len]
            codes_bgm = torch.tile(codes_bgm, (1, 1, repeats))[:, :, :needed_len]

        # 4. Inferenz-Loop (Sliding Window)
        latent_list = []
        spk_embeds = torch.zeros((1, 32, 1, 32), device=self.device)
        in_context = first_latent
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            # Wir stoppen, wenn kein volles Fenster mehr möglich ist
            for sinx in range(0, codes_vocal.shape[-1] - hop_frames, hop_frames):
                c_v_in = codes_vocal[:, :, sinx : sinx + min_frames]
                c_b_in = codes_bgm[:, :, sinx : sinx + min_frames]
                if c_v_in.shape[-1] < min_frames:
                    break
                if sinx == 0:
                    latents = self.model.inference_codes([c_v_in, c_b_in], spk_embeds, in_context, min_frames, additional_feats=[], incontext_length=first_latent_length, guidance_scale=guidance_scale, num_steps=num_steps, disable_progress=disable_progress, scenario='other_seg')
                else:
                    prev_lat_end = latent_list[-1][:, :, -ovlp_frames:].permute(0, 2, 1)
                    noise = torch.randn(prev_lat_end.shape[0], min_frames - ovlp_frames, 64, device=self.device, dtype=prev_lat_end.dtype)
                    in_context = torch.cat([prev_lat_end, noise], dim=1)
                    latents = self.model.inference_codes([c_v_in, c_b_in], spk_embeds, in_context, min_frames, additional_feats=[], incontext_length=ovlp_frames, guidance_scale=guidance_scale, num_steps=num_steps, disable_progress=disable_progress, scenario='other_seg')
                latent_list.append(latents)

        # 5. Audio Rekonstruktion (VAE + Cross-Fade)
        # Prompt-Latents hart abschneiden
        if latent_list:
            latent_list[0] = latent_list[0][:, :, first_latent_length:]

        # Sinus-Cosinus Rampe (Equal Power)
        grid = torch.linspace(0, math.pi / 2, ovlp_samples, device=self.device)
        fade_in, fade_out = torch.sin(grid).unsqueeze(0), torch.cos(grid).unsqueeze(0)

        output = None
        for i, lat in enumerate(latent_list):
            cur_audio = self.vae.decode_audio(lat.float(), chunked=chunked, chunk_size=chunk_size)
            if cur_audio.ndim == 3: cur_audio = cur_audio[0]

            if output is None:
                output = cur_audio
            else:
                # Cross-Fade auf GPU
                output[:, -ovlp_samples:] = output[:, -ovlp_samples:] * fade_out + cur_audio[:, :ovlp_samples] * fade_in
                output = torch.cat([output, cur_audio[:, ovlp_samples:]], dim=-1)

        return output[:, :target_len].cpu()


    @torch.inference_mode()
    def preprocess_audio(self, audios, threshold=0.8):
        """
        Normalisiert Audio und stellt sicher, dass das Format [B, 2, L] zurückkommt.
        Erkennt und repariert [2, 1, L] (Tencent-Style) oder [2, L].
        """
        # Schritt 1: Layout-Reparatur
        if audios.ndim == 2:
            # Erwartet [2, L], macht daraus [1, 2, L]
            assert audios.shape[0] == 2, f"2D Audio muss [2, L] sein, ist {audios.shape}"
            audios = audios.unsqueeze(0)
        elif audios.ndim == 3:
            # Erkennt [2, 1, L] und korrigiert zu [1, 2, L]
            if audios.shape[0] == 2 and audios.shape[1] == 1:
                audios = audios.transpose(0, 1)
            assert audios.shape[1] == 2, f"3D Audio muss [B, 2, L] sein, ist {audios.shape}"
        # Normalisation happens in model_septoken.py
#        max_vals = audios.abs().amax(dim=(1, 2)) 
#        scales = torch.clamp(max_vals / threshold, min=1.0)
#        return audios / scales.view(-1, 1, 1)
        return audios
    
    @torch.no_grad()
    def sound2sound(self, orig_vocal,orig_bgm, prompt_vocal=None,prompt_bgm=None, steps=50, disable_progress=False):
        codes_vocal, codes_bgm = self.sound2code(orig_vocal,orig_bgm)
        codes=[codes_vocal, codes_bgm]
        wave = self.code2sound(codes, prompt_vocal,prompt_bgm, guidance_scale=1.5, num_steps=steps, disable_progress=disable_progress)
        return wave
    
    def to(self, device=None, dtype=None, non_blocking=False):
        if device is not None:
            self.device = device
            self.model.device = device
        self.vae = self.vae.to(device, dtype, non_blocking)
        self.model = self.model.to(device, dtype, non_blocking)
        return self
