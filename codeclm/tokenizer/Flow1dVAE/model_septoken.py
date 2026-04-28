# Original work Copyright (c) Tencent AI Lab
# Refactoring and modifications Copyright (c) 2026 Siriusquirrel
#
# Part of the SongGeneration-v2-Large-16GB-Fork
# Modifications: Completely rewrote solve_euler and inference_codes to speedup processing

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from einops import repeat
from diffusers.utils.torch_utils import randn_tensor

from descript_quantize3 import ResidualVectorQuantize
from gpt2_rope2_time_new_correct_mask_noncasual_reflow import GPT2Model
from gpt2_config import GPT2Config
from musicfm.musicfm_model import MusicFMModel, MusicFMConfig
from normalization import Feature1DProcessor


class BASECFM(torch.nn.Module):
    def __init__(
        self,
        estimator
    ):
        super().__init__()
        self.sigma_min = 1e-4
        self.estimator = estimator

    def solve_euler(self, x, latent_mask_input,incontext_x, incontext_length, t_span, mu,attention_mask, guidance_scale):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_channels, mel_timesteps, n_feats)
        """
        dt = t_span[1:] - t_span[:-1]
        t = t_span[:-1]
        B, L, D = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0)
        is_incontext = (positions < incontext_length.unsqueeze(-1)).unsqueeze(-1)
        use_cfg = guidance_scale > 1.0

        if use_cfg:
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, attention_mask], 0)
            latent_mask_input = torch.cat([latent_mask_input, latent_mask_input], 0)
            incontext_x_input = torch.cat([incontext_x, incontext_x], 0)
            mu_input = torch.cat([torch.zeros_like(mu), mu], 0)
            batch_factor = 2
        else:
            incontext_x_input = incontext_x
            mu_input = mu
            batch_factor = 1

        x_next = x.clone()
        noise = x.clone()

        for ti, dti in zip(t, dt):
            target_incontext = (1 - (1 - self.sigma_min) * ti) * noise + ti * incontext_x
            x_next = torch.where(is_incontext, target_incontext, x_next)

            if use_cfg:
                current_x = torch.cat([x_next, x_next], 0)
            else:
                current_x = x_next
            model_input = torch.cat([latent_mask_input, incontext_x_input, mu_input, current_x], dim=2)
            timestep = ti.expand(batch_factor * B)

            v = self.estimator(inputs_embeds=model_input,
                            attention_mask=attention_mask,
                            time_step=timestep).last_hidden_state
            v = v[..., -D:]

            if use_cfg:
                v_uncond, v_cond = v.chunk(2, 0)
                v = v_uncond + guidance_scale * (v_cond - v_uncond)

            x_next = x_next + dti * v

        return x_next


class PromptCondAudioDiffusion(nn.Module):
    def __init__(
        self
    ):
        super().__init__()

        self.normfeat = Feature1DProcessor(dim=64)
        self.bestrq = MusicFMModel(MusicFMConfig())
        self.rsq48tobestrq = torchaudio.transforms.Resample(48000, 24000)
        self.rvq_bestrq_emb = ResidualVectorQuantize(input_dim = 1024, n_codebooks = 1, codebook_size = 16_384, codebook_dim = 32, quantizer_dropout = 0.0, stale_tolerance=200)
        self.rvq_bestrq_bgm_emb = ResidualVectorQuantize(input_dim = 1024, n_codebooks = 1, codebook_size = 16_384, codebook_dim = 32, quantizer_dropout = 0.0, stale_tolerance=200)
        self.zero_cond_embedding1 = nn.Parameter(torch.randn(32*32,))

        config = GPT2Config(n_positions=1000,n_layer=16,n_head=20,n_embd=2200,n_inner=4400)
        unet = GPT2Model(config)
        self.set_from = "random"
        self.cfm_wrapper = BASECFM(unet)
        self.mask_emb = torch.nn.Embedding(3, 24)
        print("septoken Flow-Matching Generator initialized from pretrain.")
        torch.cuda.empty_cache()

    def preprocess_audio(self, input_audios, threshold=0.9):
        assert len(input_audios.shape) == 2, input_audios.shape
        norm_value = torch.ones_like(input_audios[:,0])
        max_volume = input_audios.abs().max(dim=-1)[0]
        norm_value[max_volume>threshold] = max_volume[max_volume>threshold] / threshold
        return input_audios/norm_value.unsqueeze(-1)

    def extract_bestrq_embeds(self, input_audio_vocal_0,input_audio_vocal_1,layer):
        input_wav_mean = (input_audio_vocal_0 + input_audio_vocal_1) / 2.0
        input_wav_mean = self.bestrq(self.rsq48tobestrq(input_wav_mean), features_only = True)
        layer_results = input_wav_mean['layer_results']
        bestrq_emb = layer_results[layer]
        bestrq_emb = bestrq_emb.permute(0,2,1).contiguous()
        return bestrq_emb

    def init_device_dtype(self, device, dtype):
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def fetch_codes_batch(self, input_audios_vocal, input_audios_bgm, additional_feats,layer_vocal=7,layer_bgm=7):
        input_audio_vocal_0 = input_audios_vocal[:,0,:]
        input_audio_vocal_1 = input_audios_vocal[:,1,:]
        input_audio_vocal_0 = self.preprocess_audio(input_audio_vocal_0)
        input_audio_vocal_1 = self.preprocess_audio(input_audio_vocal_1)

        input_audio_bgm_0 = input_audios_bgm[:,0,:]
        input_audio_bgm_1 = input_audios_bgm[:,1,:]
        input_audio_bgm_0 = self.preprocess_audio(input_audio_bgm_0)
        input_audio_bgm_1 = self.preprocess_audio(input_audio_bgm_1)

        self.bestrq.eval()

        bestrq_emb = self.extract_bestrq_embeds(input_audio_vocal_0,input_audio_vocal_1,layer_vocal)
        bestrq_emb = bestrq_emb.detach()
        bestrq_emb_bgm = self.extract_bestrq_embeds(input_audio_bgm_0,input_audio_bgm_1,layer_bgm)
        bestrq_emb_bgm = bestrq_emb_bgm.detach()

        self.rvq_bestrq_emb.eval()
        quantized_bestrq_emb, codes_bestrq_emb, *_ = self.rvq_bestrq_emb(bestrq_emb) # b,d,t
        self.rvq_bestrq_bgm_emb.eval()
        quantized_bestrq_emb_bgm, codes_bestrq_emb_bgm, *_ = self.rvq_bestrq_bgm_emb(bestrq_emb_bgm) # b,d,t

        return [codes_bestrq_emb,codes_bestrq_emb_bgm]

    @torch.no_grad()
    @torch.compile(mode="reduce-overhead")
    def inference_codes(self, codes, true_latents, latent_length, incontext_length=127, 
                  guidance_scale=2, num_steps=20):
        device = self.device
        dtype = self.dtype

        codes_bestrq_emb,codes_bestrq_emb_bgm = codes

        batch_size = codes_bestrq_emb.shape[0]

        quantized_bestrq_emb,_,_=self.rvq_bestrq_emb.from_codes(codes_bestrq_emb)
        quantized_bestrq_emb_bgm,_,_=self.rvq_bestrq_bgm_emb.from_codes(codes_bestrq_emb_bgm)
        quantized_bestrq_emb = quantized_bestrq_emb.permute(0,2,1).contiguous()
        quantized_bestrq_emb_bgm = quantized_bestrq_emb_bgm.permute(0,2,1).contiguous()

        num_frames = quantized_bestrq_emb.shape[1]
        shape = (batch_size,  num_frames, 64)
        latents = randn_tensor(shape, generator=None, device=device, dtype=dtype)

        pos = torch.arange(num_frames, device=device).unsqueeze(0)
        latent_masks = (pos < latent_length).long() * 2
        latent_masks = torch.where(pos < incontext_length, torch.ones_like(latent_masks), latent_masks)

        quantized_bestrq_emb = (latent_masks > 0.5).unsqueeze(-1) * quantized_bestrq_emb \
            + (latent_masks < 0.5).unsqueeze(-1) * self.zero_cond_embedding1.reshape(1,1,1024)
        quantized_bestrq_emb_bgm = (latent_masks > 0.5).unsqueeze(-1) * quantized_bestrq_emb_bgm \
            + (latent_masks < 0.5).unsqueeze(-1) * self.zero_cond_embedding1.reshape(1,1,1024)
        true_latents = self.normfeat.project_sample(true_latents)
        incontext_latents = true_latents * (latent_masks == 1).unsqueeze(-1).float()

        attention_mask=(latent_masks > 0.5)
        B, L = attention_mask.size()
        attention_mask = attention_mask.view(B, 1, L)
        attention_mask = attention_mask * attention_mask.transpose(-1, -2)
        attention_mask = attention_mask.unsqueeze(1)
        latent_mask_input = self.mask_emb(latent_masks)

        additional_model_input = torch.cat([quantized_bestrq_emb,quantized_bestrq_emb_bgm],2)

        temperature = 1.0
        t_span = torch.linspace(0, 1, num_steps + 1, device=quantized_bestrq_emb.device)
        latents = self.cfm_wrapper.solve_euler(latents * temperature, latent_mask_input,incontext_latents, incontext_length, t_span, additional_model_input,attention_mask,  guidance_scale)

        positions = torch.arange(latents.shape[1], device=latents.device).unsqueeze(0) # Shape: [1, L]
        mask = (positions < incontext_length.unsqueeze(-1)).unsqueeze(-1)
        latents = torch.where(mask, incontext_latents, latents)

        latents = self.normfeat.return_sample(latents)
        return latents
