from tqdm import tqdm

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


class Feature1DProcessor(nn.Module):
    def __init__(self, dim: int = 100, power_std = 1., \
                 num_samples: int = 100_000, cal_num_frames: int = 600):
        super().__init__()

        self.num_samples = num_samples
        self.dim = dim
        self.power_std = power_std
        self.cal_num_frames = cal_num_frames
        self.register_buffer('counts', torch.zeros(1))
        self.register_buffer('sum_x', torch.zeros(dim))
        self.register_buffer('sum_x2', torch.zeros(dim))
        self.register_buffer('sum_target_x2', torch.zeros(dim))
        self.counts: torch.Tensor
        self.sum_x: torch.Tensor
        self.sum_x2: torch.Tensor

    @property
    def mean(self):
        mean = self.sum_x / self.counts
        if(self.counts < 10):
            mean = torch.zeros_like(mean)
        return mean

    @property
    def std(self):
        std = (self.sum_x2 / self.counts - self.mean**2).clamp(min=0).sqrt()
        if(self.counts < 10):
            std = torch.ones_like(std)
        return std

    @property
    def target_std(self):
        return 1

    def project_sample(self, x: torch.Tensor):
        assert x.dim() == 3
        if self.counts.item() < self.num_samples:
            self.counts += len(x)
            self.sum_x += x[:,:,0:self.cal_num_frames].mean(dim=(2,)).sum(dim=0)
            self.sum_x2 += x[:,:,0:self.cal_num_frames].pow(2).mean(dim=(2,)).sum(dim=0)
        rescale = (self.target_std / self.std.clamp(min=1e-12)) ** self.power_std  # same output size
        x = (x - self.mean.view(1, -1, 1)) * rescale.view(1, -1, 1)
        return x

    def return_sample(self, x: torch.Tensor):
        assert x.dim() == 3
        rescale = (self.std / self.target_std) ** self.power_std
        x = x * rescale.view(1, -1, 1) + self.mean.view(1, -1, 1)
        return x


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
        B = x.shape[0]

        if guidance_scale > 1.0:
            def double(z):
                return torch.cat([z, z], 0) if z is not None else None
            attention_mask = double(attention_mask)

        x_next = x.clone()
        noise = x.clone()

        for i in tqdm(range(len(dt))):
            ti = t[i]

            x_next[:, :incontext_length] = (
                (1 - (1 - self.sigma_min) * ti) * noise[:, :incontext_length] +
                ti * incontext_x[:, :incontext_length]
            )

            if guidance_scale > 1.0:
                model_input = torch.cat([
                    double(latent_mask_input),
                    double(incontext_x),
                    torch.cat([torch.zeros_like(mu), mu], 0),
                    double(x_next),
                ], dim=2)
                timestep = ti.expand(2 * B)
            else:
                model_input = torch.cat([
                    latent_mask_input, incontext_x, mu, x_next
                ], dim=2)
                timestep = ti.expand(B)

            v = self.estimator(inputs_embeds=model_input,
                            attention_mask=attention_mask,
                            time_step=timestep).last_hidden_state
            v = v[..., -x.shape[2]:]

            if guidance_scale > 1.0:
                v_uncond, v_cond = v.chunk(2, 0)
                v = v_uncond + guidance_scale * (v_cond - v_uncond)

            x_next = x_next + dt[i] * v

        return x_next


class PromptCondAudioDiffusion(nn.Module):
    def __init__(
        self
    ):
        super().__init__()

        self.normfeat = Feature1DProcessor(dim=64)
        self.rsq48towav2vec = torchaudio.transforms.Resample(48000, 16000)

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
        print("Transformer initialized from pretrain.")
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

    def extract_spk_embeds(self, input_audios):
        spk_embeds = self.xvecmodel(self.rsq48towav2vec(input_audios))
        spk_embeds = self.spk_linear(spk_embeds).reshape(spk_embeds.shape[0], 16, 1, 32)
        return spk_embeds

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

        if('spk' in additional_feats):
            self.xvecmodel.eval()
            spk_embeds = self.extract_spk_embeds(input_audios)
        else:
            spk_embeds = None

        return [codes_bestrq_emb,codes_bestrq_emb_bgm], [bestrq_emb,bestrq_emb_bgm], spk_embeds

    @torch.no_grad()
    def inference_codes(self, codes, spk_embeds, true_latents, latent_length, additional_feats,incontext_length=127, 
                  guidance_scale=2, num_steps=20,
                  disable_progress=True, scenario='start_seg'):
        classifier_free_guidance = guidance_scale > 1.0
        device = self.device
        dtype = self.dtype

        codes_bestrq_emb,codes_bestrq_emb_bgm = codes

        batch_size = codes_bestrq_emb.shape[0]

        quantized_bestrq_emb,_,_=self.rvq_bestrq_emb.from_codes(codes_bestrq_emb)
        quantized_bestrq_emb_bgm,_,_=self.rvq_bestrq_bgm_emb.from_codes(codes_bestrq_emb_bgm)
        quantized_bestrq_emb = quantized_bestrq_emb.permute(0,2,1).contiguous()
        quantized_bestrq_emb_bgm = quantized_bestrq_emb_bgm.permute(0,2,1).contiguous()
        if('spk' in additional_feats):
            spk_embeds = spk_embeds.repeat(1,1,quantized_bestrq_emb.shape[-2],1).detach()

        num_frames = quantized_bestrq_emb.shape[1]

        shape = (batch_size,  num_frames, 64)
        latents = randn_tensor(shape, generator=None, device=device, dtype=dtype)

        latent_masks = torch.zeros(latents.shape[0], latents.shape[1], dtype=torch.int64, device=latents.device)
        latent_masks[:,0:latent_length] = 2
        if(scenario=='other_seg'):
            latent_masks[:,0:incontext_length] = 1

        quantized_bestrq_emb = (latent_masks > 0.5).unsqueeze(-1) * quantized_bestrq_emb \
            + (latent_masks < 0.5).unsqueeze(-1) * self.zero_cond_embedding1.reshape(1,1,1024)
        quantized_bestrq_emb_bgm = (latent_masks > 0.5).unsqueeze(-1) * quantized_bestrq_emb_bgm \
            + (latent_masks < 0.5).unsqueeze(-1) * self.zero_cond_embedding1.reshape(1,1,1024)
        true_latents = true_latents.permute(0,2,1).contiguous()
        true_latents = self.normfeat.project_sample(true_latents)
        true_latents = true_latents.permute(0,2,1).contiguous()
        incontext_latents = true_latents * ((latent_masks > 0.5) * (latent_masks < 1.5)).unsqueeze(-1).float()
        incontext_length = ((latent_masks > 0.5) * (latent_masks < 1.5)).sum(-1)[0]

        attention_mask=(latent_masks > 0.5)
        B, L = attention_mask.size()
        attention_mask = attention_mask.view(B, 1, L)
        attention_mask = attention_mask * attention_mask.transpose(-1, -2)
        attention_mask = attention_mask.unsqueeze(1)
        latent_mask_input = self.mask_emb(latent_masks)

        if('spk' in additional_feats):
            additional_model_input = torch.cat([quantized_bestrq_emb,quantized_bestrq_emb_bgm, spk_embeds],2)
        else:
            additional_model_input = torch.cat([quantized_bestrq_emb,quantized_bestrq_emb_bgm],2)

        temperature = 1.0
        t_span = torch.linspace(0, 1, num_steps + 1, device=quantized_bestrq_emb.device)
        latents = self.cfm_wrapper.solve_euler(latents * temperature, latent_mask_input,incontext_latents, incontext_length, t_span, additional_model_input,attention_mask,  guidance_scale)

        latents[:,0:incontext_length,:] = incontext_latents[:,0:incontext_length,:]
        latents = latents.permute(0,2,1).contiguous()
        latents = self.normfeat.return_sample(latents)

        return latents
