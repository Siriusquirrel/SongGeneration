# Copyright (c) 2026 Siriusquirrel
# Based on the Hugging Face Transformers implementation.
#
# Part of the SongGeneration-v2-Large-16GB-Fork
# 8-bit µ-law KV-caching, Fused QKV/MLP layers, SDPA integration, flash_attn_with_kvcache.
# Static KV-Storage-Blob allocation for zero fragmentation

import torch
import torch.nn.functional as F
from torch import nn
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import is_flash_attn_2_available
from .configuration_llama import LlamaConfig

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        h_f32 = hidden_states.to(torch.float32)
        variance = h_f32.pow(2).mean(-1, keepdim=True)
        h_f32 = h_f32 * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * h_f32.to(input_dtype))

ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.register_buffer("inv_freq", torch.empty(0, device="cpu"), persistent=False)

    def _get_freqs(self, seq_len, device):
        """Standard-Frequenzberechnung. Wird von Unterklassen überschrieben."""
        t = torch.arange(seq_len, device=device)
        return torch.outer(t, self.inv_freq)

    def train(self, mode: bool = True):
        super().train(mode)
        if not mode and not hasattr(self, 'cos_cached'):
            device = self.inv_freq.device
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            seqlen = self.max_position_embeddings
            freqs = self._get_freqs(seqlen, device)
            cos_cached = freqs.cos()[None, None, :, :].to(self.inv_freq.dtype)
            sin_cached = freqs.sin()[None, None, :, :].to(self.inv_freq.dtype)
            self.register_buffer("cos_cached", cos_cached, persistent=False)
            self.register_buffer("sin_cached", sin_cached, persistent=False)
        return self

    def forward(self):
        return self.cos_cached, self.sin_cached


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _get_freqs(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor
        return torch.outer(t, self.inv_freq)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _get_freqs(self, seq_len, device):
        base = self.base
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            # Hier müssen wir inv_freq lokal neu berechnen für die Formel
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))
        else:
            inv_freq = self.inv_freq
        t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
        return torch.outer(t, inv_freq)


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fused_gate_up_proj = nn.Linear(config.hidden_size, config.intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
#        try:
#            from flash_attn.ops.activations import swiglu
#            self.mlp_fn = swiglu
#        except ImportError:
        from transformers.activations import ACT2FN
        act_fn = ACT2FN[config.hidden_act]
        def fallback_swiglu(gate, up):
            return act_fn(gate) * up
        self.mlp_fn = fallback_swiglu

    def forward(self, x):
        # down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        fused_out = self.fused_gate_up_proj(x)
        gate, up = torch.chunk(fused_out, 2, dim=-1)
        intermediate_states = self.mlp_fn(gate, up)
        return self.down_proj(intermediate_states)


class LlamaAttentionBase(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.fused_qkv_proj = nn.Linear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        scaling_type = self.config.rope_scaling.get("rope_type", "default") if self.config.rope_scaling else "default"
        if scaling_type == "default":
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def apply_rotary_pos_emb(self, q, k, cos, sin, position_ids):
        raise NotImplementedError("Subclasses must implement apply_rotary_pos_emb to choose between SDPA and Flash")

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.LongTensor, past_key_value: torch.Tensor=None, current_pos: int = 0):
        raise NotImplementedError("Subclasses must implement forward to choose between SDPA and Flash")


class LlamaAttentionSdpa(LlamaAttentionBase):
    def __init__(self, config):
        super().__init__(config)

    def apply_rotary_pos_emb(self, q, k, cos, sin, position_ids):
        half = q.shape[-1] // 2
        q_f32 = q.to(torch.float32)
        k_f32 = k.to(torch.float32)
        cos_half = cos[0, 0, position_ids, :].unsqueeze(2) # [B, S, 1, D/2]
        sin_half = sin[0, 0, position_ids, :].unsqueeze(2) # [B, S, 1, D/2]

        q_embed = torch.empty_like(q_f32)
        q1, q2 = q_f32[..., :half], q_f32[..., half:]
        q_embed[..., :half] = q1 * cos_half - q2 * sin_half
        q_embed[..., half:] = q2 * cos_half + q1 * sin_half
        k_embed = torch.empty_like(k_f32)
        k1, k2 = k_f32[..., :half], k_f32[..., half:]
        k_embed[..., :half] = k1 * cos_half - k2 * sin_half
        k_embed[..., half:] = k2 * cos_half + k1 * sin_half
        return q_embed.to(q.dtype), k_embed.to(q.dtype)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.LongTensor, past_key_value: torch.Tensor = None, current_pos: int = 0):
        bsz, q_len, _ = hidden_states.size()
        kv_seq_len = current_pos + q_len

        fused_qkv = self.fused_qkv_proj(hidden_states)
        qkv = fused_qkv.view(bsz, q_len, -1, self.head_dim)
        q = qkv[:, :, :self.num_heads]
        k = qkv[:, :, self.num_heads : self.num_heads + self.num_key_value_heads]
        v = qkv[:, :, self.num_heads + self.num_key_value_heads :]
        cos, sin = self.rotary_emb()
        q, k = self.apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        past_key_value.update(k, v, current_pos)
        k_cache, v_cache = past_key_value.fetch(kv_seq_len, bsz)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k_cache.transpose(1, 2),
            v_cache.transpose(1, 2),
            is_causal=(q_len > 1)
        )
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output)


class LlamaAttentionFlash(LlamaAttentionBase):
    def __init__(self, config):
        super().__init__(config)
        from flash_attn import flash_attn_with_kvcache
        self.flash_fn = flash_attn_with_kvcache

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.LongTensor, past_key_value: torch.Tensor = None, current_pos: int = 0):
        bsz, q_len, _ = hidden_states.size()

        fused_qkv = self.fused_qkv_proj(hidden_states)
        qkv = fused_qkv.view(bsz, q_len, -1, self.head_dim)
        q = qkv[:, :, :self.num_heads].contiguous()
        k = qkv[:, :, self.num_heads : self.num_heads + self.num_key_value_heads].contiguous()
        v = qkv[:, :, self.num_heads + self.num_key_value_heads :].contiguous()

        pk, pv = past_key_value.pk, past_key_value.pv

        cache_seqlens = position_ids[:, 0].to(torch.int32)
        cos_slice = self.rotary_emb.cos_cached[0, 0, :, :].to(q.dtype)
        sin_slice = self.rotary_emb.sin_cached[0, 0, :, :].to(q.dtype)

        attn_output = self.flash_fn(
            q=q,
            k=k,
            v=v,
            k_cache=pk,
            v_cache=pv,
            rotary_cos=cos_slice,
            rotary_sin=sin_slice,
            cache_seqlens=cache_seqlens,
            softmax_scale=self.head_dim**-0.5,
            causal=(q_len > 1),
            rotary_interleaved=False, # Llama Standard
        )
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output


class LlamaDecoderLayerSkeleton(nn.Module):
    def __init__(self, config: LlamaConfig, attn_cls: type[LlamaAttentionBase]):
        super().__init__()

        self.self_attn = attn_cls(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Fp16KVCache:
    def __init__(self, k_slice, v_slice):
        self.k = k_slice # Shape: [Max_BSZ, Max_Len, NH, HD]
        self.v = v_slice

    def update(self, k, v, current_pos):
        bsz, q_len = k.size(0), k.size(1)
        # In-place Update des statischen Buffers
        self.k[:bsz, current_pos : current_pos + q_len] = k
        self.v[:bsz, current_pos : current_pos + q_len] = v

    def fetch(self, kv_seq_len, bsz):
        # Gibt nur den aktuell genutzten Batch-Teil zurück
        return self.k[:bsz, :kv_seq_len], self.v[:bsz, :kv_seq_len]


class Q8MuLawKVCache:
    def __init__(self, k_slice, v_slice, ks_slice, vs_slice, mu, device):
        self.pk, self.pv = k_slice, v_slice
        self.pk_scales, self.pv_scales = ks_slice, vs_slice
        self.mu = mu
        self.log_mu = torch.log1p(torch.tensor(float(mu), device=device))

    def update(self, k, v, current_pos):
        bsz, q_len = k.size(0), k.size(1)
        kv_seq_len = current_pos + q_len

        k_abs_max = torch.amax(k.abs(), dim=(1, 2, 3), keepdim=True).clamp(min=1e-5)
        v_abs_max = torch.amax(v.abs(), dim=(1, 2, 3), keepdim=True).clamp(min=1e-5)
        k_norm = k / k_abs_max
        v_norm = v / v_abs_max
        k_comp = torch.sign(k_norm) * torch.log1p(self.mu * torch.abs(k_norm)) / self.log_mu
        v_comp = torch.sign(v_norm) * torch.log1p(self.mu * torch.abs(v_norm)) / self.log_mu
        self.pk[:bsz, current_pos:kv_seq_len] = (k_comp * 127.0).round().to(torch.int8)
        self.pv[:bsz, current_pos:kv_seq_len] = (v_comp * 127.0).round().to(torch.int8)
        self.pk_scales[:bsz, current_pos:kv_seq_len] = k_abs_max
        self.pv_scales[:bsz, current_pos:kv_seq_len] = v_abs_max

    def fetch(self, kv_seq_len, bsz):
        k_deq_log = self.pk[:bsz, :kv_seq_len].to(torch.float16) / 127.0
        v_deq_log = self.pv[:bsz, :kv_seq_len].to(torch.float16) / 127.0
        k_deq_norm = torch.sign(k_deq_log) * (torch.expm1(torch.abs(k_deq_log) * self.log_mu) / self.mu)
        v_deq_norm = torch.sign(v_deq_log) * (torch.expm1(torch.abs(v_deq_log) * self.log_mu) / self.mu)
        k_cache = k_deq_norm * self.pk_scales[:bsz, :kv_seq_len]
        v_cache = v_deq_norm * self.pv_scales[:bsz, :kv_seq_len]
#        # --- FEHLER-CHECK (SNR-basiert) ---
#        if current_pos % 100 == 0:
#            with torch.no_grad():
#                abs_err = torch.abs(k - k_cache[:, current_pos:kv_seq_len]).mean().item()
#                rel_err = (abs_err / k.abs().mean().clamp(min=1e-5).item()) * 100
#                print(f"Token {current_pos:04d} | Normed-Log-Int8-Err: {rel_err:.4f}%")

#        if self.num_key_value_groups > 1: # repeat_kv = 1
#            print(f"DEBUG -- repeat_kv num_key_value_groups={self.num_key_value_groups}")
#            k_cache = k_cache.unsqueeze(3).expand(-1, -1, -1, self.num_key_value_groups, -1)
#            k_cache = k_cache.view(bsz, kv_seq_len, self.num_heads, self.head_dim)
#            v_cache = v_cache.unsqueeze(3).expand(-1, -1, -1, self.num_key_value_groups, -1)
#            v_cache = v_cache.view(bsz, kv_seq_len, self.num_heads, self.head_dim)
        return k_cache, v_cache


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.use_q8 = getattr(config, "use_q8_kv_cache", False)
        self.use_flash = is_flash_attn_2_available() and getattr(config, "use_flash_attn_2", False)

        if self.use_flash:
            print(f"Initialising flash attention")
            attn_cls=LlamaAttentionFlash
        elif self.use_q8:
            print(f"Initialising sdpa with quantized kv cache")
            attn_cls=LlamaAttentionSdpa
        else:
            print(f"Initialising sdpa with fp16 kv cache")
            attn_cls=LlamaAttentionSdpa
        self.layers = nn.ModuleList([LlamaDecoderLayerSkeleton(config, attn_cls=attn_cls) for _ in range(self.num_hidden_layers)])

    def train(self, mode: bool = True):
        super().train(mode)
        if not mode and not hasattr(self, 'position_ids_cache'):
            device = next(self.parameters()).device
            pos_ids = torch.arange(self.config.max_position_embeddings, dtype=torch.long, device=device).unsqueeze(0)
            self.register_buffer("position_ids_cache", pos_ids, persistent=False)

            bsz = getattr(self.config, "max_batch_size", 2)
            max_len = self.config.max_position_embeddings
            layers = self.config.num_hidden_layers
            hd = self.hidden_size // self.config.num_attention_heads
            nh = self.config.num_key_value_heads

            if self.use_q8:
                self.register_buffer("kv_storage_blob", torch.zeros((layers, 2, bsz, max_len, nh, hd), dtype=torch.int8, device=device), persistent=False)
                self.register_buffer("kv_scales_blob", torch.ones((layers, 2, bsz, max_len, 1, 1), dtype=torch.float16, device=device), persistent=False)
                self.kv_cache = [
                    Q8MuLawKVCache(self.kv_storage_blob[i,0], self.kv_storage_blob[i,1],
                                   self.kv_scales_blob[i,0], self.kv_scales_blob[i,1], self.config.q8_kv_cache_mu, device=device)
                    for i in range(layers)
                ]
            else:
                self.register_buffer("kv_storage_blob", torch.zeros((layers, 2, bsz, max_len, nh, hd), dtype=torch.float16, device=device), persistent=False)
                self.kv_cache = [
                    Fp16KVCache(self.kv_storage_blob[i,0], self.kv_storage_blob[i,1])
                    for i in range(layers)
                ]
        return self

    @torch.inference_mode()
    @torch.compile()
    def forward(self, inputs_embeds: torch.FloatTensor, current_pos: int = 0):
        # retrieve inputs_embeds
        batch_size, seq_length, _ = inputs_embeds.shape
        position_ids = self.position_ids_cache[:, current_pos : current_pos + seq_length]
        if batch_size > 1:
            position_ids = position_ids.expand(batch_size, seq_length)
        # decoder layers
        hidden_states = inputs_embeds
        for idx, layer in enumerate(self.layers):
            residual = hidden_states
            normed = layer.input_layernorm(hidden_states)
            # Self Attention
            attn_out = layer.self_attn(
                hidden_states=normed,
                position_ids=position_ids,
                past_key_value=self.kv_cache[idx],
                current_pos=current_pos,
            )
            hidden_states = residual + attn_out
            # Fully Connected
            residual = hidden_states
            normed = layer.post_attention_layernorm(hidden_states)
            mlp_out = layer.mlp(normed)
            hidden_states = residual + mlp_out

        hidden_states = self.norm(hidden_states)
        return hidden_states


class LlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, inputs_embeds: torch.FloatTensor, current_pos: int = 0):
        hidden_states = self.model(
            inputs_embeds=inputs_embeds,
            current_pos=current_pos
        )
        logits = self.lm_head(hidden_states)

        return logits, hidden_states
