# Original work Copyright (c) Tencent AI Lab
# Refactoring and modifications Copyright (c) 2026 Siriusquirrel
#
# Part of the SongGeneration-v2-Large-16GB-Fork
# Modifications: Added penalty histogram for record_window
#                Split original lm_levo.py into lmlevo_cb0.py and lmlevo_cb12.py for per codebook optimisation
#                Removed pattern.py (Codebook pattern provider) dependency
#                Correct flushing of codebooks 1 and 2 (delay pattern)

import torch
import random
import torch.nn as nn
from codeclm.models.utils import sample_top_p, sample_top_k, multinomial
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
from codeclm.models.llama.modeling_llama import LlamaForCausalLM
from codeclm.models.llama.configuration_llama import LlamaConfig


def get_lm_model_cb12(cfg: DictConfig): #-> LMModel:
    lm_kwargs = OmegaConf.to_container(getattr(cfg, 'lm'), resolve=True)
    delay_tokens = max(cfg.codebooks_pattern.delay.delays)
    return LmModel_cb12(cfg=cfg, delay_tokens=delay_tokens, **lm_kwargs)


class LmModel_cb12(nn.Module):
    """Transformer-based language model for codebooks 1 and 2."""
    def __init__(self, 
                 code_depth: int = 8,
                 code_size: int = 1024,
                 dim: int = 128,
                 intermediate_size: int = 4096,
                 num_heads: int = 8,
                 num_layers_sub: int = 12,
                 max_position_embeddings_sub: int = 10000,
                 rope_theta_sub: float = 500000.0,
                 cfg = None,
                 delay_tokens: int = 250,
                 **kwargs):
        super().__init__()

        self.code_size = code_size + 1   # + EOS
        input_emb_dim = code_size + 2   # EOP
        self.code_depth = code_depth
        self.dim = dim
        self.cfg = cfg

        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        self.layer2_emb = nn.ModuleList([nn.Embedding(input_emb_dim, dim)
                                  for _ in range(self.code_depth)])
        kwargs.pop('max_position_embeddings', None)
        kwargs.pop('rope_theta', None)
        sub_model_cfg = LlamaConfig(
            hidden_size=dim,
            intermediate_size = intermediate_size,
            num_attention_heads = num_heads,
            num_hidden_layers = num_layers_sub,
            num_key_value_heads = num_heads,
            vocab_size = self.code_size,
            max_position_embeddings=max_position_embeddings_sub,
            rms_norm_eps= 1e-5,
            rope_theta= rope_theta_sub,
            **kwargs
        )
        self.transformer2 = LlamaForCausalLM(sub_model_cfg)
        self.linears = nn.ModuleList([nn.Linear(dim, self.code_size, bias=False) 
                                    for _ in range(code_depth - 1)])
        self.delay_tokens = delay_tokens

    @property
    def special_token_id(self) -> int:
        return self.code_size

    @property
    def eos_token_id(self) -> int:
        return self.code_size-1

    def train(self, mode=True):
        super().train(mode)
        if not mode and not hasattr(self, 'penalty_histogram'):
            device = next(self.parameters()).device
            self.register_buffer('penalty_histogram', torch.zeros((2, self.special_token_id), device=device), persistent=False)
            self.mlp = torch.compile(self.mlp, fullgraph=True, dynamic=False, mode="max-autotune")
        return self

    @torch.inference_mode()
    def generate(self, cb0_dict) -> dict:
        assert not self.training, "generation shouldn't be used in training mode."
        first_param = next(iter(self.parameters()))
        device = first_param.device
        self.current_pos = 0
        B=1

        use_sampling = cb0_dict["use_sampling"]
        temp = cb0_dict["temp"]
        top_k = cb0_dict["top_k"]
        top_p = cb0_dict["top_p"]
        cfg_coef = cb0_dict["cfg_coef"]
        ignore_tokens = cb0_dict["ignore_tokens"]
        cb0_lhs = cb0_dict["hidden_states"]
        cb0_tokens = cb0_dict["tokens"]
        max_len = cb0_tokens.shape[0]
        tokens = torch.zeros((3, max_len), dtype=torch.long, device=device)
        tokens[0] = cb0_tokens
        token_input = torch.full((2*B, 2, 1), self.special_token_id, dtype=torch.long, device=device)
        temp1 = self.layer2_emb[1](token_input[:, 0, :])
        temp2 = self.layer2_emb[2](token_input[:, 1, :])
        token_emb = temp1 + temp2
        token_emb_expanded = token_emb.expand(-1, self.delay_tokens, -1)     # [2, delay, D]
        lhs_delay = cb0_lhs[:, :self.delay_tokens, :]                        # [2, delay, D_lhs]
        input_combined = torch.cat([token_emb_expanded, lhs_delay], dim=-1)  # [2, delay, D+D_lhs]
        full_mlp_input = torch.cat([cb0_dict["prefill_cb12"], input_combined], dim=1)
        full_prefill = self.mlp(full_mlp_input)                              # [2, delay, D_model]
        self.transformer2(inputs_embeds=full_prefill, current_pos=self.current_pos)
        self.current_pos += full_prefill.shape[1]

        record_window = cb0_dict["record_window"]
        if record_window > 0:
            self.penalty_histogram.zero_()
            history_buffer = torch.zeros((2, record_window), dtype=torch.long, device=device)

        step=0
        with tqdm(total=max_len, initial=1, desc="Generating Tokens") as pbar:
            while step < max_len:
                input_combined = torch.cat([token_emb, cb0_lhs[:, step+self.delay_tokens:step+1+self.delay_tokens, :]], dim=-1)
                input_next = self.mlp(input_combined)
                _, h_states_2 = self.transformer2(inputs_embeds=input_next, current_pos=self.current_pos)
                self.current_pos += 1
                logits_cb1 = self.linears[0](h_states_2)
                logits_cb2 = self.linears[1](h_states_2)
                block_logits = torch.stack([logits_cb1, logits_cb2], dim=1)
                cond_logits, uncond_logits = block_logits.split(B, dim=0)
                logits = uncond_logits + (cond_logits - uncond_logits) * cfg_coef
                logits = logits.squeeze(2) # only one token at a time

                if record_window > 0:
                    factor = 1.1 ** self.penalty_histogram.unsqueeze(0)
                    logits = torch.where(logits > 0, logits / factor, logits * factor)
#                if ignore_tokens is not None and ignore_tokens.numel() > 0:
#                    logits.index_fill_(-1, ignore_tokens, float('-inf'))
                if use_sampling and temp > 0.0:
                    logits.div_(temp)
                    probs = torch.softmax(logits, dim=-1)
                    if top_p > 0.0:
                        next_tokens_block = sample_top_p(probs, p=top_p)
                    elif top_k > 0:
                        next_tokens_block = sample_top_k(probs, k=1)
                    else:
                        next_tokens_block = multinomial(probs, num_samples=1)
                else:
                    next_tokens_block = torch.argmax(logits, dim=-1, keepdim=True)

                tokens[1:3, step:step+1] = next_tokens_block[0, :, 0:1]
                token_input = next_tokens_block.expand(2, -1, -1)
                temp1 = self.layer2_emb[1](token_input[:, 0, :])
                temp2 = self.layer2_emb[2](token_input[:, 1, :])
                token_emb = temp1 + temp2

                if record_window > 0:
                    new_t = next_tokens_block[0, :, 0]
                    idx = step % record_window
                    if step >= record_window:
                        old_t = history_buffer[:, idx]
                        self.penalty_histogram[range(2), old_t] -= 1
                    history_buffer[:, idx] = new_t
                    self.penalty_histogram[range(2), new_t] += 1

                step += 1
                pbar.update(1)

        cb0_dict["tokens"] = tokens
        del cb0_dict["hidden_states"]
        del cb0_dict["prefill_cb12"]
        return cb0_dict
