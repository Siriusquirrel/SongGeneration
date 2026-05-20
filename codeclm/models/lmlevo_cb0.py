# Original work Copyright (c) Tencent AI Lab
# Refactoring and modifications Copyright (c) 2026 Siriusquirrel
#
# Part of the SongGeneration-v2-Large-16GB-Fork
# Modifications: Added penalty histogram for record_window
#                Split original lm_levo.py into lmlevo_cb0.py and lmlevo_cb12.py for per codebook optimisation
#                Removed pattern.py (Codebook pattern provider) dependency

import torch
import random
import torch.nn as nn
from codeclm.models.utils import sample_top_p, sample_top_k, multinomial
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
from codeclm.models.llama.modeling_llama import LlamaForCausalLM
from codeclm.models.llama.configuration_llama import LlamaConfig
from codeclm.modules.conditioners import (
    ConditioningAttributes,
    AudioCondition,
    ConditionType,
    ConditionerProvider,
    ConditionFuser,
    ClassifierFreeGuidanceDropoutInference,
    ClassifierFreeGuidanceDropout,
    AttributeDropout,
    QwTokenizerConditioner,
    QwTextConditioner,
    QuantizedEmbeddingConditioner,
)


def get_lm_model_cb0(cfg: DictConfig, version: str = 'v1'): #-> LMModel:
    lm_kwargs = OmegaConf.to_container(getattr(cfg, 'lm'), resolve=True)
    condition_provider = get_conditioner_provider(lm_kwargs["dim"], cfg, version=version)
    fuser = get_condition_fuser(cfg)
    delay_tokens = max(cfg.codebooks_pattern.delay.delays)
    return LmModel_cb0(condition_provider=condition_provider, fuser=fuser, cfg=cfg, delay_tokens=delay_tokens, **lm_kwargs)

def get_conditioner_provider(output_dim: int, cfg: DictConfig, version: str = 'v1') -> ConditionerProvider:
    """Instantiate a conditioning model."""
    cfg = getattr(cfg, 'conditioners')
    dict_cfg = {} if cfg is None else OmegaConf.to_container(cfg, resolve=True)
    conditioners: dict[str, BaseConditioner] = {}
    condition_provider_args = dict_cfg.pop('args', {})

    for cond, cond_cfg in dict_cfg.items():
        model_type = cond_cfg['model']
        model_args = cond_cfg[model_type]
        if model_type == 'QwTokenizer':
            conditioners[str(cond)] = QwTokenizerConditioner(
                output_dim=output_dim,
                version=version,
                **model_args
            )
        elif model_type == "QwTextTokenizer":
            conditioners[str(cond)] = QwTextConditioner(
                output_dim=output_dim,
                version=version,
                **model_args
            )
        elif model_type == "qt_embedding":
            conditioners[str(cond)] = QuantizedEmbeddingConditioner(
                dim=output_dim,
                **model_args
            )
        else:
            raise ValueError(f"Unrecognized conditioning model: {model_type}")
    conditioner = ConditionerProvider(conditioners, **condition_provider_args)
    return conditioner

def get_condition_fuser(cfg: DictConfig) -> ConditionFuser:
    """Instantiate a condition fuser object."""
    fuser_cfg = getattr(cfg, 'fuser')
    fuser_methods = ['prepend']
    fuse2cond = {k: fuser_cfg[k] for k in fuser_methods}
    kwargs = {k: v for k, v in fuser_cfg.items() if k not in fuser_methods}
    fuser = ConditionFuser(fuse2cond=fuse2cond, **kwargs)
    return fuser


class LmModel_cb0(nn.Module):
    """Transformer-based language model for codebook 0."""
    def __init__(self,
                 condition_provider: ConditionerProvider,
                 fuser: ConditionFuser,
                 code_size: int = 1024,
                 dim: int = 128,
                 intermediate_size: int = 4096,
                 num_heads: int = 8,
                 num_layers=36,
                 max_position_embeddings: int = 8196,
                 rope_theta: float = 100000.0,
                 cfg = None,
                 delay_tokens: int = 250,
                 **kwargs):
        super().__init__()

        self.condition_provider = condition_provider
        self.fuser = fuser
        self.code_size = code_size + 1   # + EOS
        input_emb_dim = code_size + 2   # EOP
        self.dim = dim
        self.cfg = cfg
        self.emb = nn.ModuleList([nn.Embedding(input_emb_dim, dim)])

        model_cfg = LlamaConfig(
            hidden_size=dim,
            intermediate_size = intermediate_size,
            num_attention_heads = num_heads,
            num_hidden_layers = num_layers,
            num_key_value_heads = num_heads,
            vocab_size = self.code_size,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps= 1e-5,
            rope_theta= rope_theta,
            **kwargs
        )
        self.transformer = LlamaForCausalLM(model_cfg)
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
            self.register_buffer('penalty_histogram', torch.zeros((self.special_token_id), device=device), persistent=False)
        return self

    @torch.inference_mode()
    def prepare_condition_tensors(self,
                                   text: list[str] = None,
                                   descriptions: list[str] = None,
                                   audio_qt_emb: list[torch.Tensor] = None,
                                   prepare_null_condition = False,
                                   ):
        attr = ConditioningAttributes()
        if 'description' in self.condition_provider.conditioners:
            attr["text"]["description"] = text[0] if text else ""
        if 'prompt_audio' in self.condition_provider.conditioners:
            current_emb = audio_qt_emb[0][None]
            mask = (current_emb[:, :, 0] == self.special_token_id).bool().unsqueeze(-1)
            audio_qt_seq = torch.full_like(current_emb[:, :, 0], self.eos_token_id).unsqueeze(-1)
            audio_qt_seq = torch.cat([audio_qt_seq, current_emb], dim=-1)
            mask = mask.expand(-1, -1, audio_qt_seq.shape[-1])
            audio_qt_seq[mask] = self.special_token_id
            attr["audio"]['prompt_audio'] = AudioCondition(
                tokens=audio_qt_seq.long().cuda(),
                length=torch.tensor([audio_qt_seq.shape[-1]], dtype=torch.long),
                sample_rate=[self.cfg.sample_rate],)
        if 'type_info' in self.condition_provider.conditioners:
            attr["text"]["type_info"] = descriptions[0] if descriptions else ""
        conditions = [attr]
        print("conditions", conditions)
        if prepare_null_condition:
            cfg_inference = ClassifierFreeGuidanceDropoutInference()
            null_conditions = cfg_inference(conditions, condition_types=["audio", "text"],
                                            customized=None)
            conditions = conditions + null_conditions
        tokenized_conditions = self.condition_provider.tokenize(conditions)
        condition_tensors = self.condition_provider(tokenized_conditions)
        return condition_tensors

    @torch.inference_mode()
    def generate(self, texts=None, descriptions=None, audio_qt_embs=None, max_gen_len: int=256, use_sampling: bool=True,
                 temp: float=1.0, top_k: int=250, top_p: float=0.0, cfg_coef: float=None, record_window: int=150
                 ) -> dict:
        assert not self.training, "generation shouldn't be used in training mode."
        first_param = next(iter(self.parameters()))
        device = first_param.device
        self.current_pos = 0
        B=1

        condition_tensors = self.prepare_condition_tensors(text=texts, descriptions=descriptions, audio_qt_emb=audio_qt_embs, prepare_null_condition=True)
        empty_input = torch.zeros((B*2, 0, self.dim), dtype=torch.float16, device=device)
        prefill_cb0, prefill_cb12 = self.fuser(empty_input, empty_input, condition_tensors)
        _, h_states_1 = self.transformer(inputs_embeds=prefill_cb0, current_pos=self.current_pos)
        fused_input2_combined = torch.cat([prefill_cb12, h_states_1], dim=-1)
        self.current_pos += prefill_cb0.shape[1]
        output={"prefill_cb12": fused_input2_combined}
        output["tokens"] = torch.zeros(max_gen_len, dtype=torch.long, device=device)
        output["hidden_states"] = torch.zeros((2, max_gen_len+self.delay_tokens, self.dim), dtype=torch.float16, device=device)
        output["use_sampling"] = use_sampling
        output["temp"] = temp
        output["top_k"] = top_k
        output["top_p"] = top_p
        output["cfg_coef"] = cfg_coef
        ignore_tokens = audio_qt_embs[0][0]
        ignore_tokens = ignore_tokens[ignore_tokens < self.eos_token_id]
        output["ignore_tokens"] = ignore_tokens
        output["record_window"] = record_window
        if record_window > 0:
            self.penalty_histogram.zero_()
            history_buffer = torch.zeros((record_window), dtype=torch.long, device=device)

        combined_sequence = torch.full((B*2, 1, 1), self.special_token_id, dtype=torch.long, device=device)
        special_token_emb = self.emb[0](combined_sequence[:, 0])
        # auto-regressive sampling
        step=0
        with tqdm(total=max_gen_len, initial=1, desc="Generating Tokens") as pbar:
            while step < max_gen_len:
                input_1 = self.emb[0](combined_sequence[:, 0])
                all_logits, all_h_states = self.transformer(inputs_embeds=input_1, current_pos=self.current_pos)
                self.current_pos += 1

                cond_logits, uncond_logits = all_logits.split(B, dim=0)
                logits = uncond_logits + (cond_logits - uncond_logits) * cfg_coef # [B, S, card]

                if record_window > 0:
                    factor = 1.1 ** self.penalty_histogram.view(1, 1, -1)
                    factor = factor.expand_as(logits)
                    logits = torch.where(logits > 0, logits / factor, logits * factor)

                if ignore_tokens is not None and ignore_tokens.numel() > 0:
                    logits[0].index_fill_(-1, ignore_tokens, float('-inf'))

                if use_sampling and temp > 0.0:
                    logits.div_(temp)
                    probs = torch.softmax(logits, dim=-1)
                    if top_p > 0.0:
                        next_token = sample_top_p(probs, p=top_p)
                    elif top_k > 0:
                        next_token = sample_top_k(probs, k=top_k)
                    else:
                        next_token = multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

                new_t = next_token[0,0,-1]
                if new_t == self.eos_token_id:
                    output["tokens"] = output["tokens"][:step]
                    output["hidden_states"] = output["hidden_states"][:, :step+self.delay_tokens, :]
                    break

                output["tokens"][step] = new_t
                output["hidden_states"][:, step:step+1, :] = all_h_states
                combined_sequence = new_t.view(1, 1, 1).repeat(2, 1, 1)

                if record_window > 0:
                    idx = step % record_window
                    if step >= record_window:
                        old_t = history_buffer[idx]
                        self.penalty_histogram[old_t] -= 1
                    history_buffer[idx] = new_t
                    self.penalty_histogram[new_t] += 1

                step += 1
                pbar.update(1)

        delay_input_emb = special_token_emb.expand(-1, self.delay_tokens, -1) # Shape: [2, 250, Dim]
        _, lhs_cb0 = self.transformer(inputs_embeds=delay_input_emb, current_pos=self.current_pos)
        output["hidden_states"][:, -self.delay_tokens:, :] = lhs_cb0
        self.current_pos += self.delay_tokens
        return output
