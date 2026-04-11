# Original work Copyright (c) Tencent AI Lab
# Refactoring and modifications Copyright (c) 2026 Siriusquirrel
#
# Part of the SongGeneration-v2-Large-16GB-Fork
# Modifications: histogram for record_window and correct token output in generate (flushing code books 1 and 2)

import torch
import random
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
from codeclm.models.llama.modeling_llama import LlamaForCausalLM
from codeclm.models.llama.configuration_llama import LlamaConfig
from codeclm.modules.streaming import StreamingModule
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
from codeclm.modules.pattern import CodebooksPatternProvider, DelayedPatternProvider

def sample_top_k(probs: torch.Tensor, k: int, generator=None) -> torch.Tensor:
    """Sample next token from top K values along the last dimension of the input probs tensor.

    Args:
        probs (torch.Tensor): Input probabilities with token candidates on the last dimension.
        k (int): The k in “top-k”.
    Returns:
        torch.Tensor: Sampled tokens.
    """
    top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)
    top_k_probs /= top_k_probs.sum(dim=-1, keepdim=True)
    next_token_offet = multinomial(top_k_probs, 1, generator=generator)
    return torch.gather(top_k_indices, -1, next_token_offet)

def sample_top_p(probs: torch.Tensor, p: float, generator=None) -> torch.Tensor:
    """Sample next token from top P probabilities along the last dimension of the input probs tensor.

    Args:
        probs (torch.Tensor): Input probabilities with token candidates on the last dimension.
        p (int): The p in “top-p”.
    Returns:
        torch.Tensor: Sampled tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = (probs_sum - probs_sort) > p
    probs_sort *= (~mask).float()
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token_sorted_idx = multinomial(probs_sort, num_samples=1, generator=generator)
    return torch.gather(probs_idx, -1, next_token_sorted_idx)

def multinomial(input: torch.Tensor, num_samples: int, replacement=False, *, generator=None):
    """torch.multinomial with arbitrary number of dimensions, and number of candidates on the last dimension.

    Args:
        input (torch.Tensor): The input tensor containing probabilities.
        num_samples (int): Number of samples to draw.
        replacement (bool): Whether to draw with replacement or not.
    Keywords args:
        generator (torch.Generator): A pseudorandom number generator for sampling.
    Returns:
        torch.Tensor: Last dimension contains num_samples indices
            sampled from the multinomial probability distribution
            located in the last dimension of tensor input.
    """
    input_flat = input.reshape(-1, input.shape[-1])
    samples_flat = torch.multinomial(input_flat, num_samples=num_samples, replacement=replacement, generator=generator)
    return samples_flat.reshape(*input.shape[:-1], num_samples)

def get_lm_model(cfg: DictConfig, version: str = 'v1'): #-> LMModel:
    """Instantiate a LM."""

    lm_kwargs = OmegaConf.to_container(getattr(cfg, 'lm'), resolve=True)
    # n_q: number of RVQ
    code_depth = lm_kwargs['code_depth']
    q_modeling = lm_kwargs.pop('q_modeling', None)
    # conditioner
    condition_provider = get_conditioner_provider(lm_kwargs["dim"], cfg, version=version)
    # codebook pattern: delay
    codebooks_pattern_cfg = getattr(cfg, 'codebooks_pattern')
    if codebooks_pattern_cfg.modeling is None:
        assert q_modeling is not None, \
            "LM model should either have a codebook pattern defined or transformer_lm.q_modeling"
        codebooks_pattern_cfg = OmegaConf.create(
            {'modeling': q_modeling, 'delay': {'delays': list(range(code_depth))}}
        )
    pattern_provider = get_codebooks_pattern_provider(code_depth, codebooks_pattern_cfg)
    # condition dropout
    attribute_dropout = OmegaConf.to_container(getattr(cfg, 'attribute_dropout'), resolve=True)
    cls_free_guidance = OmegaConf.to_container(getattr(cfg, 'classifier_free_guidance'), resolve=True)
    cfg_prob, cfg_coef = cls_free_guidance['training_dropout'], cls_free_guidance['inference_coef']
    # condition fuser
    fuser = get_condition_fuser(cfg)
    lm_type = lm_kwargs['lm_type'] # YCY: For consistency, choose different lm.py based on lm_type
    if lm_type == 'Llama':
        return LmModel(
            pattern_provider=pattern_provider,
            condition_provider=condition_provider,
            fuser=fuser,
            cfg_dropout=cfg_prob,
            cfg_coef=cfg_coef,
            attribute_dropout=attribute_dropout,
            cfg=cfg,
            **lm_kwargs
        )
    else:
        raise KeyError(f"Unexpected LM model {lm_type}")

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
    fuser_methods = ['sum', 'prepend']
    fuse2cond = {k: fuser_cfg[k] for k in fuser_methods}
    kwargs = {k: v for k, v in fuser_cfg.items() if k not in fuser_methods}
    fuser = ConditionFuser(fuse2cond=fuse2cond, **kwargs)
    return fuser

def get_codebooks_pattern_provider(code_depth: int, cfg: DictConfig) -> CodebooksPatternProvider:
    """Instantiate a codebooks pattern provider object."""
    pattern_providers = {
        'delay': DelayedPatternProvider,
    }
    name = cfg.modeling
    kwargs = OmegaConf.to_container(cfg.get(name), resolve=True) if hasattr(cfg, name) else {}
    klass = pattern_providers[name]
    return klass(code_depth, **kwargs)

ConditionTensors = dict[str, ConditionType]


class LmModel(StreamingModule):
    """Transformer-based language model on multiple streams of codes.

    Args:
        pattern_provider (CodebooksPatternProvider): Pattern provider for codebook interleaving.
        condition_provider (ConditioningProvider): Conditioning provider from metadata.
        fuser (ConditionFuser): Fuser handling the fusing of conditions with language model input.
        code_depth (int): Number of parallel streams to model.
        code_size (int): Cardinality, vocabulary size.
        dim (int): Dimension of the transformer encoder.
        num_heads (int): Number of heads for the transformer encoder.
        hidden_scale (int): Scale for hidden feed forward dimension of the transformer encoder.
        norm (str): Normalization method.
        norm_first (bool): Use pre-norm instead of post-norm.
        emb_lr (float, optional): Embedding-specific learning rate.
        bias_proj (bool): Use bias for output projections.
        weight_init (str, optional): Method for weight initialization.
        depthwise_init (str, optional): Method for depthwise weight initialization.
        zero_bias_init (bool): If true and bias in Linears, initialize bias to zeros.
        cfg_dropout (float): Classifier-free guidance dropout.
        cfg_coef (float): Classifier-free guidance coefficient.
        attribute_dropout (dict): Attribute dropout probabilities.
        two_step_cfg (bool): Whether to run classifier free-guidance with 2 distinct steps.
        **kwargs: Additional parameters for the transformer encoder.
    """
    def __init__(self, 
                 pattern_provider: CodebooksPatternProvider, 
                 condition_provider: ConditionerProvider,
                 fuser: ConditionFuser, 
                 code_depth: int = 8, 
                 code_size: int = 1024, 
                 dim: int = 128,
                 intermediate_size: int = 4096,
                 num_heads: int = 8,
                 norm: str = 'layer_norm', norm_first: bool = False,
                 weight_init: str = None, depthwise_init: str = None,
                 zero_bias_init: bool = False, cfg_dropout: float = 0, cfg_coef: float = 1.0,
                 attribute_dropout: dict[str, dict[str, float]] = {}, 
                 num_layers=16,
                 max_position_embeddings: int = 8196,
                 max_position_embeddings_sub: int = 10000,
                 rope_theta: float = 100000.0,
                 rope_theta_sub: float = 500000.0,
                 num_layers_sub: int = 12,
                 cfg = None,
                 **kwargs):
        super().__init__()

        self.cfg_coef = cfg_coef

        self.cfg_dropout = ClassifierFreeGuidanceDropout(p=cfg_dropout,seed=random.randint(0, 9999))
        self.att_dropout = AttributeDropout(p=attribute_dropout,seed=random.randint(0, 9999))
        self.condition_provider = condition_provider
        self.fuser = fuser
        self.code_size = code_size + 1   # + EOS
        input_emb_dim = code_size + 2   # EOP
        self.code_depth = code_depth
        self.dim = dim
        self.cfg = cfg
        self.pattern_provider = pattern_provider
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

        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        self.layer2_emb = nn.ModuleList([nn.Embedding(input_emb_dim, dim)
                                  for _ in range(self.code_depth)])
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
        # enable EOS prediction
        if code_depth > 1:
            self.linears = nn.ModuleList([nn.Linear(dim, self.code_size, bias=False) 
                                        for _ in range(code_depth - 1)])

    @property
    def special_token_id(self) -> int:
        return self.code_size

    @property
    def eos_token_id(self) -> int:
        return self.code_size-1

    @property
    def pad_token_id(self) -> int:
        return self.code_size + 1

    def train(self, mode=True):
        super().train(mode)
        if not mode and not hasattr(self, 'penalty_histogram'):
            device = next(self.parameters()).device
            self.register_buffer('penalty_histogram', torch.zeros((self.code_depth, self.special_token_id), device=device))
            self.mlp = torch.compile(self.mlp, fullgraph=True, dynamic=False, mode="max-autotune")
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
            mask = (current_emb[:, :, 0] == self.pad_token_id).bool().unsqueeze(-1)
            audio_qt_seq = torch.full_like(current_emb[:, :, 0], self.eos_token_id).unsqueeze(-1)
            audio_qt_seq = torch.cat([audio_qt_seq, current_emb], dim=-1)
            mask = mask.expand(-1, -1, audio_qt_seq.shape[-1])
            audio_qt_seq[mask] = self.pad_token_id
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

        """Apply language model on sequence and conditions.
        Given a tensor of sequence of shape [B, K, S] with K the number of codebooks and
        S the sequence steps, return the logits with shape [B, card, K, S].

        Args:
            indices (torch.Tensor): Indices of the codes to model.
            condition_tensors (dict[str, ConditionType], optional): Pre-computed conditioning
                tensors, see `conditions`.
        Returns:
            torch.Tensor: Logits.
        """
    @torch.inference_mode()
    def forward(self, sequence: torch.Tensor, condition_tensors: ConditionTensors) -> torch.Tensor:
        # B = Batch, K = Codebooks, S = Audio-Sequenzlänge
        B, K, S = sequence.shape
        assert K == self.code_depth, "Sequence shape must match the specified number of codebooks"

        input_1 = self.emb[0](sequence[:, 0])
        temp_list = [self.layer2_emb[k](sequence[:, k]) for k in range(1, K)]
        input_2 = torch.stack(temp_list).sum(dim=0)
        fused_input1, fused_input2 = self.fuser(input_1, input_2, condition_tensors)

        logits_1, h_states_1 = self.transformer(
            inputs_embeds=fused_input1,
            current_pos=self.current_pos
        )
        logits = logits_1.unsqueeze(1) # [B, 1, fused_input1.shape[1], card]

        if K > 1:
            fused_input2_combined = torch.cat([fused_input2, h_states_1], dim=-1)
            fused_input2_next = self.mlp(fused_input2_combined)

            _, h_states_2 = self.transformer2(
                inputs_embeds=fused_input2_next,
                current_pos=self.current_pos
            )
            res_logits = torch.stack([self.linears[k](h_states_2) for k in range(K - 1)], dim=1)  # [B, K-1, fused_input1.shape[1], card]
            logits = torch.cat([logits, res_logits], dim=1)  # [B, K, fused_input1.shape[1], card]

        self.current_pos += fused_input1.shape[1]

        if len(self.fuser.fuse2cond['prepend']) > 0:
            logits = logits[:, :, -S:, :]
        return logits  # [B, K, S, card]

    @torch.inference_mode()
    def generate(self,
                 texts = None,
                 descriptions = None,
                 audio_qt_embs = None,
                 num_samples: int = None,
                 max_gen_len: int = 256,
                 use_sampling: bool = True,
                 temp: float = 1.0,
                 top_k: int = 250,
                 top_p: float = 0.0,
                 cfg_coef: float = None,
                 record_window: int = 150
                 ) -> torch.Tensor:
        """Generate tokens sampling from the model given a prompt or unconditionally. Generation can
        be perform in a greedy fashion or using sampling with top K and top P strategies.

        Args:
            prompt (torch.Tensor, optional): Prompt tokens of shape [B, K, T].
            conditions_tensors (list of ConditioningAttributes, optional): List of conditions.
            num_samples (int, optional): Number of samples to generate when no prompt and no conditions are given.
            max_gen_len (int): Maximum generation length in tokens.
            use_sampling (bool): Whether to use a sampling strategy or not.
            temp (float): Sampling temperature.
            top_k (int): K for "top-k" sampling.
            top_p (float): P for "top-p" sampling.
            cfg_coeff (float, optional): Classifier-free guidance coefficient.
            callback (Callback, optional): Callback function to report generation progress.
        Returns:
            torch.Tensor: Generated tokens.
        """
        assert not self.training, "generation shouldn't be used in training mode."
        first_param = next(iter(self.parameters()))
        device = first_param.device
        self.current_pos = 0
        if record_window > 0:
            self.penalty_histogram.zero_()
            history_buffer = torch.zeros((self.code_depth, record_window), dtype=torch.long, device=device)

        # 1) Check input shapes are consistent
        possible_num_samples = []
        if num_samples is not None:
            possible_num_samples.append(num_samples)
        elif texts:
            possible_num_samples.append(len(texts))
        elif audio_qt_embs:
            possible_num_samples.append(len(audio_qt_embs))
        else:
            possible_num_samples.append(1)
        assert [x == possible_num_samples[0] for x in possible_num_samples], "Inconsistent inputs shapes"
        num_samples = possible_num_samples[0]
        condition_tensors = self.prepare_condition_tensors(text=texts, descriptions=descriptions, audio_qt_emb=audio_qt_embs, prepare_null_condition=True)
        # 4) set up startoff patterns
        start_offset = 0
        assert start_offset < max_gen_len, f"{start_offset}, {max_gen_len}"
        pattern = self.pattern_provider.get_pattern(max_gen_len)
        # this token is used as default value for codes that are not generated yet
        unknown_token = -1
        # we generate codes up to the max_gen_len that will be mapped to the pattern sequence
        B = num_samples
        gen_codes = torch.full((B, self.code_depth, max_gen_len),
                               unknown_token, dtype=torch.long, device=device)
        # create the gen_sequence with proper interleaving from the pattern: [B, K, S]
        gen_sequence, indexes, mask = pattern.build_pattern_sequence(gen_codes, self.special_token_id)
        output_codes = torch.full_like(gen_sequence, self.code_size)
        # retrieve the start_offset in the sequence:
        # it is the first sequence step that contains the `start_offset` timestep
        start_offset_sequence = pattern.get_first_step_with_timesteps(start_offset, None)
        assert start_offset_sequence is not None
        is_end = torch.zeros((B, self.code_depth, 1)).bool().to(device)
        ignore_tokens = audio_qt_embs[0][0]
        ignore_tokens = ignore_tokens[ignore_tokens < 16384]
        delay_steps = pattern.max_delay
        flush_counter = -1
        # 5) auto-regressive sampling
        with self.streaming(), torch.inference_mode():
            gen_sequence_len = gen_sequence.shape[-1]  # gen_sequence shape is [B, K, S]
            prev_offset = 0
            for offset in tqdm(range(start_offset_sequence, gen_sequence_len)):
                # get current sequence (note that the streaming API is providing the caching over previous offsets)
                curr_sequence = gen_sequence[..., prev_offset:offset]
                # sample next token from the model, next token shape is [B, K, 1]
                next_token = self._sample_next_token(
                    curr_sequence, condition_tensors, use_sampling, temp, top_k, top_p,
                    cfg_coef=cfg_coef,
                    record_window=record_window,
                    ignore_tokens = ignore_tokens
                    )
                # ensure the tokens that should be masked are properly set to special_token_id
                # as the model never output special_token_id
                valid_mask = mask[..., offset:offset+1].expand(B, -1, -1)
                next_token[~valid_mask] = self.special_token_id
                # 检查eos id
                next_token[is_end] = self.special_token_id
                if flush_counter == -1 and torch.any(next_token[:, 0, :] == self.eos_token_id):
                    flush_counter = 0
                if flush_counter >= 0:
                    flush_counter += 1
                    next_token[:, 0, :] = self.special_token_id
                    if flush_counter >= delay_steps:
                        is_end.fill_(True)
                else:
                    is_end = is_end | (next_token == self.eos_token_id)
                # ensure we don't overwrite prompt tokens, we only write over unknown tokens
                # (then mask tokens should be left as is as well, which is correct)
                gen_sequence[..., offset:offset+1] = torch.where(
                    gen_sequence[..., offset:offset+1] == unknown_token,
                    next_token, gen_sequence[..., offset:offset+1])
                if torch.all(is_end):
                    gen_sequence = gen_sequence[..., :offset+1]
                    break
                if record_window > 0:
                    new_t = next_token[0, :, 0]  # Batch 0, alle 3 Codebücher, Sample 0
                    gen_step = offset - start_offset_sequence
                    idx = gen_step % record_window
                    if gen_step >= record_window:
                        old_t = history_buffer[:, idx]
                        self.penalty_histogram[range(self.code_depth), old_t] -= 1
                    history_buffer[:, idx] = new_t
                    self.penalty_histogram[range(self.code_depth), new_t] += 1
                prev_offset = offset
        # ensure sequence has been entirely filled
        assert not (gen_sequence == unknown_token).any()
        max_gen_len = gen_sequence.shape[-1]
        output_codes[..., :max_gen_len] = gen_sequence
        # get back the codes, trimming the prompt if needed and cutting potentially incomplete timesteps
        out_codes, out_indexes, out_mask = pattern.revert_pattern_sequence(output_codes, special_token=unknown_token)
        # sanity checks over the returned codes and corresponding masks
        assert (out_codes != unknown_token).all()
        assert (out_mask == 1).all()
        # ensure the returned codes are all valid
        assert (out_codes >= 0).all() and (out_codes <= self.code_size).all()

        is_music = (out_codes[0, 0, :] < self.special_token_id)
        if is_music.any():
            indices = torch.where(is_music)[0]
            actual_t = indices[-1].item() + 1
            out_codes = out_codes[:, :, :actual_t]
        return out_codes
        
    @torch.inference_mode()
    def _sample_next_token(self,
                           sequence: torch.Tensor,
                           condition_tensors: ConditionTensors,
                           use_sampling: bool = False,
                           temp: float = 1.0,
                           top_k: int = 0,
                           top_p: float = 0.0,
                           cfg_coef: float = None,
                           record_window: int = 0,
                           ignore_tokens: torch.tensor = None) -> torch.Tensor:
        """Sample next token from the model given a sequence and a set of conditions. The model supports
        multiple sampling strategies (greedy sampling, softmax, top-k, top-p...).

        Args:
            sequence (torch.Tensor): Current sequence of shape [B, K, S]
                with K corresponding to the number of codebooks and S the number of sequence steps.
                S = 1 in streaming mode, except for the first step that contains a bigger prompt.
            condition_tensors (dict[str, ConditionType): Set of conditions. If CFG is used,
                should be twice the batch size, being the concatenation of the conditions + null conditions.
            use_sampling (bool): Whether to use a sampling strategy or not.
            temp (float): Sampling temperature.
            top_k (int): K for "top-k" sampling.
            top_p (float): P for "top-p" sampling.
            cfg_coef (float, optional): classifier free guidance coefficient
        Returns:
            next_token (torch.Tensor): Next token tensor of shape [B, K, 1].
        """
        B = sequence.shape[0]
        combined_sequence = torch.cat([sequence, sequence], dim=0)
        cfg_coef = self.cfg_coef if cfg_coef is None else cfg_coef
        all_logits = self(combined_sequence, condition_tensors=condition_tensors)
        cond_logits, uncond_logits = all_logits.split(B, dim=0)
        logits = uncond_logits + (cond_logits - uncond_logits) * cfg_coef
        logits = logits.permute(0, 1, 3, 2)  # [B, K, card, T]
        logits = logits[..., -1]             # [B, K, card]

        if record_window > 0:
            logits /= (1.1 ** self.penalty_histogram)

        if ignore_tokens is not None and ignore_tokens.numel() > 0:
            ignore_tokens = ignore_tokens.to(device=logits.device, dtype=torch.long)
        else:
            ignore_tokens = None
        if ignore_tokens is not None:
            logits.index_fill_(-1, ignore_tokens, float('-inf'))

        if use_sampling and temp > 0.0:
            logits.div_(temp)
            probs = torch.softmax(logits, dim=-1)
            # Codebook 0: Nutzt Top-P oder Top-K für die gewünschte Varianz
            p_first = probs[:, 0, :]
            if top_p > 0.0:
                next_token = sample_top_p(probs, p=top_p)
            elif top_k > 0:
                next_token_first = sample_top_k(probs[:,[0],:], k=top_k)
                next_token_res = sample_top_k(probs[:,1:,:], k=1)
                next_token = torch.cat([next_token_first,next_token_res], dim = 1)
            else:
                next_token = multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        return next_token
