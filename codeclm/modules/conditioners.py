
import torch
import torch.nn as nn
from codeclm.modules.streaming import StreamingModule
import re
from copy import deepcopy

ConditionType = tuple[torch.Tensor, torch.Tensor]  # condition, mask

# ================================================================
# Condition and Condition attributes definitions
# ================================================================

class AudioCondition:
    def __init__(self, tokens, length, sample_rate, path=None, seek_time=None):
        self.tokens = tokens
        self.length = length
        self.sample_rate = sample_rate
        self.path = path if path is not None else []
        self.seek_time = seek_time if seek_time is not None else []

    def __iter__(self):
        yield self.tokens
        yield self.length
        yield self.sample_rate
        yield self.path
        yield self.seek_time

    def __repr__(self):
        if hasattr(self.tokens, 'tolist'):
            # .detach().cpu() stellt sicher, dass wir nicht versehentlich 
            # einen Gradienten-Graphen im Log mitziehen oder CUDA-Fehler provozieren
            content = str(self.tokens.detach().cpu())
        else:
            content = str(self.tokens)

        return (f"AudioCondition( tokens={content}, length={self.length}, sr={self.sample_rate}, path={self.path} )")


class ConditioningAttributes:
    def __init__(self, text=None, audio=None):
        self.text = text if text is not None else {}
        self.audio = audio if audio is not None else {}

    def __getitem__(self, item):
        return getattr(self, item)

    @property
    def text_attributes(self):
        return self.text.keys()

    @property
    def audio_attributes(self):
        return self.audio.keys()

    @property
    def attributes(self):
        return {
            "text": self.text_attributes,
            "audio": self.audio_attributes,
        }

    def to_flat_dict(self):
        return {
            **{f"text.{k}": v for k, v in self.text.items()},
            **{f"audio.{k}": v for k, v in self.audio.items()},
        }

    @classmethod
    def from_flat_dict(cls, x):
        out = cls()
        for k, v in x.items():
            kind, att = k.split(".")
            out[kind][att] = v
        return out

    def __repr__(self):
        return f"ConditioningAttributes(text={self.text}, audio={self.audio})"

# ================================================================
# Conditioner (tokenize and encode raw conditions) definitions
# ================================================================

class BaseConditioner(nn.Module):
    """Base model for all conditioner modules.
    We allow the output dim to be different than the hidden dim for two reasons:
    1) keep our LUTs small when the vocab is large;
    2) make all condition dims consistent.

    Args:
        dim (int): Hidden dim of the model.
        output_dim (int): Output dim of the conditioner.
    """
    def __init__(self, dim: int, output_dim: int, input_token = False, padding_idx=0):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        if input_token:
            self.output_proj = nn.Embedding(dim, output_dim, padding_idx)
        else:
            self.output_proj = nn.Linear(dim, output_dim)

    def tokenize(self, *args, **kwargs):
        """Should be any part of the processing that will lead to a synchronization
        point, e.g. BPE tokenization with transfer to the GPU.

        The returned value will be saved and return later when calling forward().
        """
        raise NotImplementedError()

    def forward(self, inputs) -> ConditionType:
        """Gets input that should be used as conditioning (e.g, genre, description or a waveform).
        Outputs a ConditionType, after the input data was embedded as a dense vector.

        Returns:
            ConditionType:
                - A tensor of size [B, T, D] where B is the batch size, T is the length of the
                  output embedding and D is the dimension of the embedding.
                - And a mask indicating where the padding tokens.
        """
        raise NotImplementedError()
    
class TextConditioner(BaseConditioner):
    ...


class QwTokenizerConditioner(TextConditioner):
    def __init__(self, output_dim: int, 
                 token_path = "",
                 max_len = 300, 
                 add_token_list=[],
                 version: str = 'v1'): #""
        from transformers import Qwen2Tokenizer
        if version != 'v1':
            add_token_list.append('.')
        self.text_tokenizer = Qwen2Tokenizer.from_pretrained(token_path)
        if add_token_list != []:
            self.text_tokenizer.add_tokens(add_token_list, special_tokens=True)
        voc_size = len(self.text_tokenizer.get_vocab())
        # here initialize a output_proj (nn.Embedding) layer
        super().__init__(voc_size, output_dim, input_token=True, padding_idx=151643)
        self.max_len = max_len
        self.padding_idx =' <|endoftext|>'

        vocab = self.text_tokenizer.get_vocab()
        # struct是全部的结构
        struct_tokens = [i for i in add_token_list if i[0]=='[' and i[-1]==']']
        self.struct_token_ids = [vocab[i] for i in struct_tokens]
        self.pad_token_idx = 151643

        self.structure_emb = nn.Embedding(200, output_dim, padding_idx=0)
        # self.split_token_id = vocab["."]
        print("all structure tokens: ", {self.text_tokenizer.convert_ids_to_tokens(i):i for i in self.struct_token_ids})
    def tokenize(self, x: list[str]) -> dict[str, torch.Tensor]:
        x = ['<|im_start|>' + xi if xi is not None else "<|im_start|>" for xi in x]
        # x = [xi if xi is not None else "" for xi in x]
        inputs = self.text_tokenizer(x, return_tensors="pt", padding=True)
        return inputs

    def forward(self, inputs: dict[str, torch.Tensor]) -> ConditionType:
        """
        Add structure embeddings of {verse, chorus, bridge} to text/lyric tokens that
        belong to these structures accordingly, 
        Then delete or keep these structure embeddings.
        """
        mask = inputs['attention_mask']
        tokens = inputs['input_ids']
        B = tokens.shape[0]
        is_sp_embed = torch.any(torch.stack([tokens == i for i in self.struct_token_ids], dim=-1),dim=-1)

        tp_cover_range = torch.zeros_like(tokens)
        for b, is_sp in enumerate(is_sp_embed):
            sp_list = torch.where(is_sp)[0].tolist()
            sp_list.append(mask[b].sum())
            for i, st in enumerate(sp_list[:-1]):
                tp_cover_range[b, st: sp_list[i+1]] = tokens[b, st] - 151645

        if self.max_len is not None:
            if inputs['input_ids'].shape[-1] > self.max_len:
                warnings.warn(f"Max len limit ({self.max_len}) Exceed! \
                              {[self.text_tokenizer.convert_ids_to_tokens(i.tolist()) for i in tokens]} will be cut!")
            tokens = self.pad_2d_tensor(tokens, self.max_len, self.pad_token_idx).to(self.output_proj.weight.device)
            mask = self.pad_2d_tensor(mask, self.max_len, 0).to(self.output_proj.weight.device)
            tp_cover_range = self.pad_2d_tensor(tp_cover_range, self.max_len, 0).to(self.output_proj.weight.device)
        device = self.output_proj.weight.device
        content_embeds = self.output_proj(tokens.to(device))
        structure_embeds = self.structure_emb(tp_cover_range.to(device))
        embeds = content_embeds + structure_embeds

        return embeds, embeds, mask

    def pad_2d_tensor(self, x, max_len, pad_id):
        batch_size, seq_len = x.size()
        pad_len = max_len - seq_len

        if pad_len > 0:
            pad_tensor = torch.full((batch_size, pad_len), pad_id, dtype=x.dtype, device=x.device)
            padded_tensor = torch.cat([x, pad_tensor], dim=1)
        elif pad_len < 0:
            padded_tensor = x[:, :max_len]
        else:
            padded_tensor = x

        return padded_tensor


class QwTextConditioner(TextConditioner):
    def __init__(self, output_dim: int,
                 token_path = "", 
                 max_len = 300,
                 version: str = 'v1'): #""

        from transformers import Qwen2Tokenizer
        self.text_tokenizer = Qwen2Tokenizer.from_pretrained(token_path)
        if version != 'v1':
            self.text_tokenizer.add_tokens(['[Musicality-very-high]', '[Musicality-high]', '[Musicality-medium]', '[Musicality-low]', '[Musicality-very-low]', '[Pure-Music]', '.'], special_tokens=True)
        voc_size = len(self.text_tokenizer.get_vocab())     
        # here initialize a output_proj (nn.Embedding) layer
        super().__init__(voc_size, output_dim, input_token=True, padding_idx=151643) 
        self.max_len = max_len

    def tokenize(self, x: list[str]) -> dict[str, torch.Tensor]:
        x = ['<|im_start|>' + xi if xi is not None else "<|im_start|>" for xi in x]
        inputs = self.text_tokenizer(x, return_tensors="pt", padding=True)
        return inputs

    def forward(self, inputs: dict[str, torch.Tensor], structure_dur = None) -> ConditionType:
        """
        Add structure embeddings of {verse, chorus, bridge} to text/lyric tokens that
        belong to these structures accordingly, 
        Then delete or keep these structure embeddings.
        """
        mask = inputs['attention_mask']
        tokens = inputs['input_ids']

        if self.max_len is not None:
            if inputs['input_ids'].shape[-1] > self.max_len:
                warnings.warn(f"Max len limit ({self.max_len}) Exceed! \
                              {[self.text_tokenizer.convert_ids_to_tokens(i.tolist()) for i in tokens]} will be cut!")
            tokens = self.pad_2d_tensor(tokens, self.max_len, 151643).to(self.output_proj.weight.device)
            mask = self.pad_2d_tensor(mask, self.max_len, 0).to(self.output_proj.weight.device)

        embeds = self.output_proj(tokens)
        return embeds, embeds, mask

    def pad_2d_tensor(self, x, max_len, pad_id):
        batch_size, seq_len = x.size()
        pad_len = max_len - seq_len

        if pad_len > 0:
            pad_tensor = torch.full((batch_size, pad_len), pad_id, dtype=x.dtype, device=x.device)
            padded_tensor = torch.cat([x, pad_tensor], dim=1)
        elif pad_len < 0:
            padded_tensor = x[:, :max_len]
        else:
            padded_tensor = x

        return padded_tensor


class AudioConditioner(BaseConditioner):
    ...

class QuantizedEmbeddingConditioner(AudioConditioner):
    def __init__(self, dim: int, 
                 code_size: int, 
                 code_depth: int, 
                 max_len: int, 
                 **kwargs):
        super().__init__(dim, dim, input_token=True)
        self.code_depth = code_depth
        # add 1 for <s> token
        self.emb = nn.ModuleList([nn.Embedding(code_size+2, dim, padding_idx=code_size+1) for _ in range(code_depth)])
        # add End-Of-Text embedding
        self.EOT_emb = nn.Parameter(torch.randn(1, dim), requires_grad=True)
        self.layer2_EOT_emb = nn.Parameter(torch.randn(1, dim), requires_grad=True)
        self.output_proj = None
        self.max_len = max_len
        self.vocab_size = code_size

    def tokenize(self, x: AudioCondition) -> AudioCondition:
        """no extra ops"""
        return x

    def forward(self, x: AudioCondition):
        tokens, lengths, *_ = x
        B = tokens.shape[0]
        tokens = tokens.reshape(B, self.code_depth, -1).long()
        target_seq_len = self.max_len - 1
        current_seq_len = tokens.shape[-1]
        if current_seq_len < target_seq_len:
            tokens = F.pad(tokens, [0, target_seq_len - current_seq_len], value=self.vocab_size+1)
        else:
            tokens = tokens[:, :, :target_seq_len]
        embeds1 = self.emb[0](tokens[:, 0])
        embeds1 = torch.cat((self.EOT_emb.expand(B, 1, -1), embeds1), dim=1)
        temp_list = [self.emb[k](tokens[:, k]) for k in range(1, self.code_depth)] # B,T,D
        embeds2 = torch.stack(temp_list).sum(dim=0)
        embeds2 = torch.cat((self.layer2_EOT_emb.expand(B, 1, -1), embeds2), dim=1)
        if lengths is not None:
            lengths = torch.clamp(lengths + 1, max=self.max_len)
            assert len(lengths.shape) == 1, "Length shape should be 1 dimensional."
            mask = (torch.arange(self.max_len, device=lengths.device)[None, :] < lengths[:, None]).int()
        else:
            mask = torch.ones((B, self.max_len), device=embeds1.device, dtype=torch.int)
        return embeds1, embeds2, mask


# ================================================================
# Aggregate all conditions and corresponding conditioners
# ================================================================
class ConditionerProvider(nn.Module):
    """Prepare and provide conditions given all the supported conditioners.

    Args:
        conditioners (dict): Dictionary of conditioners.
        device (torch.device or str, optional): Device for conditioners and output condition types.
    """
    def __init__(self, conditioners: dict[str, BaseConditioner]):
        super().__init__()
        self.conditioners = nn.ModuleDict(conditioners)

    @property
    def text_conditions(self):
        return [k for k, v in self.conditioners.items() if isinstance(v, TextConditioner)]

    @property
    def audio_conditions(self):
        return [k for k, v in self.conditioners.items() if isinstance(v, AudioConditioner)]

    @property
    def has_audio_condition(self):
        return len(self.audio_conditions) > 0

    def tokenize(self, inputs: list[ConditioningAttributes]) -> dict[str, object]:
        """Match attributes/audios with existing conditioners in self, and compute tokenize them accordingly.
        This should be called before starting any real GPU work to avoid synchronization points.
        This will return a dict matching conditioner names to their arbitrary tokenized representations.

        Args:
            inputs (list[ConditioningAttributes]): List of ConditioningAttributes objects containing
                text and audio conditions.
        """
        assert all([isinstance(x, ConditioningAttributes) for x in inputs]), (
            "Got unexpected types input for conditioner! should be list[ConditioningAttributes]",
            f" but types were {set([type(x) for x in inputs])}")

        output = {}
        text = self._collate_text(inputs)
        audios = self._collate_audios(inputs)

        assert set(text.keys() | audios.keys()).issubset(set(self.conditioners.keys())), (
            f"Got an unexpected attribute! Expected {self.conditioners.keys()}, ",
            f"got {text.keys(), audios.keys()}")

        combined_items = text | audios
        for attribute, batch in combined_items.items():
            output[attribute] = self.conditioners[attribute].tokenize(batch)
        return output

    def forward(self, tokenized: dict[str, object], structure_dur = None) -> dict[str, ConditionType]:
        """Compute pairs of `(embedding, mask)` using the configured conditioners and the tokenized representations.
        The output is for example:
        {
            "genre": (torch.Tensor([B, 1, D_genre]), torch.Tensor([B, 1])),
            "description": (torch.Tensor([B, T_desc, D_desc]), torch.Tensor([B, T_desc])),
            ...
        }

        Args:
            tokenized (dict): Dict of tokenized representations as returned by `tokenize()`.
        """
        output = {}
        for attribute, inputs in tokenized.items():
            if attribute == 'description' and structure_dur is not None:
                condition1, condition2, mask = self.conditioners[attribute](inputs, structure_dur = structure_dur)
            else:
                condition1, condition2, mask = self.conditioners[attribute](inputs)
            output[attribute] = (condition1, condition2, mask)
        return output

    def _collate_text(self, samples: list[ConditioningAttributes]) -> dict[str, list[str]]:
        """Given a list of ConditioningAttributes objects, compile a dictionary where the keys
        are the attributes and the values are the aggregated input per attribute.
        For example:
        Input:
        [
            ConditioningAttributes(text={"genre": "Rock", "description": "A rock song with a guitar solo"}, wav=...),
            ConditioningAttributes(text={"genre": "Hip-hop", "description": "A hip-hop verse"}, audio=...),
        ]
        Output:
        {
            "genre": ["Rock", "Hip-hop"],
            "description": ["A rock song with a guitar solo", "A hip-hop verse"]
        }

        Args:
            samples (list of ConditioningAttributes): List of ConditioningAttributes samples.
        Returns:
            dict[str, list[str, optional]]: A dictionary mapping an attribute name to text batch.
        """
        out = {condition: [] for condition in self.text_conditions}
        for x in samples:
            for condition in self.text_conditions:
                out[condition].append(x.text[condition])
        return out

    def _collate_audios(self, samples: list[ConditioningAttributes]) -> dict[str, AudioCondition]:
        """Generate a dict where the keys are attributes by which we fetch similar audios,
        and the values are Tensors of audios according to said attributes.

        *Note*: by the time the samples reach this function, each sample should have some audios
        inside the "audio" attribute. It should be either:
        1. A real audio
        2. A null audio due to the sample having no similar audios (nullified by the dataset)
        3. A null audio due to it being dropped in a dropout module (nullified by dropout)

        Args:
            samples (list of ConditioningAttributes): List of ConditioningAttributes samples.
        Returns:
            dict[str, WavCondition]: A dictionary mapping an attribute name to wavs.
        """
        tokens = {attr: [] for attr in self.audio_conditions}
        lengths = {attr: [] for attr in self.audio_conditions}
        sample_rates = {attr: [] for attr in self.audio_conditions}
        paths = {attr: [] for attr in self.audio_conditions}
        seek_times = {attr: [] for attr in self.audio_conditions}
        out: dict[str, AudioCondition] = {}

        for sample in samples:
            for attribute in self.audio_conditions:
                token, length, sample_rate, path, seek_time = sample.audio[attribute]
                assert token.dim() == 3, f"Got tokens with dim={token.dim()}, but expected 3 [1, C, T]"
                assert token.size(0) == 1, f"Got tokens [B, C, T] with shape={token.shape}, but expected B == 1"
                tokens[attribute].append(token.squeeze(0).transpose(0, 1))
                lengths[attribute].append(length)
                sample_rates[attribute].extend(sample_rate)
                paths[attribute].extend(path)
                seek_times[attribute].extend(seek_time)

        # stack all tokens to a single tensor
        for attribute in self.audio_conditions:
            conditioner = self.conditioners[attribute]
            pad_val = getattr(conditioner, 'vocab_size', 0) + 1
            padded_tokens = torch.nn.utils.rnn.pad_sequence(tokens[attribute], batch_first=False, padding_value=pad_val)
            stacked_tokens = padded_tokens.permute(1, 2, 0)
            out[attribute] = AudioCondition(
                stacked_tokens, torch.cat(lengths[attribute]), sample_rates[attribute],
                paths[attribute], seek_times[attribute])

        return out


class ConditionFuser(StreamingModule):
    """Condition fuser handles the logic to combine the different conditions
    to the actual model input.

    Args:
        fuse2cond (dict[str, str]): A dictionary that says how to fuse
            each condition. For example:
            {
                "prepend": ["description"],
                "sum": ["genre", "bpm"],
            }
    """
    FUSING_METHODS = ["sum", "prepend"] #, "cross", "input_interpolate"] (not support in this simplest version)

    def __init__(self, fuse2cond: dict[str, list[str]]):
        super().__init__()
        assert all([k in self.FUSING_METHODS for k in fuse2cond.keys()]
        ), f"Got invalid fuse method, allowed methods: {self.FUSING_METHODS}"
        self.fuse2cond: dict[str, list[str]] = fuse2cond
        self.cond2fuse: dict[str, str] = {}
        for fuse_method, conditions in fuse2cond.items():
            for condition in conditions:
                self.cond2fuse[condition] = fuse_method

    def forward(
        self,
        input1: torch.Tensor,
        input2: torch.Tensor,
        conditions: dict[str, ConditionType]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Fuse the conditions to the provided model input.

        Args:
            input (torch.Tensor): Transformer input.
            conditions (dict[str, ConditionType]): Dict of conditions.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: The first tensor is the transformer input
                after the conditions have been fused. The second output tensor is the tensor
                used for cross-attention or None if no cross attention inputs exist.
        """
        B, T, _ = input1.shape

        if 'offsets' in self._streaming_state:
            first_step = False
            offsets = self._streaming_state['offsets']
        else:
            first_step = True
            offsets = torch.zeros(input1.shape[0], dtype=torch.long, device=input1.device)

        assert set(conditions.keys()).issubset(set(self.cond2fuse.keys())), \
            f"given conditions contain unknown attributes for fuser, " \
            f"expected {self.cond2fuse.keys()}, got {conditions.keys()}"

        # if 'prepend' mode is used,
        # the concatenation order will be the SAME with the conditions in config:
        # prepend: ['description', 'prompt_audio'] (then goes the input)
        fused_input_1 = input1
        fused_input_2 = input2
        for fuse_op in self.fuse2cond.keys():
            fuse_op_conditions = self.fuse2cond[fuse_op]
            if fuse_op == 'sum' and len(fuse_op_conditions) > 0:
                for cond in fuse_op_conditions:
                    this_cond_1, this_cond_2, cond_mask = conditions[cond]
                    fused_input_1 += this_cond_1
                    fused_input_2 += this_cond_2
            elif fuse_op == 'prepend' and len(fuse_op_conditions) > 0:
                if not first_step:
                    continue
                reverse_list = deepcopy(fuse_op_conditions)
                reverse_list.reverse()
                for cond in reverse_list:
                    this_cond_1, this_cond_2, cond_mask = conditions[cond]
                    fused_input_1 = torch.cat((this_cond_1, fused_input_1), dim=1)  # concat along T dim
                    fused_input_2 = torch.cat((this_cond_2, fused_input_2), dim=1)  # concat along T dim
            elif fuse_op not in self.FUSING_METHODS:
                raise ValueError(f"unknown op ({fuse_op})")

        if self._is_streaming:
            self._streaming_state['offsets'] = offsets + T

        return fused_input_1, fused_input_2

# ================================================================
# Condition Dropout
# ================================================================
class DropoutModule(nn.Module):
    """Base module for all dropout modules."""
    def __init__(self, seed: int = 1234):
        super().__init__()
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)


class ClassifierFreeGuidanceDropout(DropoutModule):
    """Classifier Free Guidance dropout.
    All attributes are dropped with the same probability.

    Args:
        p (float): Probability to apply condition dropout during training.
        seed (int): Random seed.
    """
    def __init__(self, p: float, seed: int = 1234):
        super().__init__(seed=seed)
        self.p = p

    def check(self, sample, condition_type, condition):
        if condition_type not in ['text', 'audio']:
            raise ValueError("dropout_condition got an unexpected condition type!"
                f" expected 'text', 'audio' but got '{condition_type}'")
        if condition not in getattr(sample, condition_type):
            raise ValueError(
                "dropout_condition received an unexpected condition!"
                f" expected audio={sample.audio.keys()} and text={sample.text.keys()}"
                f" but got '{condition}' of type '{condition_type}'!")

    def get_padding_tokens(self, tokens, sr=48000) -> AudioCondition: 
        out = tokens * 0 + 16385
        return AudioCondition(
            tokens=out, 
            length=torch.Tensor([0]).long(),
            sample_rate=[sr])

    def dropout_condition(self, 
                          sample: ConditioningAttributes, 
                          condition_type: str, 
                          condition: str) -> ConditioningAttributes:
        """Utility function for nullifying an attribute inside an ConditioningAttributes object.
        If the condition is of type "wav", then nullify it using `nullify_condition` function.
        If the condition is of any other type, set its value to None.
        Works in-place.
        """
        self.check(sample, condition_type, condition)

        if condition_type == 'audio':
            audio_cond = sample.audio[condition] 
            sample.audio[condition] = self.get_padding_tokens(audio_cond.tokens, sr=audio_cond.sample_rate[0])
        else:
            sample.text[condition] = None

        return sample

    def forward(self, samples: list[ConditioningAttributes]) -> list[ConditioningAttributes]:
        """
        Args:
            samples (list[ConditioningAttributes]): List of conditions.
        Returns:
            list[ConditioningAttributes]: List of conditions after all attributes were set to None.
        """
        # decide on which attributes to drop in a batched fashion
        # drop = torch.rand(1, generator=self.rng).item() < self.p
        # if not drop:
        #     return samples

        # nullify conditions of all attributes
        samples = deepcopy(samples)

        for sample in samples:
            drop = torch.rand(1, generator=self.rng).item()
            if drop<self.p:
                for condition_type in ["audio", "text"]:
                    for condition in sample.attributes[condition_type]:
                        self.dropout_condition(sample, condition_type, condition)
        return samples

    def __repr__(self):
        return f"ClassifierFreeGuidanceDropout(p={self.p})"


class ClassifierFreeGuidanceDropoutInference(ClassifierFreeGuidanceDropout):
    """Classifier Free Guidance dropout during inference.
    All attributes are dropped with the same probability.

    Args:
        p (float): Probability to apply condition dropout during training.
        seed (int): Random seed.
    """
    def __init__(self, seed: int = 1234):
        super().__init__(p=1, seed=seed)
        self._struct_re = re.compile(r'\[inst\]|\[outro\]|\[intro\]|\[verse\]|\[chorus\]|\[bridge\]')

    def dropout_condition_customized(self,
                                    sample: ConditioningAttributes,
                                    condition_type: str, 
                                    condition: str,
                                    customized: list = None) -> ConditioningAttributes:
        """Utility function for nullifying an attribute inside an ConditioningAttributes object.
        If the condition is of type "audio", then nullify it using `nullify_condition` function.
        If the condition is of any other type, set its value to None.
        Works in-place.
        """
        self.check(sample, condition_type, condition)

        if condition_type == 'audio':
            audio_cond = sample.audio[condition]
            sample.audio[condition] = self.get_padding_tokens(audio_cond.tokens, sr=audio_cond.sample_rate[0])
        elif customized is None:
            val = sample.text.get(condition)
            if condition in ['type_info'] and val and "[Musicality-very-high]" in val:
                sample.text[condition] = "[Musicality-very-low], ."
                print(f"cfg unconditioning: change sample.text[condition] to [Musicality-very-low]")
            else:
                sample.text[condition] = None
        else:
            text_cond = sample.text[condition]
            if "structure" in customized:
                text_cond = self._struct_re.sub("", text_cond)
                text_cond = text_cond.replace(' , ', '').replace("  ", " ")
            if '.' in customized:
                text_cond = text_cond.replace(".", " ").replace("  ", " ")

            sample.text[condition] = text_cond.strip()

        return sample

    def forward(self, samples: list[ConditioningAttributes],
                condition_types=["tokens", "text"],
                customized=None,
                ) -> list[ConditioningAttributes]:
        """
        100% dropout some condition attributes (description, prompt_tokens) or types (text, tokens) of
        samples during inference.

        Args:
            samples (list[ConditioningAttributes]): List of conditions.
        Returns:
            list[ConditioningAttributes]: List of conditions after all attributes were set to None.
        """
        new_samples = deepcopy(samples)
        for condition_type in condition_types:
            for sample in new_samples:
                target_dict = getattr(sample, condition_type, {})
                for condition in list(target_dict.keys()):
                    self.dropout_condition_customized(sample, condition_type, condition, customized)
        return new_samples


class AttributeDropout(ClassifierFreeGuidanceDropout):
    """Dropout with a given probability per attribute.
    This is different from the behavior of ClassifierFreeGuidanceDropout as this allows for attributes
    to be dropped out separately. For example, "artist" can be dropped while "genre" remains.
    This is in contrast to ClassifierFreeGuidanceDropout where if "artist" is dropped "genre"
    must also be dropped.

    Args:
        p (dict[str, float]): A dict mapping between attributes and dropout probability. For example:
            ...
            "genre": 0.1,
            "artist": 0.5,
            "audio": 0.25,
            ...
        active_on_eval (bool, optional): Whether the dropout is active at eval. Default to False.
        seed (int, optional): Random seed.
    """
    def __init__(self, p: dict[str, dict[str, float]], active_on_eval: bool = False, seed: int = 1234):
        super().__init__(p=p, seed=seed)
        self.active_on_eval = active_on_eval
        # construct dict that return the values from p otherwise 0
        self.p = p

    def forward(self, samples: list[ConditioningAttributes]) -> list[ConditioningAttributes]:
        """
        Args:
            samples (list[ConditioningAttributes]): List of conditions.
        Returns:
            list[ConditioningAttributes]: List of conditions after certain attributes were set to None.
        """
        if not self.training and not self.active_on_eval:
            return samples

        samples = deepcopy(samples)
        for condition_type, ps in self.p.items():  # for condition types [text, tokens]
            for condition, p in ps.items():  # for attributes of each type (e.g., [artist, genre])
                if torch.rand(1, generator=self.rng).item() < p:
                    for sample in samples:
                        self.dropout_condition(sample, condition_type, condition)
        return samples
