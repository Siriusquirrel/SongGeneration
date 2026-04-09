"""
Tokenizer or wrapper around existing models.
Also defines the main interface that a model must follow to be usable as an audio tokenizer.
"""

import torch
from torch import nn
# from codeclm.tokenizer.Flow1dVAE.generate_1rvq import Tango
# from codeclm.tokenizer.Flow1dVAE.generate_septoken import Tango


class AudioTokenizer(nn.Module):
    """Base API for all compression model that aim at being used as audio tokenizers
    with a language model.
    """
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """See `EncodecModel.encode`."""
        ...

    def decode(self, codes: torch.Tensor, scale: torch.Tensor = None):
        """See `EncodecModel.decode`."""
        ...

    def decode_latent(self, codes: torch.Tensor):
        """Decode from the discrete codes to continuous latent space."""
        ...


class Flow1dVAE1rvq(AudioTokenizer):
    def __init__(
        self,
        path: str = "model_2_fixed.safetensors",
        cfg: str = "",
        tango_device: str = "cuda"
        ):
        super().__init__()

        from codeclm.tokenizer.Flow1dVAE.generate_1rvq import Tango
        self.model = Tango(model_path=path, cfg=cfg, device=tango_device)
        print ("Successfully loaded checkpoint from:", path)
        self.n_quantizers = 1

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        codes = self.model.sound2code(x) # [B T] -> [B N T]
        return codes, None

    @torch.no_grad()
    def decode(self, codes: torch.Tensor, prompt = None, scale: torch.Tensor = None, ncodes=9):
        wav = self.model.code2sound(codes, prompt=prompt, guidance_scale=1.5,
                                    num_steps=10, disable_progress=False) # [B,N,T] -> [B,T]
        return wav[None]

    @torch.no_grad()
    def decode_latent(self, codes: torch.Tensor):
        """Decode from the discrete codes to continuous latent space."""
        return self.model.quantizer.from_codes(codes.transpose(1,2))[0]

    def to(self, device=None, dtype=None, non_blocking=False):
        self = super(Flow1dVAE1rvq, self).to(device, dtype, non_blocking)
        self.model = self.model.to(device, dtype, non_blocking)
        return self

    def cuda(self, device=None):
        if device is None:
            device = 'cuda:0'
        return super(Flow1dVAE1rvq, self).cuda(device)

class Flow1dVAESeparate(AudioTokenizer):
    def __init__(
        self,
        path: str = "model_2.safetensors",
        cfg: str = "",
        tango_device: str = "cuda"
        ):
        super().__init__()

        from codeclm.tokenizer.Flow1dVAE.generate_septoken import Tango
        self.model = Tango(model_path=path, cfg=cfg, device=tango_device)
        print ("Successfully loaded checkpoint from:", path)
        self.n_quantizers = 1

    @torch.no_grad()
    def encode(self, x_vocal: torch.Tensor, x_bgm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if x_vocal.ndim == 2:
            x_vocal = x_vocal.unsqueeze(1)
        if x_bgm.ndim == 2:
            x_bgm = x_bgm.unsqueeze(1)
        codes_vocal, codes_bgm = self.model.sound2code(x_vocal, x_bgm)
        return codes_vocal, codes_bgm

    @torch.no_grad()
    def decode(self, codes: torch.Tensor, prompt_vocal = None, prompt_bgm = None, chunked=False, chunk_size=128):
        # [B,N,T] -> [B,T]
        wav = self.model.code2sound(codes, prompt_vocal=prompt_vocal, prompt_bgm=prompt_bgm, guidance_scale=1.5,
                                    num_steps=10, disable_progress=False, chunked=chunked, chunk_size=chunk_size)
        return wav[None]

    @torch.no_grad()
    def decode_latent(self, codes: torch.Tensor):
        """Decode from the discrete codes to continuous latent space."""
        return self.model.quantizer.from_codes(codes.transpose(1,2))[0]

    def to(self, device=None, dtype=None, non_blocking=False):
        self = super(Flow1dVAESeparate, self).to(device, dtype, non_blocking)
        self.model = self.model.to(device, dtype, non_blocking)
        return self

    def cuda(self, device=None):
        if device is None:
            device = 'cuda:0'
        self = super(Flow1dVAESeparate, self).cuda(device)
        return self
