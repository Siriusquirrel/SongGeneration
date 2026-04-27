import torch
import torch.nn as nn
import torchaudio

from descript_quantize3 import ResidualVectorQuantize
from musicfm.musicfm_model import MusicFMModel, MusicFMConfig
from normalization import Feature1DProcessor

class PromptCondAudioDiffusion(nn.Module):
    def __init__(
        self
    ):
        super().__init__()

        self.bestrq = torch.compile(MusicFMModel(MusicFMConfig()), mode="max-autotune")
        self.rsq48tobestrq = torchaudio.transforms.Resample(48000, 24000)

        self.rvq_bestrq_emb = ResidualVectorQuantize(input_dim = 1024, n_codebooks = 1, codebook_size = 16_384, codebook_dim = 32, quantizer_dropout = 0.0, stale_tolerance=200)
        for v in self.rvq_bestrq_emb.parameters():v.requires_grad = False
        self.rvq_bestrq_emb = torch.compile(self.rvq_bestrq_emb, mode="reduce-overhead")

        print("1rvq Audio Tokenizer initialized from pretrain.")
        torch.cuda.empty_cache()

    def preprocess_audio(self, input_audios, threshold=0.9):
        assert len(input_audios.shape) == 2, input_audios.shape
        norm_value = torch.ones_like(input_audios[:,0])
        max_volume = input_audios.abs().max(dim=-1)[0]
        norm_value[max_volume>threshold] = max_volume[max_volume>threshold] / threshold
        return input_audios/norm_value.unsqueeze(-1)

    def extract_bestrq_embeds(self, input_audio_0,input_audio_1,layer):
        self.bestrq.eval()
        input_wav_mean = (input_audio_0 + input_audio_1) / 2.0
        input_wav_mean = self.bestrq(self.rsq48tobestrq(input_wav_mean), features_only = True)
        layer_results = input_wav_mean['layer_results']
        bestrq_emb = layer_results[layer]
        bestrq_emb = bestrq_emb.permute(0,2,1).contiguous()
        return bestrq_emb

    def init_device_dtype(self, device, dtype):
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def fetch_codes_batch(self, input_audios, additional_feats,layer):
        input_audio_0 = input_audios[:,0,:]
        input_audio_1 = input_audios[:,1,:]
        input_audio_0 = self.preprocess_audio(input_audio_0)
        input_audio_1 = self.preprocess_audio(input_audio_1)

        bestrq_emb = self.extract_bestrq_embeds(input_audio_0,input_audio_1,layer)
        bestrq_emb = bestrq_emb.detach()

        self.rvq_bestrq_emb.eval()
        quantized_bestrq_emb, codes_bestrq_emb, *_ = self.rvq_bestrq_emb(bestrq_emb) # b,d,t

        return [codes_bestrq_emb]

    @torch.no_grad()
    def inference_codes(self, *args, **kwargs):
        raise NotImplementedError("1rvq model does not support inference. Use septoken for decoding.")
