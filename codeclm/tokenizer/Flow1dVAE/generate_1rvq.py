# Original work Copyright (c) Tencent AI Lab
# Refactoring and modifications Copyright (c) 2026 Siriusquirrel
#
# Part of the SongGeneration-v2-Large-16GB-Fork

import torch
from model_1rvq import PromptCondAudioDiffusion
from sat_1dvae_large import get_model
from safetensors.torch import load_file

class Tango:
    def __init__(self, \
        model_path, \
        cfg,
        layer_num=6, \
        device="cuda:0"):
        
        self.sample_rate = 48000
        self.device = device

        self.vae = get_model(cfg.vae_config, cfg.vae_model)
        self.vae = self.vae.to(device).eval()
        self.layer_num = layer_num

        self.MAX_DURATION = cfg.lyric_processor.max_dur
        self.model = PromptCondAudioDiffusion()
        if model_path.endswith(".safetensors"):
            main_weights = load_file(model_path)
        else:
            main_weights = torch.load(model_path, map_location="cpu", mmap=True, weights_only=True)
        self.model.load_state_dict(main_weights, strict=False, assign=True)
        self.model.to(device)
        print ("Successfully loaded checkpoint from:", model_path)
        self.model.eval()
        self.model.init_device_dtype(torch.device(device), torch.float32)

    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def sound2code(self, orig_samples, batch_size=3):
        if(orig_samples.ndim == 2):
            audios = orig_samples.unsqueeze(0).to(self.device)
        elif(orig_samples.ndim == 3):
            audios = orig_samples.to(self.device)
        else:
            assert orig_samples.ndim in (2,3), orig_samples.shape
#        audios = self.preprocess_audio(audios) # is done in model_1rvq
        audios = audios.squeeze(0)
        orig_length = audios.shape[-1]
        min_samples = int(40 * self.sample_rate)
        # 40秒对应10个token
        output_len = int(orig_length / float(self.sample_rate) * 25) + 1
        print("output_len: ", output_len)

        while(audios.shape[-1] < min_samples):
            audios = torch.cat([audios, audios], -1)
        int_max_len=audios.shape[-1]//min_samples+1
        audios = torch.cat([audios, audios], -1)
        audios=audios[:,:int(int_max_len*(min_samples))]
        codes_list=[]

        audio_input = audios.reshape(2, -1, min_samples).permute(1, 0, 2).reshape(-1, 2, min_samples)

        for audio_inx in range(0, audio_input.shape[0], batch_size):
            # import pdb; pdb.set_trace()
            codes = self.model.fetch_codes_batch((audio_input[audio_inx:audio_inx+batch_size]), additional_feats=[],layer=self.layer_num)
            codes_list.append(torch.cat(codes, 1))
            # print("codes_list",codes_list[0].shape)

        codes = torch.cat(codes_list, 0).permute(1,0,2).reshape(1, -1)[None] # B 3 T -> 3 B T
        codes=codes[:,:,:output_len]

        return codes

    @torch.no_grad()
    def code2sound(self, *args, **kwargs):
        raise NotImplementedError("1rvq model does not support code2sound decoding. Use septoken for audio synthesis.")

    @torch.no_grad()
    def sound2sound(self, *args, **kwargs):
        raise NotImplementedError("1rvq model does not support sound2sound. Use septoken for audio synthesis.")

    @torch.no_grad()
    def sound2sound_vae(self, *args, **kwargs):
        raise NotImplementedError("1rvq model does not support sound2sound_vae. Use septoken for audio synthesis.")

    def to(self, device=None, dtype=None, non_blocking=False):
        if device is not None:
            self.device = device
            self.model.device = device
        self.vae = self.vae.to(device, dtype, non_blocking)
        self.model = self.model.to(device, dtype, non_blocking)
        return self
