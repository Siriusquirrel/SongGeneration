curl -L https://huggingface.co/tencent/SongGeneration/resolve/main/third_party/demucs/ckpt/htdemucs.pth -o ckpt/htdemucs/htdemucs.pth
curl -L https://huggingface.co/tencent/SongGeneration/resolve/main/ckpt/vae/autoencoder_music_1320k.ckpt -o ckpt/vae/autoencoder_music_1320k.ckpt
curl -L https://huggingface.co/tencent/SongGeneration/resolve/main/ckpt/model_1rvq/model_2_fixed.safetensors -o ckpt/model_1rvq/model_1rvq.safetensors
curl -L https://huggingface.co/tencent/SongGeneration/resolve/main/ckpt/model_septoken/model_2.safetensors -o ckpt/model_septoken/model_septoken.safetensors
curl -L https://huggingface.co/lglg666/SongGeneration-v2-large/resolve/main/model.pt -o ckpt/songgeneration/model_v2_large_orig.pt
