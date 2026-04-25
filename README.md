# SongGeneration - 16GB VRAM Optimized Fork

This is a performance-optimized fork of the original [SongGeneration](https://github.com/tencent-ailab/SongGeneration) project. It is specifically redesigned to run the **v2 Large model** on consumer-grade GPUs with **16GB of VRAM**.

## Key Optimizations

*   **v2 Large on 16GB VRAM**: Achieved through **8-bit µ-law quantization for KV-caching** and **FP16 model conversion** (reducing the model footprint from 13GB to **9.5GB**). Combined with **fused QKV/MLP layers**, these optimizations significantly lower the VRAM entry barrier without sacrificing output quality.
*   **Long-form Generation**: Support for song lengths up to **280 seconds**.
*   **Triple-Phase Memory Management**: The workflow is split into three independent stages to ensure only one model occupies the VRAM at a time.
*   **Precision Balance**: Latent generation is optimized for memory, while the final diffuser stage remains in **full fp32 precision** for high-quality audio output.
*   **Code Cleanup**: Redundant dependencies and unused legacy code have been removed to create a streamlined experience.

## System Requirements

The following setup was used for development and verification. While optimized for AMD hardware, it is architecturally compatible with NVIDIA systems.

*   **GPU**: Minimum 16GB VRAM (Verified on AMD RX 9070).
*   **System RAM**: 32GB System RAM (At least 26GB must be allocated to WSL2).
*   **OS**: Linux or Windows with WSL2.
*   **Environment**: ROCm 7.2.1 with librocdxg, Python 3.12, PyTorch 2.11 (sdpa), Triton-ROCm 3.6.

### Installation

1.  **Base Environment**: Install **PyTorch** with the appropriate backend for your hardware (CUDA or ROCm).
2.  **Dependencies**: 
    ```bash
    pip install -r requirements.txt
    ```

## Model Preparation

Checkpoints are not included and have to be downloaded from HuggingFace. Please see `download_ckpts.sh` and the directory structure under `ckpt/` as a guide.

Run the conversion scripts to prepare the models for the 16GB workflow:  
    *   `python ckpt/songgeneration/convert_fp16.py`  
    *   `python ckpt/songgeneration/convert_ckpt_data_structure.py`  
    *   `python ckpt/model_septoken/convert_fp32.py`  
    *   `python ckpt/model_1rvq/convert_fp32.py`

## Workflow (Three-Phase Process)

To minimize VRAM usage, execute the generation in the following sequence:

1.  **Phase 1: Conditioning** (`jsonl2conditions.sh --input sample/your.jsonl`) – Audio source separation via Demucs if audio prompt is provided.
2.  **Phase 2: Token Generation** (`conditions2tokens.sh`) – v2 Large inference using µ-law cache.
3.  **Phase 3: Audio Synthesis** (`tokens2audio.sh`) – Final rendering using model septoken and VAE.

## Configuration & Input

### Customizing `config.yaml`
#### lyric_processor
- **max_dur** is the maximum song length in seconds. Default is `280`.
#### lm
- **max_position_embeddings** is the maximum kv_cache length for the main transformer. `8210` tokens are needed for a song about 280 seconds long
- **max_position_embeddings_sub** is the maximum kv_cache for the sub transfomer. I use the same value here as I used for main.
- **use_flash_attn_2** `false` activates PyTorch sdpa which on my system is a lot faster than the flash_attn package. If you enable flash attention you automatically use the fp16 kv cache.
- **use_q8_kv_cache** `true` uses the int8 µ-law cache while `false` uses standard fp16 kv-caching.
- **q8_kv_cache_mu** You can experiment with different µ-law values here. `64.0` is the default.

### Input Format (`.jsonl`)
`jsonl2conditions.sh --input_jsonl` expects a JSONL file where each line represents a separate song:  
`{"idx": "unique_songname", "gt_lyric": "[intro-short] ; [verse] lyrics ; [outro-short]"}`  
  See `./conf/vocab.yaml` for structure tags within `gt_lyric`.  

**Optional conditioning:**
- Add `"descriptions": "style and mood description"` for specific text type info for the song.
  See `./sample/description/*` for different type info for `descriptions`.
- Add `"prompt_audio_path": "path/to/file.wav"` for specific audio prompts.
- Add `"auto_prompt_audio_type": "type"` for automatic conditioning.  
  **Supported types:** 'Pop', 'Latin', 'Rock', 'Electronic', 'Metal', 'Country', 'R&B/Soul', 'Ballad', 'Jazz', 'World', 'Hip-Hop', 'Funk', 'Soundtrack' or 'Auto'.
- **Expert Settings:** You can override global settings per song by adding a parameters string:
`"parameters": "temp:0.9, cfg_coef:1.5, record_window:50, top_p:0.9, top_k:500"`

Parameter descriptions:
- **`temp`** – Sampling temperature (higher = more creative, typical 0.6–1.0)
- **`cfg_coef`** – Classifier-Free Guidance coefficient (higher = stronger adherence to description, typical 1.5–4.0)
- **`record_window`** – Token window size for repetition penalty. Larger values more aggressively suppress repeated tokens. `0` disables the feature.
- **`top_p`** – Top-p (nucleus) sampling
- **`top_k`** – Top-k sampling

See examples under `./sample`

## Credits

*   **Original Project**: [Tencent SongGeneration](https://github.com/tencent-ailab/SongGeneration)
