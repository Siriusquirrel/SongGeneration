# Copyright (c) 2026 Siriusquirrel
# Part of the SongGeneration-v2-Large-16GB-Fork

import soundfile as sf
import torch

def load_audio(path):
    audio_np, sr = sf.read(path, dtype='float32', always_2d=True)
    assert sr == 48000, f"Audio must be 48000 Hz"
    # soundfile: (samples, channels)
    # torch: (channels, samples)
    audio = torch.from_numpy(audio_np.T)
    # Automatische Stereo-Korrektur:
    if audio.shape[0] == 1:
        # Falls Mono (1, L), dupliziere auf Stereo (2, L)
        print(f"INFO: Converting mono file {path} to stereo.")
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        # Falls mehr als 2 Kanäle, nimm nur die ersten beiden
        audio = audio[:2, :]
    return audio, sr

def save_audio(path, audio, sr=48000):
    # Von GPU holen und SICHER in Float32 wandeln
    # Das löst Probleme mit float16/bfloat16 Modellen
    audio = audio.detach().cpu().float()
    # Batch-Dimension entfernen, falls vorhanden [1, C, L] -> [C, L]
    if audio.ndim == 3:
        audio = audio.squeeze(0)
    # Transponieren für Soundfile [C, L] -> [L, C]
    audio_np = audio.numpy().T
    # Speichern (Der Subtype FLOAT sorgt für 32-bit WAV)
    sf.write(path, audio_np, sr, subtype='FLOAT')
