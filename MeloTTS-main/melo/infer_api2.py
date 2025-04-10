import numpy as np
import torch
import scipy.io.wavfile
from melo.api import TTS
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS(language="KR", device=device)

def generate_tts(text, gender, emotion, tone, output_path):
    wav = tts.infer(  # ← 여기! .tts가 아니라 .infer 또는 .__call__!
        text,
        speaker=gender,
        emotion=emotion,
        tone=tone,
    )
    wav = wav * 32768
    wav = np.clip(wav, -32768, 32767)
    wav = wav.astype(np.int16)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    scipy.io.wavfile.write(output_path, 22050, wav)
