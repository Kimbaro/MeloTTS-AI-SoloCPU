import os
import json
import torch
import numpy as np
import scipy.io.wavfile
from melo.models import SynthesizerTrn
from melo.utils import load_checkpoint

# 간단한 symbol 리스트
symbols = list("가나다라마바사아자차카타파하거너더러머버서어저처커터퍼허 ")  # 필요시 추가
n_symbols = len(symbols)
num_languages = 1  # 단일 언어

# 텍스트 → 숫자 시퀀스
def cleaned_text_to_sequence(text):
    return [symbols.index(c) if c in symbols else 0 for c in text]

# TTS 추론 함수
def generate_tts(text, speaker_id, output_path):
    config_path = "C:/Users/LT2019002/Downloads/MeloTTS_Korean_Metadata_Web/custom_configs/korean_cpu.json"
    ckpt_path = "C:/Users/LT2019002/Downloads/MeloTTS_Korean_Metadata_Web/logs/KR-default/checkpoints/G_50.pth"

    # 설정 파일 로드
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    segment_size = cfg["train"]["segment_size"]

    # 텍스트 입력 처리
    seq = cleaned_text_to_sequence(text)
    x = torch.LongTensor(seq)[None, :].to("cpu")
    x_lengths = torch.LongTensor([x.shape[1]])

    # 모델 초기화
    net_g = SynthesizerTrn(
        n_symbols,
        data_cfg["filter_length"] // 2 + 1,
        segment_size,
        n_speakers=data_cfg["n_speakers"],
        inter_channels=model_cfg["inter_channels"],
        hidden_channels=model_cfg["hidden_channels"],
        filter_channels=model_cfg["filter_channels"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        kernel_size=model_cfg["kernel_size"],
        p_dropout=model_cfg["p_dropout"],
        resblock=model_cfg["resblock"],
        resblock_kernel_sizes=model_cfg["resblock_kernel_sizes"],
        resblock_dilation_sizes=model_cfg["resblock_dilation_sizes"],
        upsample_rates=model_cfg["upsample_rates"],
        upsample_initial_channel=model_cfg["upsample_initial_channel"],
        upsample_kernel_sizes=model_cfg["upsample_kernel_sizes"],
        n_layers_trans_flow=model_cfg["n_layers_trans_flow"],  # ✅ 반드시 추가
    ).to("cpu")
    _ = net_g.eval()

    # 체크포인트 로드 (optimizer 제외)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    net_g.load_state_dict(checkpoint["net_g"])

    # 추론
    with torch.no_grad():
        sid = torch.LongTensor([speaker_id]).to("cpu")
        audio = net_g.infer(x, x_lengths, speakers=sid)[0][0, 0].cpu().numpy()

    # 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    audio = (audio * 32768).astype(np.int16)
    scipy.io.wavfile.write(output_path, data_cfg["sampling_rate"], audio)
