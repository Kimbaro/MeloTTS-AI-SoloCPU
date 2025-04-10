# app2.py

import os
import json
import torch
import numpy as np
import unicodedata
import scipy.io.wavfile
from flask import Flask, render_template_string, request
from melo.models import SynthesizerTrn

# symbols 불러오기 (219개 or 220개)
from text.symbols import symbols
n_symbols = len(symbols)
print("n_symbols 갯수 확인 :: ", n_symbols)

app = Flask(__name__)

# 자모 분리 및 시퀀스 변환
def cleaned_text_to_sequence(text):
    jamo_text = unicodedata.normalize("NFD", text)
    print("자모 분리된 텍스트:", jamo_text)
    seq = [symbols.index(c) if c in symbols else 0 for c in jamo_text]
    return seq

# TTS 생성
def generate_tts(text, speaker_id, output_path):
    print("output_path 체크 :: ", output_path)
    # config_path = "custom_configs/korean_cpu.json"
    # ckpt_path = "logs/KR-default/checkpoints/G_50.pth"

    config_path = "C:/Users/LT2019002/Downloads/MeloTTS_Korean_Metadata_Web/custom_configs/korean_cpu.json"
    ckpt_path = "C:/Users/LT2019002/Downloads/MeloTTS_Korean_Metadata_Web/logs/KR-default/checkpoints/G_50.pth"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    segment_size = cfg["train"]["segment_size"]

    seq = cleaned_text_to_sequence(text)
    x = torch.LongTensor(seq)[None, :].to("cpu")
    x_lengths = torch.LongTensor([x.shape[1]])

    net_g = SynthesizerTrn(
        n_vocab=n_symbols,
        spec_channels=data_cfg["filter_length"] // 2 + 1,
        segment_size=segment_size,
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
        use_spectral_norm=model_cfg.get("use_spectral_norm", False),
        n_layers_trans_flow=model_cfg["n_layers_trans_flow"]
    ).to("cpu")
    net_g.eval()

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    net_g.load_state_dict(checkpoint["net_g"], strict=True)

    with torch.no_grad():
        sid = torch.LongTensor([speaker_id]).to("cpu")
        tone = torch.LongTensor([[0]]).to("cpu")
        language = torch.LongTensor([[4]]).to("cpu")  # KR
        bert = torch.zeros((1, 1024, x.shape[1])).to("cpu")
        ja_bert = torch.zeros((1, 768, x.shape[1])).to("cpu")

        audio = net_g.infer(x, x_lengths, sid, tone, language, bert, ja_bert)[0][0, 0].cpu().numpy()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    audio = np.clip(audio * 5.0, -1.0, 1.0)
    audio = (audio / np.max(np.abs(audio)) * 32767).astype(np.int16)
    scipy.io.wavfile.write(output_path, data_cfg["sampling_rate"], audio)

    print("샘플레이트:", data_cfg["sampling_rate"])
    print("오디오 샘플 수:", len(audio))
    print("오디오 최대값:", np.max(np.abs(audio)))

# 웹 템플릿
HTML_TEMPLATE = """
<!doctype html>
<title>TTS Infer</title>
<h2>Text-to-Speech</h2>
<form method=post>
  텍스트: <input type=text name=text><br><br>
  성별: 
  <select name=speaker_id>
    <option value=0>남성</option>
    <option value=1>여성</option>
  </select><br><br>
  <input type=submit value=생성하기>
</form>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form["text"]
        speaker_id = int(request.form["speaker_id"])
        output_path = "static/audio.wav"
        generate_tts(text, speaker_id, output_path)
        return f"<h3>음성 생성 완료!</h3><audio controls src='/static/audio.wav'></audio>"
    return render_template_string(HTML_TEMPLATE)

if __name__ == "__main__":
    app.run(debug=True)


# import os
# import json
# import torch
# import numpy as np
# import scipy.io.wavfile
# from flask import Flask, render_template_string, request
# from melo.models import SynthesizerTrn
# # import hgtk  # 자모 분리를 위한 라이브러리
# import unicodedata
#
# # ✅ 심볼 리스트: 학습에 사용된 219개 symbols 불러오기
# from text.symbols import symbols  # 반드시 219개짜리 원본으로!
# n_symbols = len(symbols)
#
# print("n_symbols 갯수 확인 :: ", n_symbols)
#
# # Flask 앱 생성
# app = Flask(__name__)
#
# # 텍스트를 시퀀스로 변환
# # def cleaned_text_to_sequence(text):
# #     return [symbols.index(c) if c in symbols else 0 for c in text]
#
# # def cleaned_text_to_sequence(text):
# #     seq = [symbols.index(c) if c in symbols else 0 for c in text]
# #     print("입력 시퀀스 길이:", len(seq), "내용:", seq)
# #     return seq
#
# # def cleaned_text_to_sequence(text):
# #     # 자모 분리
# #     jamo_text = hgtk.text.decompose(text)
# #     print("자모 분리된 텍스트:", jamo_text)
# #     seq = [symbols.index(c) if c in symbols else 0 for c in jamo_text]
# #     return seq
# def cleaned_text_to_sequence(text):
#     jamo_text = unicodedata.normalize("NFD", text)
#     print("자모 분리된 텍스트:", jamo_text)
#     seq = [symbols.index(c) if c in symbols else 0 for c in jamo_text]
#     return seq
#
# # TTS 생성 함수
# def generate_tts(text, speaker_id, output_path):
#     print("output_path 체크 :: ", output_path)
#     config_path = "C:/Users/LT2019002/Downloads/MeloTTS_Korean_Metadata_Web/custom_configs/korean_cpu.json"
#     ckpt_path = "C:/Users/LT2019002/Downloads/MeloTTS_Korean_Metadata_Web/logs/KR-default/checkpoints/G_50.pth"
#
#
#     with open(config_path, "r", encoding="utf-8") as f:
#         cfg = json.load(f)
#     model_cfg = cfg["model"]
#     data_cfg = cfg["data"]
#     segment_size = cfg["train"]["segment_size"]
#
#     # 입력 변환
#     seq = cleaned_text_to_sequence(text)
#     x = torch.LongTensor(seq)[None, :].to("cpu")
#     x_lengths = torch.LongTensor([x.shape[1]])
#
#     # 모델 생성
#     net_g = SynthesizerTrn(
#         n_vocab=n_symbols,
#         spec_channels=data_cfg["filter_length"] // 2 + 1,
#         segment_size=segment_size,
#         n_speakers=data_cfg["n_speakers"],
#         inter_channels=model_cfg["inter_channels"],
#         hidden_channels=model_cfg["hidden_channels"],
#         filter_channels=model_cfg["filter_channels"],
#         n_heads=model_cfg["n_heads"],
#         n_layers=model_cfg["n_layers"],
#         kernel_size=model_cfg["kernel_size"],
#         p_dropout=model_cfg["p_dropout"],
#         resblock=model_cfg["resblock"],
#         resblock_kernel_sizes=model_cfg["resblock_kernel_sizes"],
#         resblock_dilation_sizes=model_cfg["resblock_dilation_sizes"],
#         upsample_rates=model_cfg["upsample_rates"],
#         upsample_initial_channel=model_cfg["upsample_initial_channel"],
#         upsample_kernel_sizes=model_cfg["upsample_kernel_sizes"],
#         use_spectral_norm=model_cfg.get("use_spectral_norm", False),
#         n_layers_trans_flow=model_cfg["n_layers_trans_flow"]
#     ).to("cpu")
#     net_g.eval()
#
#     # 체크포인트 로드 (✅ 정확한 key 사용)
#     checkpoint = torch.load(ckpt_path, map_location="cpu")
#     net_g.load_state_dict(checkpoint["net_g"], strict=True)
#
#     # 추론
#     with torch.no_grad():
#         sid = torch.LongTensor([speaker_id]).to("cpu")
#         tone = torch.LongTensor([[0]]).to("cpu")
#         language = torch.LongTensor([[4]]).to("cpu")  # KR
#         # bert = torch.zeros((1, 256, x.shape[1])).to("cpu")
#         # ja_bert = torch.zeros((1, 256, x.shape[1])).to("cpu")
#         bert = torch.zeros((1, 1024, x.shape[1])).to("cpu")
#         ja_bert = torch.zeros((1, 768, x.shape[1])).to("cpu")
#
#         audio = net_g.infer(x, x_lengths, sid, tone, language, bert, ja_bert)[0][0, 0].cpu().numpy()
#
#     # 저장
#
#
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     volume_boost = 5.0  # 3~5배 정도 증폭
#     audio = np.clip(audio * volume_boost, -1.0, 1.0)
#     # audio = (audio * 32768).astype(np.int16)
#     # audio = (audio * 32768).astype(np.int16)
#     # 클리핑 방지 + 볼륨 조절 (최대 음량 맞추기)
#     audio = (audio / np.max(np.abs(audio)) * 32767).astype(np.int16)
#     scipy.io.wavfile.write(output_path, data_cfg["sampling_rate"], audio)
#
#     print("샘플레이트:", data_cfg["sampling_rate"])  # 22050?
#     print("audio dtype:", audio.dtype)
#     print("audio max abs:", np.max(np.abs(audio)))
#     print("생성된 오디오 샘플 수:", len(audio))  # 보통 44100 = 1초 기준
#
#     print("입력 텍스트:", text)
#     print("시퀀스 길이:", len(seq))
#     print("시퀀스 내용:", seq)
#     print("오디오 샘플 수:", len(audio))
#     print("오디오 최대값:", np.max(np.abs(audio)))
#
#     # 웹 인터페이스
# HTML_TEMPLATE = """
# <!doctype html>
# <title>TTS Infer</title>
# <h2>Text-to-Speech</h2>
# <form method=post>
#   텍스트: <input type=text name=text><br><br>
#   성별:
#   <select name=speaker_id>
#     <option value=0>남성</option>
#     <option value=1>여성</option>
#   </select><br><br>
#   <input type=submit value=생성하기>
# </form>
# """
#
# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "POST":
#         text = request.form["text"]
#         speaker_id = int(request.form["speaker_id"])
#         output_path = "static/audio.wav"
#         generate_tts(text, speaker_id, output_path)
#         return f"<h3>음성 생성 완료!</h3><audio controls src='/static/audio.wav'></audio>"
#     return render_template_string(HTML_TEMPLATE)
#
# if __name__ == "__main__":
#     app.run(debug=True)
