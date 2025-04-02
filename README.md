## 3️⃣ MeloTTS 기반 TTS 구축 및 추론(Pre-trained TTS) 학습 가이드

MeloTTS는 **VITS AI 기반 신경망 음성 합성 모델**을 활용하며, 추론 및 제로샷 학습이 가능합니다.
- develop 단계임을 유의바랍니다.
- MacOS 의 경우 CPU 지원이 불가합니다.

### 📌 레퍼런스(감사합니다)

- [MeloTTS GitHub](https://github.com/myshell-ai/MeloTTS)
- [Korean Single Speaker Speech Dataset](https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset)
- [OpenVoice 기반 프레임워크](https://github.com/Nyan-SouthKorea/RealTime_zeroshot_TTS_ko)
- [아카라이브 AI 음성채널](https://arca.live/b/aispeech/103703271)
- [아카라이브 AI melotts 한국어 학습 팁](https://arca.live/b/aispeech/103056950)

### ✅ 실행 환경

- **OS**: Windows
- **언어**: Python 3.9.0
- **개발 도구**: PyCharm IDE

### ✅ 실행 방법

1. pip install -r MeloTTS-AI-SoloCPU/requirements.txt
2. infertypekimbaro.py 텍스트 전처리, config 구성
3. train_kimbaro.py 추론학습 수행
4. infertypekimbaro.py TTS 생성 수행

### ✅ 강화 학습

[강화 학습 가이드](https://github.com/myshell-ai/MeloTTS/blob/main/docs/training.md)를 참고하여 진행합니다. 학습 데이터는 [Kaggle 데이터셋](https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset)을 활용했습니다.


## License

This library is under MIT License, which means it is free for both commercial and non-commercial use.

## Acknowledgements

This implementation is based on [TTS](https://github.com/coqui-ai/TTS), [VITS](https://github.com/jaywalnut310/vits), [VITS2](https://github.com/daniilrobnikov/vits2) and [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2). We appreciate their awesome work.
