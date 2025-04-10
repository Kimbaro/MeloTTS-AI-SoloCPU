import os
import unicodedata

input_txt = "train.txt"
output_txt = "train_metadata.txt"

wav_root = ".//mnt/c/Users/LT2019002/Downloads/MeloTTS-main/MeloTTS-AI-SoloCPU/melo/cpu_based/dev/data/kss/1"
language = "KR"
speaker = "female"

# 자모 분리 함수
def split_korean(text):
    result = []
    for char in text:
        if '가' <= char <= '힣':
            decomposed = unicodedata.normalize('NFD', char)
            result.append(' '.join([j for j in decomposed if unicodedata.category(j) != 'Mn']))
        elif char.strip():
            result.append(char)
    return ' '.join(result)

# 변환 실행
with open(input_txt, 'r', encoding='utf-8') as fin, open(output_txt, 'w', encoding='utf-8') as fout:
    for line in fin:
        wav_path, text = line.strip().split("|")
        basename = os.path.basename(wav_path).replace(".wav", "")
        jamos = split_korean(text)
        tone_and_word2ph = "1 " * len(jamos.split())
        tone_and_word2ph = tone_and_word2ph.strip()
        fout.write(f"{wav_path}|{basename}|{speaker}|{language}|{text}|{jamos}|{tone_and_word2ph}|{tone_and_word2ph}\n")

print(f"✅ train_metadata.txt 생성 완료")
