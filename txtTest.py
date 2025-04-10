import os

# 원본 txt 경로
input_path = "train.txt"
# 변환 후 저장할 파일 경로
output_path = "train_converted.txt"

with open(input_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

converted = []
for line in lines:
    line = line.strip()
    if not line or "|" not in line:
        continue

    wav_path, text = line.split("|", 1)

    # ID는 wav 파일명에서 추출 (예: 1_0000.wav → 1_0000)
    file_name = os.path.basename(wav_path)
    utt_id = os.path.splitext(file_name)[0]

    # 필드 구성
    speaker = "female"
    language = "KR"
    phones = text  # 음소 추출 안 되어 있으면 텍스트로 대체
    tone = "flat"
    word2ph = " ".join(["1"] * len(text))  # 글자 수만큼 1 넣음

    converted_line = f"{utt_id}|{speaker}|{language}|{text}|{phones}|{tone}|{word2ph}"
    converted.append(converted_line)

# 결과 저장
with open(output_path, "w", encoding="utf-8") as f:
    for line in converted:
        f.write(line + "\n")

print(f"✅ 변환 완료: {output_path}")
