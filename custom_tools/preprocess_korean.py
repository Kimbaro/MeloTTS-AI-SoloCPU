import os

wav_dir = os.path.expanduser("~/datasets/korean/wavs")
transcript_path = os.path.expanduser("~/datasets/korean/transcript.v.1.2.txt")
output_file = "filelists/train.txt"

with open(transcript_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

os.makedirs("filelists", exist_ok=True)
with open(output_file, "w", encoding="utf-8") as out:
    for i, line in enumerate(lines[:50]):
        if "|" in line:
            name, text = line.strip().split("|", 1)
            wav_path = os.path.join(wav_dir, name)
            if os.path.exists(wav_path):
                out.write(f"{wav_path}|{text}\n")
print("[+] 전처리 완료:", output_file)