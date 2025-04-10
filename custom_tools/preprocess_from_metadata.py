import os

metadata_path = os.path.expanduser("~/datasets/korean/metadata.list")
output_file = "filelists/train.txt"

os.makedirs("filelists", exist_ok=True)

with open(metadata_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

with open(output_file, "w", encoding="utf-8") as out:
    for line in lines[:50]:
        parts = line.strip().split("|")
        if len(parts) >= 4:
            wav_path = parts[0].strip().replace("\\", "/").replace("C:/", "/mnt/c/")
            text = parts[3].strip()
            out.write(f"{wav_path}|{text}\n")

print(f"[+] metadata.list 기반 train.txt 생성 완료 → {output_file}")