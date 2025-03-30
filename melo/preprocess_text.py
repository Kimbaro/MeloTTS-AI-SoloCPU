import json
from collections import defaultdict
from random import shuffle
from typing import Optional

from tqdm import tqdm
import click
import os
import torch
from text.cleaner import clean_text_bert
from text.symbols import symbols, num_tones

'''
 python preprocess_text.py --metadata data/kss/metadata.2.sample.list
'''


@click.command()
@click.option(
    "--metadata",
    default="data/kss/metadata.1.sample.list",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--cleaned-path", default=None)
@click.option("--train-path", default=None)
@click.option("--val-path", default=None)
@click.option("--val-per-spk", default=4)
@click.option("--max-val-total", default=8)
@click.option("--clean/--no-clean", default=True)
def main(
        metadata: str,
        cleaned_path: Optional[str],
        train_path: str,
        val_path: str,
        val_per_spk: int,
        max_val_total: int,
        clean: bool,
):
    if train_path is None:
        train_path = os.path.join(os.path.dirname(metadata), 'train.list')
    if val_path is None:
        val_path = os.path.join(os.path.dirname(metadata), 'val.list')
    out_config_path = os.path.join(os.path.dirname(metadata), 'config.json')

    # ✅ 중복 없는 언어 목록 저장을 위한 set 생성
    language_set = set()

    if not os.path.exists(out_config_path):
        print(f"⚠️  {out_config_path} 파일이 존재하지 않습니다. 기본 config.json을 생성합니다.")
        empty_config = {
            "train": {},
            "data": {
                "spk2id": {
                    "KR-default": 0
                },
                "training_files": None,
                "validation_files": None
            },
            "model": {},
            "symbols": [],
            "num_tones": None,
            "num_languages": None
        }
        os.makedirs(os.path.dirname(out_config_path), exist_ok=True)
        with open(out_config_path, "w", encoding="utf-8") as f:
            json.dump(empty_config, f, indent=2, ensure_ascii=False)
        print(f"✅ 기본 config.json이 생성되었습니다: {out_config_path}")

    if cleaned_path is None:
        cleaned_path = metadata + ".cleaned"

    if clean:
        out_file = open(cleaned_path, "w", encoding="utf-8")
        for line in tqdm(open(metadata, encoding="utf-8").readlines()):
            try:
                # ✅ 데이터를 파싱하면서 language 값을 set에 추가
                utt, spk, language, text, text2, text3, version, en_text = line.strip().split("|")
                language_set.add(language)

                norm_text, phones, tones, word2ph, bert = clean_text_bert(text, language, device='cpu')   # CPU 자원 사용
                # norm_text, phones, tones, word2ph, bert = clean_text_bert(text, language, device='cuda:0') GPU CUDA 엔진사용

                assert len(phones) == len(tones)
                assert len(phones) == sum(word2ph)
                out_file.write(
                    "{}|{}|{}|{}|{}|{}|{}\n".format(
                        utt, spk, language, norm_text,
                        " ".join(phones),
                        " ".join([str(i) for i in tones]),
                        " ".join([str(i) for i in word2ph]),
                    )
                )
                bert_path = utt.replace(".wav", ".bert.pt")
                os.makedirs(os.path.dirname(bert_path), exist_ok=True)
                torch.save(bert.cpu(), bert_path)

            except Exception as error:
                print("❌ Error processing line:", line, error)

        out_file.close()
        metadata = cleaned_path

    spk_utt_map = defaultdict(list)
    spk_id_map = {}
    current_sid = 0

    with open(metadata, encoding="utf-8") as f:
        for line in f.readlines():
            utt, spk, language, text, phones, tones, word2ph = line.strip().split("|")
            spk_utt_map[spk].append(line)

            if spk not in spk_id_map.keys():
                spk_id_map[spk] = current_sid
                current_sid += 1

    train_list = []
    val_list = []

    for spk, utts in spk_utt_map.items():
        shuffle(utts)
        val_list += utts[:val_per_spk]
        train_list += utts[val_per_spk:]

    if len(val_list) > max_val_total:
        train_list += val_list[max_val_total:]
        val_list = val_list[:max_val_total]

    with open(train_path, "w", encoding="utf-8") as f:
        f.writelines(train_list)

    with open(val_path, "w", encoding="utf-8") as f:
        f.writelines(val_list)

    config = json.load(open(out_config_path, encoding="utf-8"))
    config["data"]["spk2id"] = spk_id_map
    config["data"]["training_files"] = train_path
    config["data"]["validation_files"] = val_path
    config["data"]["n_speakers"] = 256

    # ✅ num_languages 값을 고유한 언어 개수로 설정
    config["num_languages"] = 10
    config["num_tones"] = num_tones
    config["symbols"] = symbols

    with open(out_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"✅ num_languages 설정 완료: {10}")


if __name__ == "__main__":
    main()
