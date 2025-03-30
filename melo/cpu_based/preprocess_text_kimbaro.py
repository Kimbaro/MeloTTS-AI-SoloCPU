import json
from collections import defaultdict
from random import shuffle
from typing import Optional
from datetime import datetime

from tqdm import tqdm
import click
import os
import torch
from melo.text.cleaner import clean_text_bert
from melo.text.symbols import symbols, num_tones

#  python .\preprocess_text_kimbaro.py -o dev/ -i dev/ -e cpu

def execute(max_val_total: int, val_per_spk: int, n_speakers: int, num_languages: int, output_path: str, engine: str,
            input_path: str):
    input_file = os.path.join(os.path.dirname(input_path), 'metadata.list')
    output_file = os.path.join(os.path.dirname(output_path), 'config.json')
    training_file = os.path.join(os.path.dirname(output_path), 'train.list')
    validation_file = os.path.join(os.path.dirname(output_path), 'val.list')
    print(f"💡 학습환경을 재구성합니다."
          f"\n0. [학습엔진] {engine}"
          f"\n1. [경로설정] input_file -> {input_file}"
          f"\n2. [경로설정] output_file -> {output_file}"
          f"\n3. [경로설정] training_file -> {training_file}"
          f"\n4. [경로설정] validation_file -> {validation_file}")
    train_list = []
    val_list = []
    config = {
        "train": {
            "log_interval": 200,
            "eval_interval": 1000,
            "seed": 52,
            "epochs": 10000,
            "learning_rate": 0.0003,
            "betas": [0.8, 0.99],
            "eps": 1e-09,
            "batch_size": 6,
            "fp16_run": False,
            "lr_decay": 0.999875,
            "segment_size": 16384,
            "init_lr_ratio": 1,
            "warmup_epochs": 0,
            "c_mel": 45,
            "c_kl": 1.0,
            "skip_optimizer": True
        },
        "data": {
            "max_wav_value": 32768.0,
            "sampling_rate": 3000,
            "filter_length": 2048,
            "hop_length": 512,
            "win_length": 2048,
            "n_mel_channels": 128,
            "mel_fmin": 0.0,
            "mel_fmax": None,
            "add_blank": True,
            "n_speakers": n_speakers,
            "cleaned_text": True,
            "spk2id": {"KR-default": 0},
            "training_files": training_file,
            "validation_files": validation_file
        },
        "model": {
            "use_spk_conditioned_encoder": True,
            "use_noise_scaled_mas": True,
            "use_mel_posterior_encoder": False,
            "use_duration_discriminator": True,
            "inter_channels": 192,
            "hidden_channels": 192,
            "filter_channels": 768,
            "n_heads": 2,
            "n_layers": 6,
            "n_layers_trans_flow": 3,
            "kernel_size": 3,
            "p_dropout": 0.1,
            "resblock": "1",
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_rates": [8, 8, 2, 2, 2],
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": [16, 16, 8, 2, 2],
            "n_layers_q": 3,
            "use_spectral_norm": False,
            "gin_channels": 256
        },
        "symbols": ["_", "\"", "(", ")", "*", "/", ":", "AA", "E", "EE", "En", "N", "OO", "Q", "V", "[", "\\", "]", "^",
                    "a", "a:", "aa", "ae", "ah", "ai", "an", "ang", "ao", "aw", "ay", "b", "by", "c", "ch", "d", "dh",
                    "dy", "e", "e:", "eh", "ei", "en", "eng", "er", "ey", "f", "g", "gy", "h", "hh", "hy", "i", "i0",
                    "i:", "ia", "ian", "iang", "iao", "ie", "ih", "in", "ing", "iong", "ir", "iu", "iy", "j", "jh", "k",
                    "ky", "l", "m", "my", "n", "ng", "ny", "o", "o:", "ong", "ou", "ow", "oy", "p", "py", "q", "r",
                    "ry", "s", "sh", "t", "th", "ts", "ty", "u", "u:", "ua", "uai", "uan", "uang", "uh", "ui", "un",
                    "uo", "uw", "v", "van", "ve", "vn", "w", "x", "y", "z", "zh", "zy", "~", "¡", "¿", "æ", "ç", "ð",
                    "ø", "ŋ", "œ", "ɐ", "ɑ", "ɒ", "ɔ", "ɕ", "ə", "ɛ", "ɜ", "ɡ", "ɣ", "ɥ", "ɦ", "ɪ", "ɫ", "ɬ", "ɭ", "ɯ",
                    "ɲ", "ɵ", "ɸ", "ɹ", "ɾ", "ʁ", "ʃ", "ʊ", "ʌ", "ʎ", "ʏ", "ʑ", "ʒ", "ʝ", "ʲ", "ˈ", "ˌ", "ː", "̃", "̩",
                    "β", "θ", "ᄀ", "ᄁ", "ᄂ", "ᄃ", "ᄄ", "ᄅ", "ᄆ", "ᄇ", "ᄈ", "ᄉ", "ᄊ", "ᄋ", "ᄌ", "ᄍ", "ᄎ", "ᄏ", "ᄐ", "ᄑ",
                    "ᄒ", "ᅡ", "ᅢ", "ᅣ", "ᅤ", "ᅥ", "ᅦ", "ᅧ", "ᅨ", "ᅩ", "ᅪ", "ᅫ", "ᅬ", "ᅭ", "ᅮ", "ᅯ", "ᅰ", "ᅱ", "ᅲ", "ᅳ",
                    "ᅴ", "ᅵ", "ᆨ", "ᆫ", "ᆮ", "ᆯ", "ᆷ", "ᆸ", "ᆼ", "ㄸ", "!", "?", "…", ",", ".", "'", "-", "SP", "UNK"],
        "num_tones": 16,
        "num_languages": num_languages
    }
    spk_utt_map = preprocess_metadata(engine=engine, input_file=input_file)
    generate_config(output_file=output_file, config=config)
    generate_train(
        max_val_total=max_val_total,
        val_per_spk=val_per_spk,
        training_file=training_file,
        validation_file=validation_file,
        spk_utt_map=spk_utt_map)
    exit(1)


def setup_input_data() -> list[str]:
    return []


# AI가 인식하도록 메타데이터를 정제합니다.
def preprocess_metadata(engine: str, input_file: str) -> defaultdict:
    spk_utt_map = defaultdict(list)  # <- train.list, val.list 생성에 필요합니다.

    if not os.path.exists(input_file):
        print(f"⚠️[{input_file}] 이 경로상에 존재하지 않음, 학습 불가하여 종료합니다")
        exit(1)
    else:
        print(f"🔄 [{input_file}] parsing start!=========================>")
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        preprocess_convet_data = input_file + "." + timestamp + ".preprocess"
        print(f"🔄 preprocess metadata start: {preprocess_convet_data}")
        out_file = open(preprocess_convet_data, "w", encoding="utf-8")
        for line in tqdm(open(input_file, encoding="utf-8").readlines()):
            try:
                # 메타데이터 형식에 따라 파싱부는 수정이 필요한 부분 입니다.
                # print(f"-> {line}")
                utt, spk, language, text, text2, text3, version, en_text = line.strip().split("|")
                if engine == 'cpu':
                    # CPU 자원 사용
                    norm_text, phones, tones, word2ph, bert = clean_text_bert(text, language, device='cpu')
                elif engine == 'cuda':
                    # GPU 자원 사용
                    norm_text, phones, tones, word2ph, bert = clean_text_bert(text, language, device='cuda')
                else:
                    print(f"❌ 존재하지 않는 엔진: {engine}")
                    exit(1)

                assert len(phones) == len(tones)
                assert len(phones) == sum(word2ph)

                cleanText = "{}|{}|{}|{}|{}|{}|{}\n".format(
                    utt, spk, language, norm_text,
                    " ".join(phones),
                    " ".join([str(i) for i in tones]),
                    " ".join([str(i) for i in word2ph]),
                );

                spk_utt_map[spk].append(cleanText)
                out_file.write(
                    cleanText
                )
                print("\r✔️ success:", cleanText)
            except Exception as error:
                print("\r❌ error:", line, error)
        print(f"🔄 [{input_file}] parsing complete!=========================>")
        return spk_utt_map


def generate_config(output_file: str, config) -> list[str]:
    # 매 실행마다 기존 config.json 백업 및 새로 생성
    train_list = []
    print(f"🔄 config.json init start: {output_file}")
    if os.path.exists(output_file):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        backup_path = f"{output_file}.{timestamp}"
        os.rename(output_file, backup_path)
        print(f"✅ config.json 백업성공: {backup_path}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"✅ config.json 생성완료: {output_file}")
    return []


# AI가 학습하기 위한 트레이닝 정보를 생성합니다.
def generate_train(max_val_total: int, val_per_spk: int, training_file: str, validation_file: str,
                   spk_utt_map: defaultdict):
    train_list = []
    val_list = []
    print(f"✅ train.list 재구성시작: {training_file}")

    for spk, utts in spk_utt_map.items():
        shuffle(utts)
        val_list += utts[:val_per_spk]
        train_list += utts[val_per_spk:]

    if len(val_list) > max_val_total:
        train_list += val_list[max_val_total:]
        val_list = val_list[:max_val_total]

    print(f"✅ train_list 재구성시작: {train_list}")
    print(f"✅ val_list 재구성시작:   {val_list}")

    with open(training_file, "w", encoding="utf-8") as f:
        f.writelines(train_list)
    with open(validation_file, "w", encoding="utf-8") as f:
        f.writelines(val_list)

    # if not os.path.exists(
    # training_file):
    #     with open(training
    #     _file, "w") as f:
    #         f.writelines()

    # spk_utt_map = defaultdict(list)  # map 생성
    # train_list = []
    # training_file = os.path.join(os.path.dirname(output_path), 'train.list')
    #
    # with open(training_file, encoding="utf-8") as f:
    #     for line in f.readlines():
    #         utt, spk, language, text, phones, tones, word2ph = line.strip().split("|")
    #         spk_utt_map[spk].append(line)
    #
    # train_list = []
    # val_list = []
    #
    # for spk, utts in spk_utt_map.items():
    #     shuffle(utts)
    #     val_list += utts[:val_per_spk]
    #     train_list += utts[val_per_spk:]
    #
    # if len(val_list) > max_val_total:
    #     train_list += val_list[max_val_total:]
    #     val_list = val_list[:max_val_total]


# AI가 학습하기 위한 트레이닝 정보를 생성합니다.
def generate_validation(n_speakers: int, num_languages: int, output_path: str, val_per_spk: int, max_val_total: int):
    print(f"🔄 validation_file init start: {training_file}")
    val_list = []
    # print('output_file -> ', output_file)
    # print('training_file -> ', training_file)
    # print('validation_file -> ', validation_file)


@click.command()
@click.option("--input-path", "-i", default="metadata.list", help="Input metadata file path")
@click.option("--output-path", "-o", default="config.json", help="Output config file path")
@click.option("--engine", "-e", default="cpu", help="추론학습 엔진을 선택합니다. cpu, cuda")
@click.option("--n-speakers", default=256, help="Number of speakers")
@click.option("--num-languages", default=10, help="Number of languages")
@click.option("--val-per-spk", default=4)
@click.option("--max-val-total", default=8)
def main(n_speakers: int,
         num_languages: int,
         input_path: str,
         output_path: str,
         engine: str,
         val_per_spk: int,
         max_val_total: int):
    execute(max_val_total=max_val_total, val_per_spk=val_per_spk, n_speakers=n_speakers, num_languages=num_languages,
            output_path=output_path,
            input_path=input_path,
            engine=engine)
    # generate_config(n_speakers=n_speakers, num_languages=num_languages, output_path=output_path,
    #                 val_per_spk=val_per_spk, max_val_total=max_val_total)

    print(f"✅ Config file saved to {output_path}")


if __name__ == "__main__":
    main()
