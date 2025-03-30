import os
import click
from datetime import datetime
from melo.api import TTS

'''

python train.py --c data/kss/config.json --model KR-default --pretrain_G logs/KR-default/G_0.pth --pretrain_D logs/KR-default/D.pth --pretrain_dur logs/KR-default/DUR.pth

tip : 학습한 데이터를 참조하여 TTS를 생성해봅니다.

파일명	                역할	                                                사용 여부
G_*.pth	Generator       (음성을 생성하는 핵심 모델)	                            ✅ 사용해야 함
D_*.pth	Discriminator   (훈련 중에만 사용됨, 평가에는 필요 없음)	                ❌ 필요 없음
DUR_*.pth	Duration    (VITS2에서 음성 길이를 조정할 때 사용됨, 선택적)	        ⭕ 옵션


tip : 사용된 데이터는 G_0.pth 이므로 TTS 생성 시 동일한 모델을 써야합니다.
python inferTypeKimbaro.py -t '안녕하세요!. 티티에스 품질 100% 테스트 중입니다. 중요한 요소라고 생각되는 음성, 목소리, 어조를 유심히 살펴봐주시기 바랍니다.' -m checkpoint -o ../test -l 'KR'

'''

@click.command()
@click.option('--ckpt_path', '-m', type=str, required=True, help="Path to the checkpoint file")
@click.option('--text', '-t', type=str, required=True, help="Text to speak")
@click.option('--language', '-l', type=str, default="KR", help="Language of the model")
@click.option('--output_dir', '-o', type=str, default="outputs", help="Path to save the generated audio")
def main(ckpt_path, text, language, output_dir):
    """TTS 모델을 사용하여 음성을 생성하는 스크립트"""

    print(f"✅ 체크포인트 로드: {ckpt_path}")
    print(f"📢 변환할 텍스트: {text}")

    # 설정 파일 경로 (모델과 같은 폴더에 있는 config.json 사용)
    config_path = os.path.join(os.path.dirname('data/kss/'), 'config.json')

    # TTS 모델 로드
    try:
        model = TTS(language=language, config_path=config_path, ckpt_path=ckpt_path)
        print("✅ 모델 로드 성공!")
    except Exception as e:
        raise RuntimeError(f"❌ 모델 로드 실패: {e}")

    # 현재 시간 기반으로 고유한 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 화자별 음성 생성
    for spk_name, spk_id in model.hps.data.spk2id.items():
        save_path = os.path.join(output_dir, spk_name, f"output_{timestamp}.wav")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        print(f"🎙️ 화자 '{spk_name}' 음성 생성 중...")
        try:
            model.tts_to_file(text, spk_id, save_path)
            print(f"✅ 음성 저장 완료: {save_path}")
        except Exception as e:
            print(f"❌ 음성 변환 실패 (화자: {spk_name}): {e}")


if __name__ == "__main__":
    main()
