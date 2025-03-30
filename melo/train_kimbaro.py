"""
 python train.py --c test/config.json --model KR-default --pretrain_G test/training/G_0.pth --pretrain_D test/training/D_0.pth --pretrain_dur test/training/DUR.pth
"""

# flake8: noqa: E402
import argparse
import os
import traceback

import torch
from matplotlib.animation import writers
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging
from torch.optim import AdamW

from melo.models import SynthesizerTrn, DurationDiscriminator, MultiPeriodDiscriminator

logging.getLogger("numba").setLevel(logging.WARNING)
import commons
import utils
from data_utils import (
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate,
    DistributedBucketSampler,
)

from losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols
from torch.nn import DataParallel


# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = (
#     True  # If encontered training problem,please try to disable TF32.
# )
# torch.set_float32_matmul_precision("medium")
#
# torch.backends.cudnn.benchmark = True
# torch.nn.attention.sdpa_kernel("flash")
# torch.backends.cuda.enable_flash_sdp(True)
# # torch.backends.cuda.enable_mem_efficient_sdp(
# #     True
# # )  # Not available if torch version is lower than 2.0
# torch.backends.cuda.enable_math_sdp(True)
# global_step = 0


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        # print(f"set_requires_grad -> {param.requires_grad}")
        param.requires_grad = requires_grad


def run(config: str, model: str, pretrain_G: str, pretrain_D: str, pretrain_dur: str):
    hps = utils.get_hparams()

    # CPU만 사용하도록 설정
    device = torch.device('cpu')

    torch.manual_seed(hps.train.seed)

    global global_step
    collate_fn = TextAudioSpeakerCollate()
    if True:  # 단일 프로세스 모드
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))
    print(f"ML 훈련파일경로 -> {hps.data.training_files}")
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
    collate_fn = TextAudioSpeakerCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=16,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True,
        prefetch_factor=4,
    )
    # train_loader = DataLoader(
    #     train_dataset,
    #     num_workers=8,
    #     shuffle=True,
    #     pin_memory=True,
    #     batch_size=hps.train.batch_size,
    # )

    if True:  # 평가 데이터셋
        eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)
        eval_loader = DataLoader(
            eval_dataset,
            num_workers=0,
            shuffle=False,
            batch_size=1,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

    # Noise scaling 관련 코드
    if (
            "use_noise_scaled_mas" in hps.model.keys()
            and hps.model.use_noise_scaled_mas is True
    ):
        print("⚠️AI VITS2 Model을 사용합니다.")
        mas_noise_scale_initial = 0.01
        noise_scale_delta = 2e-6
    else:
        print("⚠️AI VITS1 Model을 사용합니다.")
        mas_noise_scale_initial = 0.0
        noise_scale_delta = 0.0

    # Duration Discriminator
    if (
            "use_duration_discriminator" in hps.model.keys()
            and hps.model.use_duration_discriminator is True
    ):
        print("Using duration discriminator for VITS2")
        net_dur_disc = DurationDiscriminator(
            hps.model.hidden_channels,
            hps.model.hidden_channels,
            3,
            0.1,
            gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0,
        ).to(device)  # CUDA를 사용하지 않고 CPU로.
    else:
        net_dur_disc = None

    # Generator 및 Discriminator 모델 정의 (CPU 모드로만 사용)
    print(f"️️️⚠️net_g 설정")
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        mas_noise_scale_initial=mas_noise_scale_initial,
        noise_scale_delta=noise_scale_delta,
        **hps.model,
    ).to(device)  # CUDA를 사용하지 않고 CPU로.

    print(f"️️️⚠️ net_d 설정")
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).to(device)
    set_requires_grad(net_g, True)
    set_requires_grad(net_d, True)
    print(f"️️️⚠️ optim_g 설정")
    optim_g = AdamW(
        filter(lambda p: p.requires_grad, net_g.parameters()),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    print(f"️️️⚠️ optim_d 설정")
    optim_d = AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    print(f"️️️⚠️ optim_dur_disc 설정")
    if net_dur_disc is not None:
        optim_dur_disc = AdamW(
            net_dur_disc.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )
    else:
        optim_dur_disc = None

    net_g = net_g.to(device)
    # net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
    net_d = net_d.to(device)

    # 모델 파라미터가 requires_grad=True로 설정되어 있는지 확인
    for param in net_g.parameters():
        if not param.requires_grad:
            print(f"Parameter {param} requires_grad is False, setting it to True")
            param.requires_grad = True

    for param in net_d.parameters():
        if not param.requires_grad:
            print(f"Parameter {param} requires_grad is False, setting it to True")
            param.requires_grad = True

    if net_dur_disc is not None:
        for param in net_dur_disc.parameters():
            if not param.requires_grad:
                print(f"Parameter {param} requires_grad is False, setting it to True")
                param.requires_grad = True

    # DataParallel 대신 단일 CPU에서만 실행되므로 DDP 관련 코드 제거
    net_g = DataParallel(net_g)  # 여러 CPU에서 처리할 경우를 대비해 DataParallel 사용
    net_d = DataParallel(net_d)

    # 모델 체크포인트 로딩 부분
    # pretrain_G, pretrain_D, pretrain_dur = load_pretrain_model()

    # hps.pretrain_G = hps.pretrain_G or pretrain_G
    # hps.pretrain_D = hps.pretrain_D or pretrain_D
    # hps.pretrain_dur = hps.pretrain_dur or pretrain_dur

    hps.pretrain_G = pretrain_G
    hps.pretrain_D = pretrain_D
    hps.pretrain_dur = pretrain_dur

    print(f"️️️⚠️ pretrain_G 설정")
    if hps.pretrain_G:
        utils.load_checkpoint(
            hps.pretrain_G,
            net_g,
            None,
            skip_optimizer=True
        )

    print(f"️️️⚠️ pretrain_D 설정")
    if hps.pretrain_D:
        utils.load_checkpoint(
            hps.pretrain_D,
            net_d,
            None,
            skip_optimizer=True
        )

    print(f"️️️⚠️ net_dur_disc 설정")
    if net_dur_disc is not None:
        if hps.pretrain_dur:
            utils.load_checkpoint(
                hps.pretrain_dur,
                net_dur_disc,
                None,
                skip_optimizer=True
            )

    try:
        epoch_str = 1
        global_step = (epoch_str - 1) * len(train_loader)
    except Exception as e:
        print(e)
        epoch_str = 1
        global_step = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    if net_dur_disc is not None:
        scheduler_dur_disc = torch.optim.lr_scheduler.ExponentialLR(
            optim_dur_disc, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
        )
    else:
        scheduler_dur_disc = None

    if hps.train.fp16_run:
        scaler = GradScaler('cuda', enabled=True)
    else:
        print(f"️️️⚠️내장 그래픽 사용에 따른 scaler 제거")
        scaler = None

    print(f"️️️⚠️ 학습 준비중...")
    for epoch in range(epoch_str, hps.train.epochs + 1):
        print(f"️️️⚠️ 학습 시작")
        try:
            # 학습 및 평가
            train_and_evaluate(
                0,
                epoch,
                hps,
                [net_g, net_d, net_dur_disc],
                [optim_g, optim_d, optim_dur_disc],
                [scheduler_g, scheduler_d, scheduler_dur_disc],
                scaler,
                loaders=[train_loader, eval_loader],
                logger=logger,
                writers=[writer, writer_eval],
            )
        except Exception as e:
            print(e)
            torch.cuda.empty_cache()

        # 스케줄러 업데이트
        scheduler_g.step()
        scheduler_d.step()
        if net_dur_disc is not None:
            scheduler_dur_disc.step()


def train_and_evaluate(
        rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers
):
    try:
        net_g, net_d, net_dur_disc = nets
        optim_g, optim_d, optim_dur_disc = optims
        scheduler_g, scheduler_d, scheduler_dur_disc = schedulers
        train_loader, eval_loader = loaders
        # print(f"훈련 데이터 train_loader-> [{train_loader}]")
        # print(f"훈련 데이터 eval_loader-> [{eval_loader}]")

        if writers is not None:
            writer, writer_eval = writers
        global global_step

        # 모델을 CPU로 이동
        if net_dur_disc is not None:
            net_dur_disc = net_dur_disc.to(device="cpu")

        net_g.train()
        net_d.train()
        if net_dur_disc is not None:
            net_dur_disc.train()

        for batch_idx, (
                x,
                x_lengths,
                spec,
                spec_lengths,
                y,
                y_lengths,
                speakers,
                tone,
                language,
                bert,
                ja_bert,
        ) in enumerate(tqdm(train_loader)):
            if net_g.module.use_noise_scaled_mas:
                current_mas_noise_scale = (
                        net_g.module.mas_noise_scale_initial
                        - net_g.module.noise_scale_delta * global_step
                )
                net_g.module.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)

            # 텐서를 CPU로 이동
            x, x_lengths = x.to(device="cpu"), x_lengths.to(device="cpu")
            spec, spec_lengths = spec.to(device="cpu"), spec_lengths.to(device="cpu")
            y, y_lengths = y.to(device="cpu"), y_lengths.to(device="cpu")
            speakers = speakers.to(device="cpu")
            tone = tone.to(device="cpu")
            language = language.to(device="cpu")
            bert = bert.to(device="cpu")
            ja_bert = ja_bert.to(device="cpu")
            print(f"️️️⚠️ Tensor를 CPU 모드로 변경")
            print(f"️️️⚠️훈련 시작")
            # Generator Forward
            (
                y_hat,
                l_length,
                attn,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
                (hidden_x, logw, logw_),
            ) = net_g(
                x, x_lengths, spec, spec_lengths, speakers, tone, language, bert, ja_bert
            )

            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )

            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            y = commons.slice_segments(
                y, ids_slice * hps.data.hop_length, hps.train.segment_size
            )
            print(f"️️️⚠️[GANs | START] 텐서 생성기와 구별자를 생성을 진행합니다.")
            print(f"️️️⚠️[GANs | Discriminator Loss] 텐서 데이터의 진짜와 가짜를 구분 하기 위한 학습 환경을 구성 합니다.")
            # Discriminator Loss
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
            loss_disc_all = loss_disc  # ✅ 유지

            if net_dur_disc:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(hidden_x.detach(), x_mask.detach(), logw.detach(),
                                                        logw_.detach())
                loss_dur_disc, losses_dur_disc_r, losses_dur_disc_g = discriminator_loss(y_dur_hat_r, y_dur_hat_g)
                loss_dur_disc_all = loss_dur_disc  # ✅ 유지

            print(f"️️️⚠️[GANs | Discriminator Optimization] 텐서 데이터의 진짜와 가짜를 구분 하기 위한 학습 단계를 시작합니다.")
            # Discriminator Optimization
            optim_d.zero_grad()
            if scaler:
                scaler.scale(loss_disc_all).backward()
                scaler.unscale_(optim_d)
                grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)  # ✅ 유지
                scaler.step(optim_d)
            else:
                loss_disc_all.backward()
                grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)  # ✅ 유지
                optim_d.step()

            if net_dur_disc:
                optim_dur_disc.zero_grad()
                if scaler:
                    scaler.scale(loss_dur_disc_all).backward()
                    scaler.unscale_(optim_dur_disc)
                    commons.clip_grad_value_(net_dur_disc.parameters(), None)
                    scaler.step(optim_dur_disc)
                else:
                    loss_dur_disc_all.backward()
                    commons.clip_grad_value_(net_dur_disc.parameters(), None)
                    optim_dur_disc.step()

            print(f"️️️⚠️[GANs | Generator Loss] 텐서 데이터의 가짜 데이터 생성 환경을 구성합니다.")
            # Generator Loss
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            if net_dur_disc:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(hidden_x, x_mask, logw, logw_)

            loss_dur = torch.sum(l_length.float())
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, losses_gen = generator_loss(y_d_hat_g)
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl  # ✅ 유지

            if net_dur_disc:
                loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g)
                loss_gen_all += loss_dur_gen

            print(f"️️️⚠️[GANs | Generator Loss] 텐서 데이터의 학습 내용을 정확히 이해하는지 체크합니다. (속이기 위한 시도)")
            # Generator Optimization
            optim_g.zero_grad()
            if scaler:
                scaler.scale(loss_gen_all).backward()
                scaler.unscale_(optim_g)
                grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)  # ✅ 유지
                scaler.step(optim_g)
                scaler.update()
            else:
                loss_gen_all.backward()
                grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)  # ✅ 유지
                optim_g.step()

            print(f"️️️⚠️[model_dir={hps.model_dir},rank={rank}]결과를 저장합니다.")
            if rank == 0:
                if global_step % hps.train.log_interval == 0:
                    lr = optim_g.param_groups[0]["lr"]
                    losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
                    logger.info(
                        "Train Epoch: {} [{:.0f}%]".format(
                            epoch, 100.0 * batch_idx / len(train_loader)
                        )
                    )
                    logger.info([x.item() for x in losses] + [global_step, lr])

                    scalar_dict = {
                        "loss/g/total": loss_gen_all,
                        "loss/d/total": loss_disc_all,
                        "learning_rate": lr,
                        "grad_norm_d": grad_norm_d,
                        "grad_norm_g": grad_norm_g,
                    }
                    scalar_dict.update(
                        {
                            "loss/g/fm": loss_fm,
                            "loss/g/mel": loss_mel,
                            "loss/g/dur": loss_dur,
                            "loss/g/kl": loss_kl,
                        }
                    )
                    scalar_dict.update(
                        {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
                    )
                    scalar_dict.update(
                        {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
                    )
                    scalar_dict.update(
                        {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
                    )

                    image_dict = {
                        "slice/mel_org": utils.plot_spectrogram_to_numpy(
                            y_mel[0].data.cpu().numpy()
                        ),
                        "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                            y_hat_mel[0].data.cpu().numpy()
                        ),
                        "all/mel": utils.plot_spectrogram_to_numpy(
                            mel[0].data.cpu().numpy()
                        ),
                        "all/attn": utils.plot_alignment_to_numpy(
                            attn[0, 0].data.cpu().numpy()
                        ),
                    }
                    utils.summarize(
                        writer=writer,
                        global_step=global_step,
                        images=image_dict,
                        scalars=scalar_dict,
                    )

                print(
                    f"global_step % hps.train.eval_interval == 0 --> ??? {global_step % hps.train.eval_interval == 0}")
                if global_step % hps.train.eval_interval == 0:
                    evaluate(hps, net_g, eval_loader, writer_eval)
                    utils.save_checkpoint(
                        net_g,
                        optim_g,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
                    )
                    utils.save_checkpoint(
                        net_d,
                        optim_d,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
                    )
                    if net_dur_disc is not None:
                        utils.save_checkpoint(
                            net_dur_disc,
                            optim_dur_disc,
                            hps.train.learning_rate,
                            epoch,
                            os.path.join(hps.model_dir, "DUR_{}.pth".format(global_step)),
                        )
                    keep_ckpts = getattr(hps.train, "keep_ckpts", 5)
                    if keep_ckpts > 0:
                        utils.clean_checkpoints(
                            path_to_models=hps.model_dir,
                            n_ckpts_to_keep=keep_ckpts,
                            sort_by_time=True,
                        )
                    print(f"️️️⚠️[{hps.model_dir}]체크포인트 저장 완료")
        global_step += 1
    except Exception as error:
        print("\r❌ error:", error)
        traceback.print_exc()
        exit()
    if rank == 0:
        logger.info("====> Epoch: {}".format(epoch))
    torch.cuda.empty_cache()  # CPU만 사용하므로 이 줄은 생략할 수도 있음
    print(f"️️️⚠️훈련 및 평가 종료 [학습 횟수:{global_step}]")


def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    image_dict = {}
    audio_dict = {}
    print("Evaluating ...")
    with torch.no_grad():
        for batch_idx, (
                x,
                x_lengths,
                spec,
                spec_lengths,
                y,
                y_lengths,
                speakers,
                tone,
                language,
                bert,
                ja_bert,
        ) in enumerate(eval_loader):
            # Remove CUDA-related operations
            # x, x_lengths = x.cuda(), x_lengths.cuda()
            # spec, spec_lengths = spec.cuda(), spec_lengths.cuda()
            # y, y_lengths = y.cuda(), y_lengths.cuda()
            # speakers = speakers.cuda()
            # bert = bert.cuda()
            # ja_bert = ja_bert.cuda()
            # tone = tone.cuda()
            # language = language.cuda()

            x, x_lengths = x, x_lengths
            spec, spec_lengths = spec, spec_lengths
            y, y_lengths = y, y_lengths
            speakers = speakers
            tone = tone
            language = language
            bert = bert
            ja_bert = ja_bert

            for use_sdp in [True, False]:
                y_hat, attn, mask, *_ = generator.module.infer(
                    x,
                    x_lengths,
                    speakers,
                    tone,
                    language,
                    bert,
                    ja_bert,
                    y=spec,
                    max_len=1000,
                    sdp_ratio=0.0 if not use_sdp else 1.0,
                )
                y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length

                mel = spec_to_mel_torch(
                    spec,
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1).float(),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                image_dict.update(
                    {
                        f"gen/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(
                            y_hat_mel[0].cpu().numpy()
                        )
                    }
                )
                audio_dict.update(
                    {
                        f"gen/audio_{batch_idx}_{use_sdp}": y_hat[
                                                            0, :, : y_hat_lengths[0]
                                                            ]
                    }
                )
                image_dict.update(
                    {
                        f"gt/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(
                            mel[0].cpu().numpy()
                        )
                    }
                )
                audio_dict.update({f"gt/audio_{batch_idx}": y[0, :, : y_lengths[0]]})

    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate,
    )
    generator.train()
    print('Evaluate done')
    torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Training Script")

    # 파라미터 정의
    parser.add_argument('--c', type=str, required=True, help='Path to config file')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--pretrain_G', type=str, required=True, help='Path to pretrained generator model')
    parser.add_argument('--pretrain_D', type=str, required=True, help='Path to pretrained discriminator model')
    parser.add_argument('--pretrain_dur', type=str, required=True, help='Path to pretrained duration model')

    args = parser.parse_args()

    print('parsing 완료')

    return args


if __name__ == "__main__":
    # 파라미터 파싱
    args = parse_args()

    # run 함수에 파라미터 주입
    run(args.c, args.model, args.pretrain_G, args.pretrain_D, args.pretrain_dur)

"""
python train.py --c test/config.json --model KR-default --pretrain_G test/training/G_0.pth --pretrain_D test/training/D_0.pth --pretrain_dur test/training/DUR.pth
"""
