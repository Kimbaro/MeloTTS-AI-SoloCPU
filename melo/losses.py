import torch


import torch

import torch

#파이토치 학습 절차

def feature_loss(fmap_r, fmap_g):
    # fmap_r과 fmap_g가 리스트라면 내부 요소 확인 후 변환
    def convert_to_tensor_list(fmap):
        if isinstance(fmap, list):
            # 내부 요소가 모두 텐서라면 torch.stack() 가능
            if all(isinstance(f, torch.Tensor) for f in fmap):
                return torch.stack(fmap)
            else:
                # 내부 요소가 리스트라면 개별적으로 텐서 변환 (단, 크기 불균형이 있으면 그대로 유지)
                return [torch.tensor(f, dtype=torch.float32) if isinstance(f, list) and all(isinstance(x, (int, float)) for x in f) else f for f in fmap]
        return fmap  # 이미 텐서면 그대로 반환

    fmap_r = convert_to_tensor_list(fmap_r)
    fmap_g = convert_to_tensor_list(fmap_g)

    print("1st fmap_r:", type(fmap_r), len(fmap_r) if isinstance(fmap_r, list) else fmap_r.shape)
    print("1st fmap_g:", type(fmap_g), len(fmap_g) if isinstance(fmap_g, list) else fmap_g.shape)

    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        # dr과 dg가 리스트라면 내부 요소 확인 후 변환
        dr = convert_to_tensor_list(dr)
        dg = convert_to_tensor_list(dg)

        print("2nd fmap_r:", type(dr), len(dr) if isinstance(dr, list) else dr.shape)
        print("2nd fmap_g:", type(dg), len(dg) if isinstance(dg, list) else dg.shape)

        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2



def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses

def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)
    return l
