import torch
ckpt = torch.load("test/training/G_0.pth", map_location="cpu")
print(ckpt.keys())  # 정상적으로 로드되는지 확인
