import torch
import torch.nn as nn


class CNNBranch(nn.Module):
    """
    CNN分支：只负责提取图像特征图
    输入:
        x: [B, 1, 28, 28]
    输出:
        feat_map: [B, 64, 7, 7]
    """
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), #第一层卷积，输入1图，输出16图，即16个卷积核，卷积核大小为3x3，填充1圈
            nn.BatchNorm2d(16),#16个输出分别归一化
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 最大池化 28 -> 14

            nn.Conv2d(16, 32, kernel_size=3, padding=1),#第二层卷积，上一层16个输入，本层32个卷积核32个输出
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 14 -> 7

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        feat_map = self.features(x)   # [B, 64, 7, 7]  最终是32x64
        return feat_map