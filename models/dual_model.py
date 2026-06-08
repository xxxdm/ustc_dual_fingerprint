import torch
import torch.nn as nn

from models.attention_branch import AttentionBranch
from models.cnn_branch import CNNBranch


class DualChannelNet(nn.Module):
    """
    双分支模型：
    - seq_branch 输出序列特征图 [B, L, D]
    - img_branch 输出图像特征图 [B, C, H, W]
    - 在主模型中统一完成池化、投影、融合、分类
    """

    def __init__(
        self,
        num_classes=20,
        vocab_size=256,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        ff_dim=128,#隐藏层层数
        dropout=0.1,
        cnn_out_dim=128,#cnn输出投影维度
        fusion_dim=128,#dual的隐藏层
        mode="dual",                # dual / seq_only / img_only
        fusion_type="concat",       # concat / add
        use_pos_encoding=True
    ):
        super().__init__()

        self.mode = mode
        self.fusion_type = fusion_type
        self.embed_dim = embed_dim
        self.seq_out_dim = 128#序列输出维度
        self.img_feat_channels = 64#特征图通道数

        # 1) 分支：只做特征提取
        self.seq_branch = AttentionBranch(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            use_pos_encoding=use_pos_encoding
        )#输出[n，784，64]

        self.img_branch = CNNBranch()#[n，64，7，7]

    
        # [B, L, D] -> [B, D, L] -> pool -> [B, D] -> proj -> [B, 128]
        self.seq_pool = nn.AdaptiveAvgPool1d(1)#平均池化，假如有10个样本，每个样本有784个元素序列，每个元素映射成64维的向量，则先对矩阵转置成[10,64,784]，再平均池化成[10，64，1]=[10，64]
        self.seq_proj =nn.Sequential(
            nn.Linear(embed_dim, self.seq_out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )#最终输出seq[n，128]

        # 3) 图像分支：特征图 -> 向量
        # [B, 64, 7, 7] -> pool -> [B, 64] -> proj -> [B, cnn_out_dim]
        self.img_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.img_proj = nn.Sequential(
            nn.Linear(self.img_feat_channels, cnn_out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )#最终输出[n，128]

        # 4) 分类器输入维度
        if mode == "dual":
            if fusion_type == "concat":
                classifier_in_dim = self.seq_out_dim + cnn_out_dim#拼接attention和cnn后的维度，分类器的输入为序列拼接图像[N,128]
            elif fusion_type == "add":
                if cnn_out_dim != self.seq_out_dim:
                    raise ValueError(
                        f"For add fusion, cnn_out_dim must be {self.seq_out_dim}."
                    )
                classifier_in_dim = self.seq_out_dim
            else:
                raise ValueError("fusion_type must be 'concat' or 'add'")

        elif mode == "seq_only":
            classifier_in_dim = self.seq_out_dim

        elif mode == "img_only":
            classifier_in_dim = cnn_out_dim

        else:
            raise ValueError("mode must be 'dual', 'seq_only', or 'img_only'")

        # 5) 分类器
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, fusion_dim),#全连接层1，特征精炼
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes)#全连接层2，分类
        )

    def forward(self, seq, img):
        """
        seq: [B, L]
        img: [B, 1, 28, 28]
        """

        #序列分支 
        if self.mode in ["dual", "seq_only"]:
            seq_feat_map = self.seq_branch(seq)               # [B, L, D]
            seq_feat = seq_feat_map.transpose(1, 2)          # [B, D, L]
            seq_feat = self.seq_pool(seq_feat).squeeze(-1)   # [B, D]
            seq_feat = self.seq_proj(seq_feat)               # [B, 128]

        #图像分支 
        if self.mode in ["dual", "img_only"]:
            img_feat_map = self.img_branch(img)              # [B, 64, 7, 7]
            img_feat = self.img_pool(img_feat_map).flatten(1) # [B, 64]
            img_feat = self.img_proj(img_feat)               # [B, cnn_out_dim]

        #模式控制
        if self.mode == "seq_only":
            feat = seq_feat

        elif self.mode == "img_only":
            feat = img_feat

        else:  # dual
            if self.fusion_type == "concat":
                feat = torch.cat([seq_feat, img_feat], dim=1)
            elif self.fusion_type == "add":
                feat = seq_feat + img_feat
            else:
                raise ValueError("fusion_type must be 'concat' or 'add'")

        #  分类 
        out = self.classifier(feat)
        return out