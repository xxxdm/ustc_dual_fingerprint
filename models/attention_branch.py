import torch
import torch.nn as nn

#输入序列会进入模型后会在嵌入曾将序列每一个元素处理成一个个向量，位置编码器会将每个向量位置进行编码，得到相对位置
class PositionalEncoding(nn.Module):#位置编码器，nn.module是深度学习的模块
    def __init__(self, d_model, max_len=2048):#初始化，self提前确定好正弦位置编码，初始化向量长度和最大向量数量
        super().__init__()

        pe = torch.zeros(max_len, d_model)#建立空白位置编码表，len x d_model，用于储存
        position = torch.arange(0, max_len).unsqueeze(1).float()#生成位置索引
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-torch.log(torch.tensor(10000.0)) / d_model)
        )#在不同位置生成不同频率的正弦波，使得不同位置的包的相对位置可以计算出来

        pe[:, 0::2] = torch.sin(position * div_term)#偶数位置编码取正弦
        pe[:, 1::2] = torch.cos(position * div_term)#奇数取预先
        pe = pe.unsqueeze(0)   # [1, max_len, d_model]

        self.register_buffer("pe", pe)

    def forward(self, x):#前向传播
        """
        x: [B, L, D]
        """
        return x + self.pe[:, :x.size(1)]


class AttentionBranch(nn.Module):
    """
    Transformer分支（seq_model）：
    只负责提取序列特征图

    输入:
        x:[B, L]
    输出:
        feat_map:[B, L, D]
    """
    def __init__(
        self,
        vocab_size=256,#unit8
        embed_dim=64,#嵌入曾维度
        num_heads=4,#头数
        num_layers=2,#编码层数
        ff_dim=128,#隐藏层
        dropout=0.1,
        use_pos_encoding=True
    ):
        super().__init__()

        self.use_pos_encoding = use_pos_encoding#布尔值决定是否使用pe

        self.embedding = nn.Embedding(vocab_size, embed_dim)#定义嵌入曾
        self.pos_encoding = PositionalEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,#隐藏层维度，增强模型非线性学习能力，分层提取特征
            dropout=dropout,#随机丢弃一部分神经元，抑制噪声
            batch_first=True,
            activation="gelu"
        )#定义单层编码层
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )#堆叠编码曾

    def forward(self, x):
        """
        x:[B, L]
        return: [B, L, D]
        """
        x = self.embedding(x)      # [B, L, D]
        if self.use_pos_encoding:
            x = self.pos_encoding(x)
        feat_map = self.encoder(x) # [B, L, D]
        return feat_map#输出为64维