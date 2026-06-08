
##把npz文件变成pytorch张量，供模型训练器法标准化地使用数据，提供此过程训练器可得到样本的序列，图像和标签


import numpy as np
import torch
from torch.utils.data import Dataset

class USTCDualDataset(Dataset):
    def __init__(self, npz_path, max_samples=-1):#self是样本，输入是npz文件的所有样本，不限制数量
        data = np.load(npz_path, allow_pickle=True)#data做为中间变量充当内存使用，会将样本内的所有数据存入data，由于样本以字典的形式存在，故需要allow_pickle
        self.seq = data["seq"]#将npz文件内的所有序列存到seq类
        self.img = data["img"]
        self.label = data["label"]

        if max_samples is not None and max_samples > 0:
            self.seq = self.seq[:max_samples]
            self.img = self.img[:max_samples]
            self.label = self.label[:max_samples]

    def __len__(self):#返回当前npz文件的总样本数
        return len(self.label)

    def __getitem__(self, idx):#将npz文件的序列图像和标签全转换成合适的张量并返回
        seq = torch.tensor(self.seq[idx], dtype=torch.long)
        img = torch.tensor(self.img[idx], dtype=torch.float32)#灰度图要求是浮点类型
        label = torch.tensor(self.label[idx], dtype=torch.long)
        return seq, img, label