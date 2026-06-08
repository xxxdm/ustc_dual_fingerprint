##按比例切分数据集把已经构建好的全量样本集合，按照设定比例切分成训练集0.7、验证集0.15和测试集0.15

import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split#数据拆分两部分，规定seed，按seed打乱，按标签分层抽样

#save_subset()函数作用是把某一组索引对应的数据保存成一个 .npz 文件。
def save_subset(split_dir, name, seq, img, label, group_id, indices):#name是子集名，indices是样本引索
    out_path = os.path.join(split_dir, f"{name}.npz")#拼接输出的子集npz保存路径
    np.savez_compressed(
        out_path,
        seq=seq[indices],
        img=img[indices],
        label=label[indices],
        group_id=group_id[indices]
    )#压缩保存
    print(f"{name}: {len(indices)} -> {out_path}")
  
  #split_by_group()函数用于按标签分层随机切分
  ##思路：先读取切分所需参数，创建输出目录，构造样本的索引下标再根据下标进行拆分成训练集，验证集，测试机，最后保存并打印标签分布
def split_by_group():
    with open("configs/config_seq_only.yaml", "r", encoding="utf-8") as f:#打开配置文件
        cfg = yaml.safe_load(f)

    processed_dir = cfg["data"]["processed_dir"]
    split_dir = cfg["data"]["split_dir"]#子集npz保存路径
    all_file = cfg["data"]["all_file"]

    train_ratio = cfg["data"]["train_ratio"]#训练集比例
    val_ratio = cfg["data"]["val_ratio"]
    test_ratio = cfg["data"]["test_ratio"]
    seed = cfg["seed"]#随机种子

    os.makedirs(split_dir, exist_ok=True)#创建输出保存目录

    path = os.path.join(processed_dir, all_file)#构造全量样本路径
    data = np.load(path, allow_pickle=True)#根据构造的路径寻找文件，加载全两样本

    seq = data["seq"]#分别取出样本的各个信息
    img = data["img"]
    label = data["label"]
    group_id = data["group_id"]

    idx = np.arange(len(label))#生成样本引索，引索长度为总样本数

    # 第一阶段：先拆成train 与 temp
    train_idx, temp_idx = train_test_split(
        idx,#全量样本索引
        train_size=train_ratio,#训练集拆分比例
        random_state=seed,#随机打乱的seed
        shuffle=True,#随机打乱
        stratify=label#分层抽样
    )#先得到引索，把引索顺序打乱，根据每个引索的每个样本的类别映射值按比例0.7：0.3拆分，

    # 第二阶段：temp 再拆分成 val 和 test
    temp_label = label[temp_idx]
    val_part = val_ratio / (val_ratio + test_ratio)

    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_part,
        random_state=seed,
        shuffle=True,
        stratify=temp_label
    )

    save_subset(split_dir, "train", seq, img, label, group_id, train_idx)
    save_subset(split_dir, "val", seq, img, label, group_id, val_idx)
    save_subset(split_dir, "test", seq, img, label, group_id, test_idx)

    print("train label distribution:", np.unique(label[train_idx], return_counts=True))
    print("val label distribution:", np.unique(label[val_idx], return_counts=True))
    print("test label distribution:", np.unique(label[test_idx], return_counts=True))