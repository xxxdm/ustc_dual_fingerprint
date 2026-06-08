from utils.build_dataset import build_dataset
from utils.split_dataset import split_by_group

if __name__ == "__main__":
    build_dataset()  ##生成原始数据的全量样本
    split_by_group() ##对所有样本进行划分，将原始数据根据训练集验证集和测试集的方式划分