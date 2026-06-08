#输出验证集和测试集的评估结果

import torch
from tqdm import tqdm
from utils.metrics import compute_metrics

@torch.no_grad()#评估是关闭梯度计算，步反向传播更新参数
def evaluate(model, dataloader, criterion, device):#评估函数，模型，数据加载器，损失函数，设备
    model.eval()#进入模型评估模式，dropout不在随机丢弃
    total_loss = 0.0
    y_true = []
    y_pred = []

    for seq, img, label in tqdm(dataloader, desc="Evaluating", leave=False):#对每次迭代的三个变量进行处理
        seq = seq.to(device)#向设备传输变量
        img = img.to(device)
        label = label.to(device)

        logits = model(seq, img)#前向传播分类概率分数
        loss = criterion(logits, label)#根据分类的分数和标签计算损失哈数

        total_loss += loss.item()#损失累加
        pred = torch.argmax(logits, dim=1)#取每个样本的预测结果

        y_true.extend(label.cpu().numpy().tolist())#取所有真实标签
        y_pred.extend(pred.cpu().numpy().tolist())#取所有预测

    metrics = compute_metrics(y_true, y_pred, average="macro")#计算召回，f1，精确率，准确率
    metrics["loss"] = total_loss / max(len(dataloader), 1)#计算平均损失
    return metrics#返回评估指标