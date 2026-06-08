
##这个文件负责把预测结果转成评价指标，准确率，混淆矩阵
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def compute_metrics(y_true, y_pred, average="macro"):#真实标签，预测标签，默认参数
    acc = accuracy_score(y_true, y_pred)#计算准确率
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )#计算精确率召回和f1
    cm = confusion_matrix(y_true, y_pred)#生成混淆矩阵
    return {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "confusion_matrix": cm
    }