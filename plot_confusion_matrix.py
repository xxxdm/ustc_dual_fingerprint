import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(
    cm,
    class_names,
    save_path="checkpoints/plots/confusion_matrix.png",
    normalize=False,
    figsize=(14, 12),
    cmap="Blues",
    title=None
):
    """
    绘制混淆矩阵热力图

    参数:
        cm: np.ndarray, shape = [num_classes, num_classes]
        class_names: list[str], 类别名称列表
        save_path: 保存路径
        normalize: 是否按行归一化
        figsize: 图像大小
        cmap: 颜色映射
        title: 图标题
    """
    cm = np.array(cm, dtype=np.float64)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_display = cm / row_sums
        fmt = ".2f"
        if title is None:
            title = "Normalized Confusion Matrix"
    else:
        cm_display = cm
        fmt = ".0f"
        if title is None:
            title = "Confusion Matrix"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor="gray"
    )

    plt.title(title, fontsize=16)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()

    print(f"[INFO] Saved confusion matrix figure to: {save_path}")


if __name__ == "__main__":
    # ===== 20类类别名，可按需要修改顺序，但必须和标签编号一致 =====
    class_names = [
        "BitTorrent", "Facetime", "FTP", "Gmail", "MySQL",
        "Outlook", "Skype", "SMB", "Weibo", "WorldOfWarcraft",
        "Cridex", "Geodo", "Htbot", "Miuref", "Neris",
        "Nsis-ay", "Shifu", "Tinba", "Virut", "Zeus"
    ]

    # ===== 把测试输出的 confusion_matrix 直接粘到这里 =====
    cm = np.array([
    [639, 0, 1, 22, 1, 82, 0, 0, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 750, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 740, 1, 2, 1, 2, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [30, 0, 1, 644, 0, 71, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 1, 0, 749, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [35, 0, 3, 15, 0, 685, 1, 2, 6, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0],
 [1, 0, 0, 1, 0, 0, 748, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 1, 0, 2, 1496, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 1, 0, 0, 1, 6, 2991, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 750, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 730, 0, 1, 19, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 748, 1, 0, 0, 0, 0, 0, 0, 0],
 [1, 0, 1, 0, 0, 1, 1, 0, 3, 0, 0, 0, 732, 0, 3, 8, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 743, 0, 0, 0, 2, 0, 0],
 [0, 0, 1, 1, 0, 2, 0, 0, 0, 0, 0, 0, 3, 0, 641, 50, 0, 0, 52, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 25, 712, 0, 0, 7, 0],
 [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 16, 0, 0, 1, 731, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 12, 0, 0, 0, 736, 0, 0],
 [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 2, 1, 92, 32, 0, 0, 619, 1],
 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 4, 0, 1, 0, 0, 0, 0, 743]
])
    # 1) 原始数量版
    plot_confusion_matrix(
        cm=cm,
        class_names=class_names,
        save_path="checkpoints/seq_only_nope/plots/confusion_matrix_count.png",
        normalize=False,
        title="Confusion Matrix (Count)"
    )

    # 2) 归一化版
    plot_confusion_matrix(
        cm=cm,
        class_names=class_names,
        save_path="checkpoints/seq_only_nope/plots/confusion_matrix_normalized.png",
        normalize=True,
        title="Confusion Matrix (Normalized)"
    )