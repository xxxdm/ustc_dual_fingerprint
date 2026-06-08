import os
import yaml
import numpy as np
import matplotlib.pyplot as plt


def load_config(config_path="configs/config_dual_concat.yaml"):
    """
    读取配置文件
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_weibo_label(cfg):
    """
    从 dataset_map.normal_10 中获取 Weibo 的标签编号
    """
    normal_10 = cfg["dataset_map"]["normal_10"]
    if "Weibo" not in normal_10:
        raise ValueError("配置文件中未找到 Weibo 标签")
    return normal_10["Weibo"]


def seq_to_gray_image(seq, image_size=28):
    """
    将长度为 784 的字节序列转换为 28x28 灰度图
    """
    arr = np.array(seq, dtype=np.float32)

    expected_len = image_size * image_size
    if len(arr) < expected_len:
        raise ValueError(f"序列长度不足，期望 {expected_len}，实际 {len(arr)}")

    arr = arr[:expected_len]
    img = arr.reshape(image_size, image_size)

    # 归一化到 [0, 1]
    if img.max() > 1.0:
        img = img / 255.0

    return img


def save_single_gray_image(img, save_path):
    """
    保存单张灰度图
    """
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def main():
    # 1. 读取配置
    cfg = load_config("configs/config_dual_concat.yaml")

    # 2. 数据路径
    npz_path = os.path.join(cfg["data"]["processed_dir"], cfg["data"]["all_file"])
    # 如果你想从测试集读取，可以改成：
    # npz_path = os.path.join(cfg["data"]["split_dir"], cfg["data"]["test_file"])

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"数据文件不存在: {npz_path}")

    # 3. 读取数据
    data = np.load(npz_path, allow_pickle=True)
    seqs = data["seq"]        # [N, 784]
    labels = data["label"]    # [N]

    # 4. 获取 Weibo 标签
    weibo_label = get_weibo_label(cfg)

    # 5. 找到 Weibo 样本
    weibo_indices = np.where(labels == weibo_label)[0]

    if len(weibo_indices) < 3:
        raise ValueError(f"Weibo 样本不足 3 个，当前仅找到 {len(weibo_indices)} 个")

    selected_indices = weibo_indices[:3]

    # 6. 创建输出目录
    save_dir = "visualizations/weibo_gray_images"
    os.makedirs(save_dir, exist_ok=True)

    # 7. 保存三张独立灰度图
    for i, idx in enumerate(selected_indices, start=1):
        seq = seqs[idx]
        img = seq_to_gray_image(seq, image_size=cfg["data"]["image_size"])

        save_path = os.path.join(save_dir, f"weibo_packet_{i}.png")
        save_single_gray_image(img, save_path)

        print(f"✅ 已保存第 {i} 张图: {save_path}")

    print("\n🎉 完成：3 个 Weibo 流量包灰度图已分别生成。")


if __name__ == "__main__":
    main()