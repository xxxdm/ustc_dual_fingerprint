import os
import re
import csv
import math
from typing import List, Dict, Optional

import matplotlib.pyplot as plt



# 1. 基础配置

LOG_DIR = "logs"
SAVE_DIR = "compare_plots"
os.makedirs(SAVE_DIR, exist_ok=True)

LOG_FILES = {
    "seq_only": os.path.join(LOG_DIR, "seq_only.log"),
    "img_only": os.path.join(LOG_DIR, "img_only.log"),
    "dual_concat": os.path.join(LOG_DIR, "dual_concat.log"),
    "seq_only_nope": os.path.join(LOG_DIR, "seq_only_nope.log"),
}

MODEL_DISPLAY_NAMES = {
    "seq_only": "Seq-Only",
    "img_only": "Img-Only",
    "dual_concat": "Dual-Channel",
    "seq_only_nope": "Seq-Only-NoPE",
}



# 2. 日志解析函数

def safe_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def safe_int(x: str) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def is_valid_number(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def parse_log_file(log_path: str) -> List[Dict]:
    """
    尽量兼容多种常见训练日志格式。
    目标提取：
        epoch
        train_loss
        val_loss
        val_accuracy

    返回：
        [
            {
                "epoch": 1,
                "train_loss": 0.1234,
                "val_loss": 0.2345,
                "val_accuracy": 0.9876
            },
            ...
        ]
    """

    if not os.path.exists(log_path):
        print(f"[WARN] Log file not found: {log_path}")
        return []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # 常见 epoch 匹配
    epoch_patterns = [
        re.compile(r"Epoch\s*[:\[]?\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE),
        re.compile(r"Epoch\s*[:\[]?\s*(\d+)", re.IGNORECASE),
        re.compile(r"\[Epoch\s+(\d+)\]", re.IGNORECASE),
        re.compile(r"epoch\s*=\s*(\d+)", re.IGNORECASE),
    ]

    # train loss 常见写法
    train_loss_patterns = [
        re.compile(r"train[_\s-]*loss\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
        re.compile(r"loss\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
    ]

    # val loss 常见写法
    val_loss_patterns = [
        re.compile(r"val[_\s-]*loss\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
        re.compile(r"valid[_\s-]*loss\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
    ]

    # val accuracy 常见写法
    val_acc_patterns = [
        re.compile(r"val[_\s-]*acc(?:uracy)?\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
        re.compile(r"valid[_\s-]*acc(?:uracy)?\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
        re.compile(r"accuracy\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
        re.compile(r"acc\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
    ]

    records = []
    current = {
        "epoch": None,
        "train_loss": None,
        "val_loss": None,
        "val_accuracy": None
    }

    def flush_current():
        nonlocal current, records
        if current["epoch"] is not None:
            # 防止重复 epoch 被多次 append，后面去重
            records.append(current.copy())
        current = {
            "epoch": None,
            "train_loss": None,
            "val_loss": None,
            "val_accuracy": None
        }

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # 1) 识别 epoch
        epoch_found = None
        for p in epoch_patterns:
            m = p.search(line_stripped)
            if m:
                epoch_found = safe_int(m.group(1))
                break

        # 如果新遇到 epoch，并且 current 已经有内容，则先保存旧记录
        if epoch_found is not None:
            if current["epoch"] is not None and current["epoch"] != epoch_found:
                flush_current()
            current["epoch"] = epoch_found

        # 2) train loss
        for p in train_loss_patterns:
            m = p.search(line_stripped)
            if m:
                val = safe_float(m.group(1))
                if val is not None:
                    # 避免 val_loss 被 "loss" 覆盖
                    if "val" not in line_stripped.lower() and "valid" not in line_stripped.lower():
                        current["train_loss"] = val
                        break

        # 3) val loss
        for p in val_loss_patterns:
            m = p.search(line_stripped)
            if m:
                val = safe_float(m.group(1))
                if val is not None:
                    current["val_loss"] = val
                    break

        # 4) val accuracy
        for p in val_acc_patterns:
            m = p.search(line_stripped)
            if m:
                val = safe_float(m.group(1))
                if val is not None:
                    low = line_stripped.lower()
                    # 尽量只抓验证准确率
                    if "val" in low or "valid" in low:
                        current["val_accuracy"] = val
                        break

    # 最后一条补上
    if current["epoch"] is not None:
        flush_current()

    # 去重：同一个 epoch 可能被多次记录，保留信息最完整的一条 =====
    epoch_map = {}
    for r in records:
        ep = r["epoch"]
        if ep is None:
            continue

        score = sum([
            1 if is_valid_number(r["train_loss"]) else 0,
            1 if is_valid_number(r["val_loss"]) else 0,
            1 if is_valid_number(r["val_accuracy"]) else 0
        ])

        if ep not in epoch_map:
            epoch_map[ep] = (score, r)
        else:
            old_score = epoch_map[ep][0]
            if score >= old_score:
                # 用信息更完整的替换
                old_r = epoch_map[ep][1]
                merged = old_r.copy()
                for k in ["train_loss", "val_loss", "val_accuracy"]:
                    if is_valid_number(r[k]):
                        merged[k] = r[k]
                epoch_map[ep] = (score, merged)

    final_records = []
    for ep in sorted(epoch_map.keys()):
        final_records.append(epoch_map[ep][1])

    return final_records



# 3. 保存解析结果 CSV

def save_records_to_csv(records: List[Dict], save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy"])
        for r in records:
            writer.writerow([
                r.get("epoch"),
                r.get("train_loss"),
                r.get("val_loss"),
                r.get("val_accuracy"),
            ])



# 4. 绘图函数

def plot_multi_model_curve(
    model_records: Dict[str, List[Dict]],
    metric_key: str,
    model_keys: List[str],
    title: str,
    ylabel: str,
    save_path: str
):
    plt.figure(figsize=(10, 6))

    for model_key in model_keys:
        records = model_records.get(model_key, [])
        xs, ys = [], []
        for r in records:
            val = r.get(metric_key)
            if r.get("epoch") is not None and is_valid_number(val):
                xs.append(r["epoch"])
                ys.append(val)

        if len(xs) == 0:
            print(f"[WARN] No valid data for {model_key} - {metric_key}")
            continue

        plt.plot(xs, ys, marker="o", linewidth=2, label=MODEL_DISPLAY_NAMES.get(model_key, model_key))

    plt.title(title, fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved figure: {save_path}")



# 5. 主流程

def main():
    # 解析所有日志
    model_records = {}
    for model_key, log_path in LOG_FILES.items():
        records = parse_log_file(log_path)
        model_records[model_key] = records

        csv_path = os.path.join(SAVE_DIR, f"{model_key}_parsed.csv")
        save_records_to_csv(records, csv_path)

        print(f"[INFO] {model_key}: parsed {len(records)} epochs from {log_path}")
        if len(records) > 0:
            print(f"       first record: {records[0]}")
            print(f"       last  record: {records[-1]}")

   
    # 对比实验：seq_only / img_only / dual_concat
  
    compare_models = ["seq_only", "img_only", "dual_concat"]

    plot_multi_model_curve(
        model_records=model_records,
        metric_key="val_loss",
        model_keys=compare_models,
        title="Validation Loss Comparison of Different Models",
        ylabel="Validation Loss",
        save_path=os.path.join(SAVE_DIR, "compare_val_loss_seq_img_dual.png")
    )

    plot_multi_model_curve(
        model_records=model_records,
        metric_key="val_accuracy",
        model_keys=compare_models,
        title="Validation Accuracy Comparison of Different Models",
        ylabel="Validation Accuracy",
        save_path=os.path.join(SAVE_DIR, "compare_val_acc_seq_img_dual.png")
    )

    # 可选：训练损失对比
    plot_multi_model_curve(
        model_records=model_records,
        metric_key="train_loss",
        model_keys=compare_models,
        title="Train Loss Comparison of Different Models",
        ylabel="Train Loss",
        save_path=os.path.join(SAVE_DIR, "compare_train_loss_seq_img_dual.png")
    )

  
    # 消融实验：seq_only / seq_only_nope
 
    ablation_models = ["seq_only", "seq_only_nope"]

    plot_multi_model_curve(
        model_records=model_records,
        metric_key="val_loss",
        model_keys=ablation_models,
        title="Validation Loss Comparison of Ablation Models",
        ylabel="Validation Loss",
        save_path=os.path.join(SAVE_DIR, "ablation_val_loss_seq_vs_nope.png")
    )

    plot_multi_model_curve(
        model_records=model_records,
        metric_key="val_accuracy",
        model_keys=ablation_models,
        title="Validation Accuracy Comparison of Ablation Models",
        ylabel="Validation Accuracy",
        save_path=os.path.join(SAVE_DIR, "ablation_val_acc_seq_vs_nope.png")
    )

    # 可选：训练损失对比
    plot_multi_model_curve(
        model_records=model_records,
        metric_key="train_loss",
        model_keys=ablation_models,
        title="Train Loss Comparison of Ablation Models",
        ylabel="Train Loss",
        save_path=os.path.join(SAVE_DIR, "ablation_train_loss_seq_vs_nope.png")
    )

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()