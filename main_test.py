# 训练完之后，加载最佳模型，在独立测试集上输出最终结果

import os
import yaml
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.dataset import USTCDualDataset
from models.dual_model import DualChannelNet
from trainer.evaluate import evaluate
from trainer.checkpoint import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file"
    )
    return parser.parse_args()


def get_num_classes(cfg):
    mode = cfg["task"]["mode"]
    if mode == "binary":
        return 2
    elif mode == "10class":
        return 10
    elif mode == "20class":
        return 20
    else:
        raise ValueError("Unknown task mode")


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print(f"Using config: {args.config}")

    device_str = cfg["train"]["device"]
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"

    device = torch.device(device_str)

    num_classes = get_num_classes(cfg)

    # experiment 配置实验参数
    exp_cfg = cfg.get("experiment", {})
    exp_mode = exp_cfg.get("mode", "dual")
    fusion_type = exp_cfg.get("fusion_type", "concat")
    use_pos_encoding = exp_cfg.get("use_pos_encoding", True)

    test_path = os.path.join(cfg["data"]["split_dir"], cfg["data"]["test_file"])
    test_set = USTCDualDataset(test_path)
    test_loader = DataLoader(
        test_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"]
    )

    model = DualChannelNet(
        num_classes=num_classes,
        vocab_size=cfg["model"]["vocab_size"],
        embed_dim=cfg["model"]["embed_dim"],
        num_heads=cfg["model"]["num_heads"],
        num_layers=cfg["model"]["num_layers"],
        ff_dim=cfg["model"]["ff_dim"],
        dropout=cfg["model"]["dropout"],
        cnn_out_dim=cfg["model"]["cnn_out_dim"],
        fusion_dim=cfg["model"]["fusion_dim"],
        mode=exp_mode,
        fusion_type=fusion_type,
        use_pos_encoding=use_pos_encoding
    ).to(device)

    ckpt_path = os.path.join(cfg["checkpoint"]["dir"], cfg["checkpoint"]["best"])
    print(f"Loading checkpoint: {ckpt_path}")
    print(f"Experiment mode: {exp_mode}")
    print(f"Fusion type: {fusion_type}")
    print(f"Use positional encoding: {use_pos_encoding}")

    load_checkpoint(
        ckpt_path,
        model,
        map_location=device
    )

    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, test_loader, criterion, device)

    print("Test Results:")
    print(f"loss:      {metrics['loss']:.4f}")
    print(f"accuracy:  {metrics['accuracy']:.4f}")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall:    {metrics['recall']:.4f}")
    print(f"f1:        {metrics['f1']:.4f}")
    print("confusion_matrix:")
    print(metrics["confusion_matrix"])


if __name__ == "__main__":
    main()