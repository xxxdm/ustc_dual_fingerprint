# 读配置、构造 DataLoader、建优化器、建学习率调度器

import os
import yaml
import torch
import argparse
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from utils.seed import set_seed
from utils.logger import get_logger
from utils.dataset import USTCDualDataset
from models.dual_model import DualChannelNet
from trainer.train import train_model


def parse_args():#解析命令行参数，因为有多个config配置文件，未配好配置文件前不能直接点击右上角编译
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",#这个default已经修改，无法使用
        help="Path to config file"
    )
    return parser.parse_args()#返回读取命令


def get_num_classes(cfg):#确定分类任务
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

    # 1.读取指定配置文件
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2.固定随机种子
    set_seed(cfg["seed"])
    torch.set_num_threads(cfg["train"]["cpu_threads"])

    # 3.创建日志
    logger = get_logger(cfg["log"]["dir"], cfg["log"]["file"])#创建特定的log目录
    logger.info(f"Using config: {args.config}")#记录当前使用配置文件

    # 4.设备设置
    device_str = cfg["train"]["device"]

    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("配置要求使用 GPU(cuda)，但当前环境检测不到可用 CUDA。")
        logger.info("Using device: cuda")
        logger.info(f"GPU name: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    else:
        device_str = "cpu"
        logger.info("Using device: cpu")

    device = torch.device(device_str)

    # 5.类别数
    num_classes = get_num_classes(cfg)

    # 6. 实验配置（带默认值）
    exp_cfg = cfg.get("experiment", {})
    exp_mode = exp_cfg.get("mode", "dual")
    fusion_type = exp_cfg.get("fusion_type", "concat")
    use_pos_encoding = exp_cfg.get("use_pos_encoding", True)

    # 7.拼接数据路径
    train_path = os.path.join(cfg["data"]["split_dir"], cfg["data"]["train_file"])
    val_path = os.path.join(cfg["data"]["split_dir"], cfg["data"]["val_file"])

    # 8.数据集划分
    train_set = USTCDualDataset(
        train_path,
        max_samples=cfg["train"]["max_train_samples"]
    )
    val_set = USTCDualDataset(
        val_path,
        max_samples=cfg["train"]["max_val_samples"]
    )

    logger.info(f"Train samples: {len(train_set)}")
    logger.info(f"Val samples: {len(val_set)}")

    # 9.DataLoader
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,#训练时二次打乱样本顺序
        num_workers=cfg["train"]["num_workers"],
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=(device.type == "cuda")
    )

    # 10. 模型
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

    logger.info(f"Model device: {next(model.parameters()).device}")
    logger.info(f"Experiment mode: {exp_mode}")
    logger.info(f"Fusion type: {fusion_type}")
    logger.info(f"Use positional encoding: {use_pos_encoding}")
    logger.info(f"Checkpoint dir: {cfg['checkpoint']['dir']}")

    # 11. 优化器和学习率调度器
    optimizer = Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"]
    )

    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)#每10个batch进行一次学习率优化

    # 12. 训练
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=cfg["train"]["epochs"],
        checkpoint_dir=cfg["checkpoint"]["dir"],
        logger=logger,
        mixed_precision=cfg["train"]["mixed_precision"],
        grad_accum_steps=cfg["train"]["grad_accum_steps"],
        early_stop_patience=cfg["train"]["early_stop_patience"],
        resume=False   
    )


if __name__ == "__main__":
    main()