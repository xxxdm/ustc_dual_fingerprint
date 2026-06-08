#训练断点，保存训练进度，以暂停训练

import os
import torch


def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler=None,
    scaler=None,
    best_metric=None,
    config=None,
    experiment_info=None
):
    """
    保存训练断点
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
        "config": config,
        "experiment_info": experiment_info
    }

    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        ckpt["scaler_state_dict"] = scaler.state_dict()

    torch.save(ckpt, path)


def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    map_location="cpu",
    strict=True
):
    """
    加载训练断点
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=map_location)

    model.load_state_dict(ckpt["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    epoch = ckpt.get("epoch", 0)
    best_metric = ckpt.get("best_metric", None)
    config = ckpt.get("config", None)
    experiment_info = ckpt.get("experiment_info", None)

    return epoch, best_metric, config, experiment_info