import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from trainer.evaluate import evaluate


def plot_training_curves(train_losses, val_losses, val_accs, save_dir):#画图并保存训练损失列表，验证损失列表，验证准确率列表，图片保存目录
    os.makedirs(save_dir, exist_ok=True)#创建保存目录
    epochs = list(range(1, len(train_losses) + 1))#横坐标，从1到30

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))#图文位置和尺寸

    # Loss curve
    axes[0].plot(epochs, train_losses, label="Train Loss", marker="o")
    axes[0].plot(epochs, val_losses, label="Val Loss", marker="o")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy curve
    axes[1].plot(epochs, val_accs, label="Val Accuracy", marker="o")
    axes[1].set_title("Validation Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()#自动调整布局，放重叠
    save_path = os.path.join(save_dir, "training_curves.png")#构造保存路径
    plt.savefig(save_path, dpi=300)#根据路径和分辨率保存图像
    plt.close()

    return save_path


def save_checkpoint(state, path):#保存训练断点
    os.makedirs(os.path.dirname(path), exist_ok=True)#确认保存路径
    torch.save(state, path)#保存模型参数，优化器状态和历史记录


def load_checkpoint(path, model, optimizer=None, scheduler=None):#加载最后的训练伦次模型参数，且会恢复学习率调度器和adma优化器的状态
    ckpt = torch.load(path, map_location="cpu")#加载断点训练内容
    model.load_state_dict(ckpt["model_state_dict"])#恢复模型参数，学习权重等

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])#恢复adam优化器状态

    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])#恢复学习率调度器状态

    start_epoch = ckpt.get("epoch", 0) + 1#从断点开始迭代
    best_f1 = ckpt.get("best_f1", 0.0)#恢复历史最好的f1

    history = ckpt.get("history", {
        "train_losses": [],
        "val_losses": [],
        "val_accs": []
    })#恢复训练前的曲线历史，以免图像出现断点

    return start_epoch, best_f1, history

##初始化参数，创建checkpoint列表，回复记录，每轮训练结束更新学习率
def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs,
    checkpoint_dir,
    logger,
    mixed_precision=True,
    grad_accum_steps=1,#梯度累计步数
    early_stop_patience=5,
    resume=True
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    latest_ckpt_path = os.path.join(checkpoint_dir, "latest.pt")#最新训练状态保存路径（断点）
    best_ckpt_path = os.path.join(checkpoint_dir, "best.pt")#（最佳模型保存路径）

    criterion = torch.nn.CrossEntropyLoss()#创建损失函数

    amp_enabled = (device == "cuda" and mixed_precision)#使用混合精度训练
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)#混合精度训练的梯度缩放器，稳定训练

    start_epoch = 0
    best_f1 = 0.0
    no_improve_epochs = 0#早停计数器

    train_losses = []
    val_losses = []
    val_accs = []

    # Resume training 恢复最后一轮训练的各个参数，接着记录日志
    if resume and os.path.exists(latest_ckpt_path):#有checkpoint文件时
        logger.info(f"Resuming from checkpoint: {latest_ckpt_path}")#恢复日志
        start_epoch, best_f1, history = load_checkpoint(
            latest_ckpt_path, model, optimizer, scheduler
        )#加载checkpoint，模型参数，优化器状态等
        train_losses = history.get("train_losses", [])#恢复训练损失历史
        val_losses = history.get("val_losses", [])
        val_accs = history.get("val_accs", [])
        logger.info(f"Resume start_epoch={start_epoch}, best_f1={best_f1:.4f}")#更新日志

    for epoch in range(start_epoch, epochs):
        model.train()#模型训练开始
        optimizer.zero_grad()#清空优化器梯度，adam优化器每次更新参数都需要删除旧的梯度，用新的梯度进行更新

        running_loss = 0.0#初始化本轮损失（损失值是每一轮训练根据每一个样本计算损失取平均所得）
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for step, batch in enumerate(pbar):#每批次16个样本，按批次读取所有的样本
            # 兼容两种 dataset 返回形式：
            # 1) tuple/list: seq, img, label
            # 2) dict: {"seq":..., "img":..., "label":...}
            if isinstance(batch, (list, tuple)):
                seq, img, label = batch
            elif isinstance(batch, dict):
                seq = batch["seq"]
                img = batch["img"]
                label = batch["label"]
            else:
                raise TypeError(f"Unsupported batch type: {type(batch)}")#按数据类型类型取出seq，img和label

            seq = seq.to(device, non_blocking=True)#序列输入到设备
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled):#在混合精度条件下执行以下操作
                outputs = model(seq, img)#数据输入模型，前向传播，得到分类输出（类别预测）
                loss = criterion(outputs, label)#交叉熵计算损失
                loss = loss / grad_accum_steps#缩放损失，实际未缩放

            scaler.scale(loss).backward()#反向传播

            if (step + 1) % grad_accum_steps == 0:#每个batch更新一次参数
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * grad_accum_steps#计算真实平均损失
            current_loss = running_loss / (step + 1)#计算当前step平均loss
            pbar.set_postfix(loss=f"{current_loss:.4f}")#更新进度条显示

        # 处理最后不足 grad_accum_steps 的残余梯度，实际每个batch都更新
        if len(train_loader) % grad_accum_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()#清空梯度

        train_loss = running_loss / max(len(train_loader), 1)#计算训练平均损失（分母为总batch数）

        # Validation 验证集，每轮训练完成进行一次评估
        metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device
        )

        val_loss = metrics["loss"]
        val_acc = metrics["accuracy"]
        val_f1 = metrics["f1"]

        # Step scheduler  更新参数=原参数-梯度x学习率  学习率为0.5
        if scheduler is not None:
            scheduler.step()#学习率更新

        # Record history 保存迭代过程的损失函数和准确率并保存到日志中
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        logger.info(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_f1={val_f1:.4f}"
        )

        # Save latest checkpoint 保存训练断点，包括迭代过程的评价指标和学习率调度器，adam优化器状态
        latest_state = {
            "epoch": epoch,
            "best_f1": best_f1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "history": {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_accs": val_accs
            }
        }
        save_checkpoint(latest_state, latest_ckpt_path)

        # Save best checkpoint 保存最佳模型参数
        if val_f1 > best_f1:
            best_f1 = val_f1
            no_improve_epochs = 0

            best_state = {
                "epoch": epoch,
                "best_f1": best_f1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "history": {
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "val_accs": val_accs
                }
            }
            save_checkpoint(best_state, best_ckpt_path)
            logger.info(f"Best model updated. best_f1={best_f1:.4f}")
        else:
            no_improve_epochs += 1

        # Early stopping 早停，若连续5轮损失值和准确率不优化，则停止
        if no_improve_epochs >= early_stop_patience:
            logger.info(
                f"Early stopping triggered after {no_improve_epochs} epochs without improvement."
            )
            break

    #  画图
    plot_dir = os.path.join(checkpoint_dir, "plots")
    plot_path = plot_training_curves(train_losses, val_losses, val_accs, plot_dir)
    logger.info(f"Training curves saved to: {plot_path}")

    # 把数值直接写日志
    logger.info(f"train_losses: {train_losses}")
    logger.info(f"val_losses: {val_losses}")
    logger.info(f"val_accs: {val_accs}")