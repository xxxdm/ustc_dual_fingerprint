import os
import re
import matplotlib.pyplot as plt


def plot_from_log(log_path, save_dir="checkpoints/plots"):
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    os.makedirs(save_dir, exist_ok=True)

    train_losses = []
    val_losses = []
    val_accs = []
    epochs = []

    pattern = re.compile(
        r"Epoch\s+(\d+)/\d+\s+\|\s+train_loss=([0-9.]+)\s+\|\s+val_loss=([0-9.]+)\s+\|\s+val_acc=([0-9.]+)\s+\|\s+val_f1=([0-9.]+)"
    )

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                train_loss = float(match.group(2))
                val_loss = float(match.group(3))
                val_acc = float(match.group(4))

                epochs.append(epoch)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                val_accs.append(val_acc)

    if len(epochs) == 0:
        print("[WARN] No epoch records matched in log file.")
        return

    print("epochs:", epochs)
    print("train_losses:", train_losses)
    print("val_losses:", val_losses)
    print("val_accs:", val_accs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

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

    plt.tight_layout()

    save_path = os.path.join(save_dir, "training_curves_from_log.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"[INFO] Figure saved to: {save_path}")


if __name__ == "__main__":
    # 改成你的实际日志文件路径
    log_path = "logs/train.log"
    plot_from_log(log_path)