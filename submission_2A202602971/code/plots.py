"""plots.py — Vẽ biểu đồ cho từng thí nghiệm và biểu đồ so sánh giữa các thí nghiệm.

Mỗi thí nghiệm một ảnh: figures/<exp_id>.png gồm 3 ô (Loss, Val Metrics, Grad Norm).
Ảnh so sánh nhóm: figures/compare_<nhóm>.png.
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch
         (2) val_acc và val_macro_f1 theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    """
    cfg = result["cfg"]
    hist = result["history"]
    summary = result.get("summary", {})

    epochs = hist["epoch"]
    best_ep = summary.get("best_epoch", epochs[-1] if epochs else 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # 1. Đường cong Loss
    axes[0].plot(epochs, hist["train_loss"], label="Train Loss (eval mode)", color="#1f77b4", linewidth=2)
    axes[0].plot(epochs, hist["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2)
    axes[0].axvline(best_ep, color="green", linestyle="--", alpha=0.7, label=f"Best Ep ({best_ep})")
    axes[0].set_title("Loss (Train vs Val)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Loss", fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=9)

    # 2. Đường cong Accuracy & Macro-F1
    axes[1].plot(epochs, hist["val_acc"], label="Val Accuracy", color="#2ca02c", linewidth=2)
    if "val_macro_f1" in hist and len(hist["val_macro_f1"]) > 0:
        axes[1].plot(epochs, hist["val_macro_f1"], label="Val Macro-F1", color="#d62728", linewidth=2)
    axes[1].axvline(best_ep, color="green", linestyle="--", alpha=0.7, label=f"Best Ep ({best_ep})")
    axes[1].set_title("Validation Metrics", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("Score (0..1)", fontsize=10)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=9)

    # 3. Chuẩn Gradient (L2) trước khi cắt
    axes[2].plot(epochs, hist["grad_norm"], label="Avg Grad Norm (pre-clip)", color="#9467bd", linewidth=2)
    axes[2].set_title("Gradient Norm (Pre-clip)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Epoch", fontsize=10)
    axes[2].set_ylabel("L2 Norm", fontsize=10)
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend(fontsize=9)

    # Tiêu đề tổng quát với thông số chính
    exp_id = cfg.get("exp_id", "exp")
    opt = cfg.get("optimizer", "opt")
    lr = cfg.get("lr", "lr")
    batch = cfg.get("batch", "batch")
    drop = cfg.get("dropout", 0.0)
    init = cfg.get("init", "he")
    clip = cfg.get("clip_norm", "None")

    fig.suptitle(
        f"[{exp_id}] Opt: {opt} | lr: {lr} | batch: {batch} | drop: {drop} | init: {init} | clip: {clip}",
        fontsize=13, fontweight="bold", y=1.03
    )

    out_file = Path(path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(str(out_file), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ 'val_macro_f1', 'val_loss', 'grad_norm') của nhiều thí nghiệm
    trên cùng một trục để so sánh trực tiếp.
    """
    if not results:
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for res in results:
        cfg = res["cfg"]
        hist = res["history"]
        exp_id = cfg.get("exp_id", "unnamed")
        if metric in hist:
            ax.plot(hist["epoch"], hist[metric], label=exp_id, linewidth=2)

    ax.set_title(title or f"So sánh {metric}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left")

    out_file = Path(path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(str(out_file), dpi=150, bbox_inches="tight")
    plt.close(fig)
