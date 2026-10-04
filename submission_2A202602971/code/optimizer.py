"""optimizer.py — Bộ tối ưu, scheduler và cắt gradient (clip gradient).

Được dùng torch.optim.* và torch.nn.utils.clip_grad_norm_.
"""
from __future__ import annotations

import torch

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9, betas=(0.9, 0.999), eps: float = 1e-8):
    """Trả về một torch.optim.Optimizer.

    Hỗ trợ:
      - "sgd"          : SGD chuẩn
      - "sgd_momentum" : SGD kết hợp momentum
      - "adam"         : Adam (L2 kết hợp vào gradient)
      - "adamw"        : AdamW (weight decay tách riêng)
    """
    opt_name = name.lower()
    if opt_name not in OPTIMIZERS:
        raise ValueError(f"Bộ tối ưu không hợp lệ: '{name}'. Hỗ trợ: {OPTIMIZERS}")

    # Chuyển params thành list nếu là generator để kiểm tra
    if opt_name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif opt_name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif opt_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def build_scheduler(optimizer, name: str | None, total_steps: int, **kwargs):
    """(Tuỳ chọn) Bộ lập lịch tốc độ học, ví dụ CosineAnnealingLR."""
    if name is None:
        return None
    sched_name = name.lower()
    if sched_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, **kwargs)
    elif sched_name == "step":
        step_size = kwargs.get("step_size", max(1, total_steps // 3))
        gamma = kwargs.get("gamma", 0.5)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        raise ValueError(f"Scheduler không hỗ trợ: '{name}'")


def clip_gradients(params, max_norm: float | None) -> float:
    """Cắt gradient theo chuẩn L2 toàn cục, và TRẢ VỀ chuẩn gradient TRƯỚC KHI cắt.

    Khi max_norm là None hoặc <= 0: tính chuẩn L2 mà không cắt (max_norm = inf).
    """
    if max_norm is None or max_norm <= 0:
        total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=float("inf"))
    else:
        total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=float(max_norm))
    return float(total_norm)
