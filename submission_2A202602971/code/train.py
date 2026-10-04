"""train.py — Đặt seed, đánh giá, vòng huấn luyện run_experiment(cfg, data), dự đoán và xuất file nộp.

Mọi thí nghiệm chỉ là đổi dict cfg rồi gọi lại run_experiment.
"""
from __future__ import annotations

import sys
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Bảo đảm in được tiếng Việt/Unicode trên terminal Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from data import iterate_batches, prepare_data
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, build_scheduler, clip_gradients
from plots import plot_run
from results_table import save_result

# Cấu hình mặc định = BASELINE (M-base).
DEFAULT_CFG = dict(
    exp_id="base-s1",
    group="baseline",
    description="Baseline M-base (SGD+Momentum, lr=0.03)",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=0.03,                   # Giá trị chuẩn ổn định cho SGD momentum 0.9
    weight_decay=0.0,
    momentum=0.9,
    batch=512,
    epochs=20,
    hidden=(256, 128),
    dropout=0.0,
    init="he",
    clip_norm=None,            # None = không clip; hoặc số thực, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cố định cho random, numpy, torch (và cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    num_classes = cm.shape[0]
    f1_list = []
    for c in range(num_classes):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - tp)
        fn = float(cm[c, :].sum() - tp)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * p * r) / (p + r) if (p + r) > 0 else 0.0
        f1_list.append(f1)
    return float(np.mean(f1_list))


def compute_loss(logits: torch.Tensor, y: torch.Tensor, loss_name: str) -> torch.Tensor:
    """Tính hàm mất mát:
       "ce"  : Cross-Entropy nhận logits thô và nhãn int64.
       "mse" : Mean Squared Error giữa logits và one-hot của y.
    """
    loss_mode = loss_name.lower()
    if loss_mode == "ce":
        return F.cross_entropy(logits, y)
    elif loss_mode == "mse":
        y_one_hot = F.one_hot(y, num_classes=logits.shape[-1]).to(dtype=logits.dtype)
        return F.mse_loss(logits, y_one_hot)
    else:
        raise ValueError(f"Hàm loss không hỗ trợ: '{loss_name}'. Hỗ trợ: 'ce', 'mse'")


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits."""
    model.eval()
    preds_list = []
    N = len(X)
    for i in range(0, N, batch_size):
        xb = X[i:i + batch_size]
        logits = model(xb)
        preds_list.append(torch.argmax(logits, dim=-1))
    return torch.cat(preds_list, dim=0)


@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1, cm) ở chế độ eval() và no_grad."""
    model.eval()
    total_loss = 0.0
    N = len(X)
    all_preds = []
    num_classes = 7

    for i in range(0, N, batch_size):
        xb = X[i:i + batch_size]
        yb = y[i:i + batch_size]
        logits = model(xb)

        if loss_name.lower() == "ce":
            loss = F.cross_entropy(logits, yb, reduction="sum")
        else:
            y_oh = F.one_hot(yb, num_classes=logits.shape[-1]).to(dtype=logits.dtype)
            loss = F.mse_loss(logits, y_oh, reduction="sum")

        total_loss += loss.item()
        all_preds.append(torch.argmax(logits, dim=-1))

    preds = torch.cat(all_preds, dim=0)
    acc = float((preds == y).float().mean().item())

    # Xây dựng ma trận nhầm lẫn 7x7
    y_cpu = y.cpu().numpy()
    preds_cpu = preds.cpu().numpy()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_cpu, preds_cpu):
        cm[t, p] += 1

    macro_f1 = macro_f1_from_confusion(cm)

    # Với MSE, F.mse_loss chia tổng số phần tử (N * 7)
    divisor = N * (num_classes if loss_name.lower() == "mse" else 1)
    avg_loss = total_loss / divisor

    return {
        "loss": float(avg_loss),
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        "cm": cm,
    }


def run_experiment(cfg: dict, data: dict, verbose: bool = True) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt kết quả."""
    seed = cfg.get("seed", 1)
    set_seed(seed)

    device = data["X_tr"].device
    is_cuda = device.type == "cuda"

    # 1. Khởi tạo Model
    hidden = cfg.get("hidden", (256, 128))
    dropout = cfg.get("dropout", 0.0)
    init_mode = cfg.get("init", "he")
    model = MLP(hidden=hidden, dropout=dropout, init=init_mode).to(device)

    # 2. Khởi tạo Optimizer & Scheduler
    lr = float(cfg.get("lr", 0.03))
    opt_name = cfg.get("optimizer", "sgd_momentum")
    weight_decay = float(cfg.get("weight_decay", 0.0))
    momentum = float(cfg.get("momentum", 0.9))
    optimizer = build_optimizer(opt_name, model.parameters(), lr=lr,
                                weight_decay=weight_decay, momentum=momentum)

    epochs = int(cfg.get("epochs", 20))
    batch_size = int(cfg.get("batch", 512))
    scheduler = build_scheduler(optimizer, cfg.get("scheduler"), total_steps=epochs)

    # 3. Precision (FP32, FP16, BF16)
    precision = cfg.get("precision", "fp32").lower()
    scaler = torch.amp.GradScaler("cuda") if (precision == "fp16" and is_cuda) else None

    # 4. Đo Loss bước 0 trên tập Validation (trước bất kỳ bước cập nhật nào)
    step0_val = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"])
    step0_loss = step0_val["loss"]

    if verbose:
        print(f"[{cfg.get('exp_id', 'exp')}] Khởi tạo xong. Loss bước 0 trên val = {step0_loss:.4f}")

    # Tập con train cố định 50,000 mẫu để đo train_loss ở chế độ eval()
    eval_sub_size = min(50000, len(data["X_tr"]))
    X_tr_eval = data["X_tr"][:eval_sub_size]
    y_tr_eval = data["y_tr"][:eval_sub_size]

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "grad_norm": [],
        "epoch_time_s": [],
    }

    best_val_loss = float("inf")
    best_epoch = 1
    best_state = None
    best_val_acc = 0.0
    best_val_macro_f1 = 0.0
    diverged = False

    generator = torch.Generator(device=device).manual_seed(seed)

    for epoch in range(1, epochs + 1):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        model.train()
        grad_norms = []

        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], batch_size=batch_size,
                                      generator=generator, shuffle=True):
            optimizer.zero_grad(set_to_none=True)

            # Forward với Precision tương ứng
            if precision == "fp16" and is_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(xb)
                    loss = compute_loss(logits, yb, cfg["loss"])
            elif precision == "bf16":
                dev_type = "cuda" if is_cuda else "cpu"
                with torch.autocast(device_type=dev_type, dtype=torch.bfloat16):
                    logits = model(xb)
                    loss = compute_loss(logits, yb, cfg["loss"])
            else:
                logits = model(xb)
                loss = compute_loss(logits, yb, cfg["loss"])

            # Kiểm tra NaN/inf
            if torch.isnan(loss) or torch.isinf(loss):
                diverged = True
                print(f"CẢNH BÁO: Loss bị phân kỳ (NaN/inf) tại epoch {epoch}!")
                break

            # Backward & Step
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gn = clip_gradients(model.parameters(), cfg.get("clip_norm"))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                gn = clip_gradients(model.parameters(), cfg.get("clip_norm"))
                optimizer.step()

            grad_norms.append(gn)

        if diverged:
            break

        if scheduler is not None:
            scheduler.step()

        if is_cuda:
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - t0

        # Đánh giá cuối epoch ở chế độ eval()
        tr_eval = evaluate(model, X_tr_eval, y_tr_eval, loss_name=cfg["loss"])
        val_eval = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"])
        avg_gn = float(np.mean(grad_norms)) if grad_norms else 0.0

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_eval["loss"])
        history["val_loss"].append(val_eval["loss"])
        history["val_acc"].append(val_eval["acc"])
        history["val_macro_f1"].append(val_eval["macro_f1"])
        history["grad_norm"].append(avg_gn)
        history["epoch_time_s"].append(epoch_time)

        # Lưu best epoch theo val_loss
        if val_eval["loss"] < best_val_loss:
            best_val_loss = val_eval["loss"]
            best_epoch = epoch
            best_val_acc = val_eval["acc"]
            best_val_macro_f1 = val_eval["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose:
            print(f"Epoch {epoch:2d}/{epochs:2d} | "
                  f"Train Loss: {tr_eval['loss']:.4f} | "
                  f"Val Loss: {val_eval['loss']:.4f} | "
                  f"Val Acc: {val_eval['acc']:.4f} | "
                  f"Val F1: {val_eval['macro_f1']:.4f} | "
                  f"GradNorm: {avg_gn:.3f} | {epoch_time:.2f}s")

    peak_mem_MB = float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if is_cuda else 0.0
    avg_time_per_epoch = float(np.mean(history["epoch_time_s"])) if history["epoch_time_s"] else 0.0

    summary = {
        "step0_loss": float(step0_loss),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "final_train_loss": float(history["train_loss"][-1]) if history["train_loss"] else float("nan"),
        "final_val_loss": float(history["val_loss"][-1]) if history["val_loss"] else float("nan"),
        "val_acc": float(best_val_acc),
        "val_macro_f1": float(best_val_macro_f1),
        "time_per_epoch_s": round(avg_time_per_epoch, 3),
        "peak_mem_MB": round(peak_mem_MB, 2),
        "diverged": bool(diverged),
    }

    result = {
        "cfg": cfg,
        "history": history,
        "summary": summary,
        "best_state": best_state if best_state is not None else model.state_dict(),
    }
    return result


def write_predictions(row_id: np.ndarray, preds: np.ndarray, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`."""
    out_file = Path(path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("row_id,pred\n")
        for rid, pred in zip(row_id, preds):
            f.write(f"{rid},{int(pred)}\n")


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi CSV."""
    device = data["X_eval"].device
    hidden = cfg.get("hidden", (256, 128))
    model = MLP(hidden=hidden, dropout=0.0, init=cfg.get("init", "he")).to(device)
    model.load_state_dict(result["best_state"])
    preds = predict(model, data["X_eval"])
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
    print(f"[✓] Đã lưu file dự đoán eval tại: {pred_path}")


def test_baseline(quick: bool = True):
    """Hàm nghiệm thu Part 2: kiểm tra pipeline huấn luyện, lưu log JSON và ảnh PNG.

    Nếu quick=True: chạy 2 epoch để kiểm tra pipeline trong vài giây.
    Nếu quick=False: chạy đủ 20 epoch.
    """
    print("=" * 60)
    print("BẮT ĐẦU KIỂM TRA NGHIỆM THU PART 2 (Pipeline & Baseline)")
    print("=" * 60)

    # 1. Nạp dữ liệu
    data = prepare_data(device="cpu")

    # 2. Cấu hình kiểm tra
    epochs = 2 if quick else 20
    test_cfg = dict(
        exp_id="test-base-part2" if quick else "base-s1",
        group="baseline",
        description=f"Baseline M-base ({epochs} epochs)",
        loss="ce",
        optimizer="sgd_momentum",
        lr=0.05,
        weight_decay=0.0,
        momentum=0.9,
        batch=512,
        epochs=epochs,
        hidden=(256, 128),
        dropout=0.0,
        init="he",
        clip_norm=None,
        precision="fp32",
        seed=1,
    )

    # 3. Chạy pipeline
    res = run_experiment(test_cfg, data, verbose=True)

    # 4. Kiểm tra kết quả
    summary = res["summary"]
    hist = res["history"]

    assert len(hist["epoch"]) == epochs, f"Số epoch ghi lại sai: {len(hist['epoch'])}"
    assert summary["step0_loss"] > 0, "Loss bước 0 không hợp lệ"
    assert summary["val_acc"] > 0.40, f"Accuracy quá thấp: {summary['val_acc']}"
    print(f"[✓] 1. Pipeline huấn luyện chạy thành công:")
    print(f"    - Loss bước 0: {summary['step0_loss']:.4f}")
    print(f"    - Best Val Loss: {summary['best_val_loss']:.4f} (Epoch {summary['best_epoch']})")
    print(f"    - Val Acc: {summary['val_acc']:.4%}, Val Macro-F1: {summary['val_macro_f1']:.4f}")
    print(f"    - Thời gian trung bình/epoch: {summary['time_per_epoch_s']:.2f}s")

    # 5. Lưu kết quả JSON và vẽ biểu đồ
    base_dir = Path(__file__).resolve().parent.parent  # submission_<MSSV>/
    res_dir = base_dir / "results"
    fig_dir = base_dir / "figures"
    json_path = save_result(res, str(res_dir))
    fig_path = str(fig_dir / f"{test_cfg['exp_id']}.png")
    plot_run(res, fig_path)

    assert Path(json_path).exists(), f"Không tìm thấy file JSON: {json_path}"
    assert Path(fig_path).exists(), f"Không tìm thấy file ảnh biểu đồ: {fig_path}"

    print(f"[✓] 2. Đã lưu log JSON tại: {json_path}")
    print(f"[✓] 3. Đã vẽ biểu đồ 3 ô tại: {fig_path}")

    print("=" * 60)
    print(">>> KẾT QUẢ: PART 2 ĐÃ HOÀN THÀNH VÀ ĐẠT TẤT CẢ TIÊU CHUẨN NGHIỆM THU! <<<")
    print("=" * 60)


def run_baseline_seeds(seeds=(1, 2, 3), epochs: int = 20, lr: float = 0.05):
    """Chạy baseline M-base với nhiều seed khác nhau để đo độ lệch chuẩn và độ nhiễu seed."""
    print("=" * 60)
    print(f"CHẠY BASELINE VỚI CÁC SEED: {seeds} ({epochs} epochs, lr={lr})")
    print("=" * 60)
    data = prepare_data(device="cpu")
    base_dir = Path(__file__).resolve().parent.parent
    res_dir = base_dir / "results"
    fig_dir = base_dir / "figures"

    results = []
    for s in seeds:
        cfg = dict(
            exp_id=f"base-s{s}",
            group="baseline",
            description=f"Baseline M-base (Seed {s})",
            loss="ce",
            optimizer="sgd_momentum",
            lr=lr,
            weight_decay=0.0,
            momentum=0.9,
            batch=512,
            epochs=epochs,
            hidden=(256, 128),
            dropout=0.0,
            init="he",
            clip_norm=None,
            precision="fp32",
            seed=s,
        )
        print(f"\n>>> Đang chạy {cfg['exp_id']} (Seed {s})...")
        res = run_experiment(cfg, data, verbose=True)
        save_result(res, str(res_dir))
        plot_run(res, str(fig_dir / f"{cfg['exp_id']}.png"))
        results.append(res)

    f1_scores = [r["summary"]["val_macro_f1"] for r in results]
    acc_scores = [r["summary"]["val_acc"] for r in results]
    mean_f1, std_f1 = float(np.mean(f1_scores)), float(np.std(f1_scores))
    mean_acc, std_acc = float(np.mean(acc_scores)), float(np.std(acc_scores))

    print("\n" + "=" * 60)
    print(f"KẾT QUẢ BASELINE QUA {len(seeds)} SEEDS:")
    for s, acc, f1 in zip(seeds, acc_scores, f1_scores):
        print(f"  - Seed {s}: Val Acc = {acc:.4%}, Val Macro-F1 = {f1:.4f}")
    print(f"  => Val Macro-F1: {mean_f1:.4f} ± {std_f1:.4f} (Ngưỡng nhiễu 2σ = {2*std_f1:.4f})")
    print(f"  => Val Accuracy: {mean_acc:.4%} ± {std_acc:.4%}")
    print("=" * 60)
    return results
