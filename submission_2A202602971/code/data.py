"""data.py — Nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import torch

# Bảo đảm in được tiếng Việt/Unicode trên terminal Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def _resolve_processed_dir(processed_dir: str = "data/processed") -> Path:
    """Tìm đường dẫn thư mục data/processed một cách linh hoạt dù chạy từ thư mục nào."""
    p = Path(processed_dir)
    if p.exists() and (p / "train.npz").exists():
        return p

    # Thử các vị trí tương đối
    candidates = [
        Path(__file__).resolve().parents[2] / "data" / "processed",
        Path(__file__).resolve().parents[1] / "data" / "processed",
        Path("data/processed"),
        Path("../data/processed"),
        Path("../../data/processed"),
    ]
    for cand in candidates:
        if cand.exists() and (cand / "train.npz").exists():
            return cand
    raise FileNotFoundError(f"Không tìm thấy thư mục {processed_dir} chứa train.npz và eval.npz. Hãy chạy scripts/split_data.py trước.")


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    """
    p = _resolve_processed_dir(processed_dir)
    train_data = np.load(p / "train.npz")
    eval_data = np.load(p / "eval.npz")

    X_train_full = train_data["X"].astype(np.float32)
    y_train_full = train_data["y"].astype(np.int64)
    X_eval = eval_data["X"].astype(np.float32)
    y_eval = eval_data["y"].astype(np.int64)
    eval_row_id = eval_data["row_id"]

    assert X_train_full.ndim == 2 and X_train_full.shape[1] == 54, f"Shape X_train không hợp lệ: {X_train_full.shape}"
    assert y_train_full.ndim == 1 and len(y_train_full) == len(X_train_full), "y_train không khớp độ dài"
    assert X_eval.ndim == 2 and X_eval.shape[1] == 54, f"Shape X_eval không hợp lệ: {X_eval.shape}"
    assert y_eval.ndim == 1 and len(y_eval) == len(X_eval), "y_eval không khớp độ dài"
    assert len(eval_row_id) == len(X_eval), "eval_row_id không khớp độ dài"

    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, random_state=seed, stratify=y
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    """
    mean = np.mean(X_tr[:, :N_NUMERIC], axis=0).astype(np.float32)
    std = np.std(X_tr[:, :N_NUMERIC], axis=0).astype(np.float32)
    # Tránh chia cho 0 nếu có cột bất biến
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên."""
    X_norm = X.copy()
    X_norm[:, :N_NUMERIC] = (X_norm[:, :N_NUMERIC] - mean) / std
    return X_norm


def prepare_data(device: str = "cpu", val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id
    """
    X_train_full, y_train_full, X_eval_raw, y_eval_raw, eval_row_id = load_split(processed_dir)
    X_tr_raw, y_tr_raw, X_val_raw, y_val_raw = make_val_split(
        X_train_full, y_train_full, val_fraction=val_fraction, seed=seed
    )

    # Chuẩn hoá CHỈ fit trên train (tránh rò rỉ dữ liệu)
    mean, std = fit_standardizer(X_tr_raw)
    X_tr = apply_standardizer(X_tr_raw, mean, std)
    X_val = apply_standardizer(X_val_raw, mean, std)
    X_eval = apply_standardizer(X_eval_raw, mean, std)

    dev = torch.device(device)
    t_X_tr = torch.tensor(X_tr, dtype=torch.float32, device=dev)
    t_y_tr = torch.tensor(y_tr_raw, dtype=torch.int64, device=dev)
    t_X_val = torch.tensor(X_val, dtype=torch.float32, device=dev)
    t_y_val = torch.tensor(y_val_raw, dtype=torch.int64, device=dev)
    t_X_eval = torch.tensor(X_eval, dtype=torch.float32, device=dev)
    t_y_eval = torch.tensor(y_eval_raw, dtype=torch.int64, device=dev)

    val_classes, counts = np.unique(y_val_raw, return_counts=True)
    majority_class = val_classes[np.argmax(counts)]
    majority_acc = float(np.mean(y_val_raw == majority_class))

    return {
        "X_tr": t_X_tr,
        "y_tr": t_y_tr,
        "X_val": t_X_val,
        "y_val": t_y_val,
        "X_eval": t_X_eval,
        "y_eval": t_y_eval,
        "eval_row_id": eval_row_id,
        "mean": mean,
        "std": std,
        "majority_class": majority_class,
        "majority_acc": majority_acc,
    }


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Các bước:
      1. nếu shuffle: perm = torch.randperm(len(X), generator=generator, device=X.device); ngược lại arange
      2. for i in range(0, N, batch_size): idx = perm[i:i+batch_size]; yield X[idx], y[idx]
    """
    N = len(X)
    if shuffle:
        perm = torch.randperm(N, generator=generator, device=X.device)
    else:
        perm = torch.arange(N, device=X.device)

    for i in range(0, N, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]


def test_data_pipeline():
    """Hàm kiểm tra toàn diện Part 0 để nghiệm thu qua terminal."""
    print("=" * 60)
    print("BẮT ĐẦU KIỂM TRA NGHIỆM THU PART 0 (data.py)")
    print("=" * 60)

    # 1. Chạy prepare_data
    data = prepare_data(device="cpu")

    # 2. Kiểm tra shape
    assert data["X_tr"].shape == (371847, 54), f"Lỗi shape X_tr: {data['X_tr'].shape}"
    assert data["y_tr"].shape == (371847,), f"Lỗi shape y_tr: {data['y_tr'].shape}"
    assert data["X_val"].shape == (92962, 54), f"Lỗi shape X_val: {data['X_val'].shape}"
    assert data["y_val"].shape == (92962,), f"Lỗi shape y_val: {data['y_val'].shape}"
    assert data["X_eval"].shape == (116203, 54), f"Lỗi shape X_eval: {data['X_eval'].shape}"
    assert data["y_eval"].shape == (116203,), f"Lỗi shape y_eval: {data['y_eval'].shape}"
    print("[✓] 1. Kích thước tập dữ liệu hoàn toàn chính xác:")
    print(f"    - X_tr: {data['X_tr'].shape}, y_tr: {data['y_tr'].shape}")
    print(f"    - X_val: {data['X_val'].shape}, y_val: {data['y_val'].shape}")
    print(f"    - X_eval: {data['X_eval'].shape}, y_eval: {data['y_eval'].shape}")

    # 3. Kiểm tra chuẩn hoá trên 10 cột liên tục
    tr_num = data["X_tr"][:, :N_NUMERIC].numpy()
    mean_val = np.mean(tr_num, axis=0)
    std_val = np.std(tr_num, axis=0)
    assert np.all(np.abs(mean_val) < 1e-3), f"Mean khác 0: {mean_val}"
    assert np.all(np.abs(std_val - 1.0) < 1e-3), f"Std khác 1: {std_val}"
    print("[✓] 2. Chuẩn hoá 10 cột đầu trên X_tr chính xác: Mean ≈ 0, Std ≈ 1")

    # 4. Kiểm tra 44 cột nhị phân không bị chuẩn hoá (chỉ có 0 và 1)
    tr_bin = data["X_tr"][:, N_NUMERIC:].numpy()
    unique_bin = np.unique(tr_bin)
    assert np.all(np.isin(unique_bin, [0.0, 1.0])), f"Cột nhị phân bị thay đổi giá trị: {unique_bin}"
    print("[✓] 3. 44 cột nhị phân (one-hot) giữ nguyên giá trị 0/1")

    # 5. Kiểm tra phân tầng nhãn
    tr_labels, tr_counts = np.unique(data["y_tr"].numpy(), return_counts=True)
    val_labels, val_counts = np.unique(data["y_val"].numpy(), return_counts=True)
    tr_ratio = tr_counts / len(data["y_tr"])
    val_ratio = val_counts / len(data["y_val"])
    assert np.all(np.abs(tr_ratio - val_ratio) < 1e-3), "Phân tầng nhãn giữa train và val bị lệch"
    print("[✓] 4. Phân tầng nhãn giữa train và val cân bằng hoàn hảo:")
    for lbl, r_tr, r_val in zip(tr_labels, tr_ratio, val_ratio):
        print(f"    - Lớp {lbl}: train = {r_tr:.4%}, val = {r_val:.4%}")

    # 6. Kiểm tra baseline đoán lớp đa số
    assert abs(data["majority_acc"] - 0.4876) < 1e-3, f"Lệch accuracy lớp đa số: {data['majority_acc']}"
    print(f"[✓] 5. Chiến lược đoán lớp đa số (lớp {data['majority_class']}): accuracy = {data['majority_acc']:.4%}")

    # 7. Kiểm tra batch generator
    batch_count = 0
    sample_count = 0
    for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], batch_size=512, shuffle=True):
        batch_count += 1
        sample_count += len(xb)
        assert xb.shape[1] == 54
        assert len(xb) == len(yb)
    assert sample_count == len(data["X_tr"]), "Tổng số mẫu qua iterate_batches không khớp"
    assert batch_count == int(np.ceil(len(data["X_tr"]) / 512)), "Số batch không khớp"
    print(f"[✓] 6. Generator chia lô hoạt động tốt: {batch_count} batches với tổng {sample_count} mẫu (batch_size=512)")

    print("=" * 60)
    print(">>> KẾT QUẢ: PART 0 ĐÃ HOÀN THÀNH VÀ ĐẠT TẤT CẢ TIÊU CHUẨN NGHIỆM THU! <<<")
    print("=" * 60)
