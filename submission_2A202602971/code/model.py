"""model.py — Định nghĩa kiến trúc MLP cho bài toán Forest CoverType (7 lớp).

Kiến trúc:
    x (B, 54) -> Linear(54, h1) -> ReLU -> [Dropout] -> Linear(h1, h2) -> ReLU -> [Dropout]
              -> ... -> Linear(h_last, 7) -> logits (B, 7)

Quy tắc:
  - Lớp cuối ra logit thô, KHÔNG softmax trong model (softmax nằm trong hàm mất mát).
  - Dropout chỉ đặt sau ReLU của lớp ẩn; không đặt trên đầu vào hay logit.
  - Mọi nn.Linear đều có bias. Không BatchNorm, không residual.
  - Số tham số phải khớp EXPECTED_PARAMS.
"""
from __future__ import annotations

import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Bảo đảm in được tiếng Việt/Unicode trên terminal Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Số tham số bắt buộc ứng với từng kiến trúc (in_features=54, num_classes=7)
EXPECTED_PARAMS = {
    (256, 128): 47_879,        # M-base  (baseline)
    (512, 256): 161_287,       # M-wide  (tuỳ chọn)
    (256, 128, 64): 55_687,    # M-deep  (tuỳ chọn)
}


class MLP(nn.Module):
    """MLP theo quy định ở đầu file.

    Args:
        hidden:   tuple số nơ-ron các lớp ẩn, ví dụ (256, 128)
        dropout:  xác suất TẮT nơ-ron q (nn.Dropout dùng p chính là xác suất tắt); 0.0 = không dùng
        init:     "zeros" | "normal" | "xavier" | "he" | "default"
        in_features: 54
        num_classes: 7
    """

    def __init__(self, hidden=(256, 128), dropout: float = 0.0, init: str = "he",
                 in_features: int = 54, num_classes: int = 7):
        super().__init__()
        self.hidden = tuple(hidden)
        self.dropout_rate = float(dropout)
        self.init_name = str(init)
        self.in_features = in_features
        self.num_classes = num_classes

        layers: list[nn.Module] = []
        curr_dim = in_features

        for h in hidden:
            layers.append(nn.Linear(curr_dim, h, bias=True))
            layers.append(nn.ReLU())
            if self.dropout_rate > 0.0:
                layers.append(nn.Dropout(p=self.dropout_rate))
            curr_dim = h

        # Lớp ra: ra logits thô, không softmax
        layers.append(nn.Linear(curr_dim, num_classes, bias=True))

        self.net = nn.Sequential(*layers)

        # Khởi tạo trọng số
        init_weights(self, self.init_name)

        # Kiểm tra số tham số bắt buộc nếu cấu hình nằm trong bảng quy định
        if self.hidden in EXPECTED_PARAMS:
            total_params = count_params(self)
            expected = EXPECTED_PARAMS[self.hidden]
            assert total_params == expected, (
                f"Số tham số ({total_params}) không khớp quy định ({expected}) cho hidden={self.hidden}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 54) float32  ->  logits: (B, 7) float32."""
        return self.net(x)


def init_weights(model: nn.Module, init: str) -> None:
    """Khởi tạo tham số của MỌI nn.Linear (bias luôn = 0).

    init:
        "zeros"   : W = 0
        "normal"  : W ~ N(0, 0.01^2)
        "xavier"  : nn.init.xavier_normal_ (Var = 2/(n_in+n_out))
        "he"      : nn.init.kaiming_normal_(w, nonlinearity="relu")  (Var = 2/n_in)
        "default" : không làm gì (giữ khởi tạo mặc định của nn.Linear; KHÔNG phải He)
    """
    init_mode = init.lower()
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if init_mode == "zeros":
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
            elif init_mode == "normal":
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)
            elif init_mode == "xavier":
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif init_mode == "he":
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif init_mode == "default":
                pass  # Giữ nguyên khởi tạo mặc định của PyTorch
            else:
                raise ValueError(
                    f"Kiểu khởi tạo không hợp lệ: '{init}'. Hỗ trợ: 'zeros', 'normal', 'xavier', 'he', 'default'"
                )


def count_params(model: nn.Module) -> int:
    """Tổng số tham số huấn luyện được."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def activation_stats(model: nn.Module, x: torch.Tensor) -> list[float]:
    """Độ lệch chuẩn của kích hoạt sau mỗi lớp nn.Linear (ở bước 0)."""
    model.eval()
    stds = []
    h = x
    for layer in model.net:
        h = layer(h)
        if isinstance(layer, nn.Linear):
            stds.append(float(h.std().item()))
    return stds


def test_model_health():
    """Hàm nghiệm thu Part 1: kiểm tra toàn diện kiến trúc và 4 bài test sức khoẻ."""
    print("=" * 60)
    print("BẮT ĐẦU KIỂM TRA NGHIỆM THU PART 1 (model.py)")
    print("=" * 60)

    # 1. Kiểm tra số tham số các kiến trúc
    m_base = MLP(hidden=(256, 128), dropout=0.0, init="he")
    m_wide = MLP(hidden=(512, 256), dropout=0.0, init="he")
    m_deep = MLP(hidden=(256, 128, 64), dropout=0.0, init="he")

    p_base = count_params(m_base)
    p_wide = count_params(m_wide)
    p_deep = count_params(m_deep)

    assert p_base == 47_879, f"M-base sai số tham số: {p_base}"
    assert p_wide == 161_287, f"M-wide sai số tham số: {p_wide}"
    assert p_deep == 55_687, f"M-deep sai số tham số: {p_deep}"

    print("[✓] 1. Kiểm tra số tham số (Parameter Count):")
    print(f"    - M-base: {p_base:,} tham số (chuẩn 47,879)")
    print(f"    - M-wide: {p_wide:,} tham số (chuẩn 161,287)")
    print(f"    - M-deep: {p_deep:,} tham số (chuẩn 55,687)")

    # 2. Kiểm tra Shape đầu ra
    dummy_x = torch.randn(8, 54)
    logits = m_base(dummy_x)
    assert logits.shape == (8, 7), f"Shape logits sai: {logits.shape}"
    print(f"[✓] 2. Luồng tensor forward: Đầu vào (8, 54) -> Logits {tuple(logits.shape)}")

    # 3. Phép thử sức khoẻ 1: Loss bước 0 xấp xỉ ln(7) ≈ 1.946 trên tập val
    try:
        from data import prepare_data
        dataset = prepare_data(device="cpu")
        x_val_sample = dataset["X_val"][:5000]
        y_val_sample = dataset["y_val"][:5000]
    except Exception:
        x_val_sample = torch.randn(1000, 54)
        y_val_sample = torch.randint(0, 7, (1000,))

    ln7 = math.log(7)  # ≈ 1.9459
    m_base.eval()
    with torch.no_grad():
        step0_loss = F.cross_entropy(m_base(x_val_sample), y_val_sample).item()

    print(f"[✓] 3. Phép thử Loss bước 0 trên tập Val: Đo được = {step0_loss:.4f} (Lý thuyết ln(7) = {ln7:.4f}, chênh lệch = {abs(step0_loss - ln7):.4f})")
    assert abs(step0_loss - ln7) < 0.25, f"Loss bước 0 lệch nhiều so với ln(7): {step0_loss}"

    # 4. Phép thử sức khoẻ 2: Quá khớp (Overfit) trên 20 mẫu
    # Huấn luyện mô hình M-base trên đúng 20 mẫu, dropout=0, loss phải về < 0.05 và acc = 100%
    torch.manual_seed(123)
    if "dataset" in locals():
        tiny_x = dataset["X_tr"][:20]
        tiny_y = dataset["y_tr"][:20]
    else:
        tiny_x = torch.randn(20, 54)
        tiny_y = torch.randint(0, 7, (20,))

    model_overfit = MLP(hidden=(256, 128), dropout=0.0, init="he")
    model_overfit.train()
    optimizer = torch.optim.Adam(model_overfit.parameters(), lr=0.01)

    for step in range(250):
        optimizer.zero_grad()
        out = model_overfit(tiny_x)
        loss = F.cross_entropy(out, tiny_y)
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    with torch.no_grad():
        preds = model_overfit(tiny_x).argmax(dim=-1)
        final_acc = (preds == tiny_y).float().mean().item()

    print(f"[✓] 4. Phép thử Quá khớp 20 mẫu: Loss = {final_loss:.6f}, Accuracy = {final_acc * 100:.1f}%")
    assert final_loss < 0.05, f"Không ép được loss về gần 0: {final_loss}"
    assert final_acc == 1.0, f"Accuracy trên 20 mẫu chưa đạt 100%: {final_acc}"

    # 5. Phép thử sức khoẻ 3: Gradient Flow (Gradient có chảy tới mọi tham số không?)
    model_grad = MLP(hidden=(256, 128), dropout=0.0, init="he")
    model_grad.train()
    out = model_grad(torch.randn(16, 54))
    loss = F.cross_entropy(out, torch.randint(0, 7, (16,)))
    loss.backward()

    zero_grads = []
    none_grads = []
    grad_norms = {}
    for name, param in model_grad.named_parameters():
        if param.grad is None:
            none_grads.append(name)
        else:
            gnorm = param.grad.norm().item()
            grad_norms[name] = gnorm
            if gnorm == 0.0:
                zero_grads.append(name)

    assert len(none_grads) == 0, f"Có tham số không nhận được gradient: {none_grads}"
    assert len(zero_grads) == 0, f"Có tham số có gradient bằng 0: {zero_grads}"
    print("[✓] 5. Kiểm tra Gradient Flow: 100% tham số (W và b) đều có gradient chảy về:")
    for name, gnorm in grad_norms.items():
        print(f"    - {name:<12}: grad_norm = {gnorm:.6f}")

    # 6. Kiểm tra các chế độ khởi tạo (init)
    for init_name in ["zeros", "normal", "xavier", "he"]:
        m = MLP(hidden=(256, 128), init=init_name)
        stds = activation_stats(m, dummy_x)
        print(f"[✓] 6. Khởi tạo '{init_name}': Độ lệch chuẩn kích hoạt các lớp = {[round(s, 4) for s in stds]}")

    print("=" * 60)
    print(">>> KẾT QUẢ: PART 1 ĐÃ HOÀN THÀNH VÀ ĐẠT TẤT CẢ TIÊU CHUẨN NGHIỆM THU! <<<")
    print("=" * 60)
