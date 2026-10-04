"""run_experiments.py — Tự động thực hiện các thí nghiệm Part 3 theo 7 chủ đề.

Các chủ đề:
  1. loss       : Cross-Entropy vs MSE
  2. optimizer  : SGD, SGD+Momentum, Adam (lr=1e-3, 3e-4), AdamW (lr=1e-3)
  3. hparam     : Batch 128, Batch 2048, M-wide, M-deep
  4. dropout    : Dropout q=0.1, q=0.3
  5. clipping   : High LR (0.5) No-clip vs Clip norm=1.0
  6. amp        : BFloat16 Autocast vs FP32
  7. init       : Zeros, Normal, Xavier, He
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

# Bảo đảm in được tiếng Việt/Unicode trên terminal Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from data import prepare_data
from train import run_experiment
from plots import plot_run, plot_compare
from results_table import save_result, load_results

# Định nghĩa danh sách các thí nghiệm
EXPERIMENTS = [
    # --- CHỦ ĐỀ 1: HÀM MẤT MÁT (LOSS) ---
    {
        "exp_id": "loss-mse",
        "group": "loss",
        "description": "MSE Loss trên nhãn one-hot",
        "loss": "mse",
        "optimizer": "sgd_momentum",
        "lr": 0.05,
        "batch": 512,
        "epochs": 20,
        "hidden": (256, 128),
        "dropout": 0.0,
        "init": "he",
        "clip_norm": None,
        "precision": "fp32",
        "seed": 1,
        "hypothesis": "MSE có gradient bị bão hòa khi dự đoán sai lệch nhiều, không tối ưu cho phân loại đa lớp nên macro-F1 sẽ thấp hơn CE.",
    },

    # --- CHỦ ĐỀ 2: BỘ TỐI ƯU (OPTIMIZER) ---
    {
        "exp_id": "opt-sgd-lr0.05",
        "group": "optimizer",
        "description": "SGD thuần không momentum (lr=0.05)",
        "loss": "ce",
        "optimizer": "sgd",
        "lr": 0.05,
        "momentum": 0.0,
        "batch": 512,
        "epochs": 20,
        "hidden": (256, 128),
        "dropout": 0.0,
        "init": "he",
        "clip_norm": None,
        "precision": "fp32",
        "seed": 1,
        "hypothesis": "Không có momentum, SGD thuần hội tụ chậm hơn nhiều và dễ bị kẹt ở vùng thung lũng phẳng.",
    },
    {
        "exp_id": "opt-adam-lr1e-3",
        "group": "optimizer",
        "description": "Adam (lr=1e-3)",
        "loss": "ce",
        "optimizer": "adam",
        "lr": 0.001,
        "batch": 512,
        "epochs": 20,
        "hidden": (256, 128),
        "dropout": 0.0,
        "init": "he",
        "clip_norm": None,
        "precision": "fp32",
        "seed": 1,
        "hypothesis": "Adam với adaptive learning rate sẽ giảm loss cực nhanh ở các epoch đầu và đạt F1 cạnh tranh với SGD+Momentum.",
    },
    {
        "exp_id": "opt-adam-lr3e-4",
        "group": "optimizer",
        "description": "Adam (lr=3e-4)",
        "loss": "ce",
        "optimizer": "adam",
        "lr": 0.0003,
        "batch": 512,
        "epochs": 20,
        "hidden": (256, 128),
        "dropout": 0.0,
        "init": "he",
        "clip_norm": None,
        "precision": "fp32",
        "seed": 1,
        "hypothesis": "Adam với lr nhỏ hơn (3e-4) sẽ hội tụ mượt mà hơn nhưng có thể cần nhiều epoch hơn để đạt đỉnh.",
    },
    {
        "exp_id": "opt-adamw-lr1e-3",
        "group": "optimizer",
        "description": "AdamW (lr=1e-3, weight_decay=0.01)",
        "loss": "ce",
        "optimizer": "adamw",
        "lr": 0.001,
        "weight_decay": 0.01,
        "batch": 512,
        "epochs": 20,
        "hidden": (256, 128),
        "dropout": 0.0,
        "init": "he",
        "clip_norm": None,
        "precision": "fp32",
        "seed": 1,
        "hypothesis": "AdamW tách biệt weight decay khỏi gradient update giúp kiểm soát độ lớn trọng số tốt hơn Adam, cải thiện nhẹ generalization.",
    },

    # --- CHỦ ĐỀ 3: HYPERPARAMETERS (BATCH SIZE & ARCHITECTURE) ---
    {
        "exp_id": "hp-batch128",
        "group": "hparam",
        "description": "Batch size nhỏ (128)",
        "loss": "ce",
        "optimizer": "sgd_momentum",
        "lr": 0.05,
        "batch": 128,
        "epochs": 20,
        "hidden": (256, 128),
        "dropout": 0.0,
        "init": "he",
        "clip_norm": None,
        "precision": "fp32",
        "seed": 1,
        "hypothesis": "Batch 128 có số bước cập nhật gấp 4 lần trong 1 epoch (2905 bước vs 727 bước) và gradient ngẫu nhiên hơn, giúp thoát khỏi cực tiểu địa phương tốt hơn.",
    },
    {
        "exp_id": "hp-batch2048",
        "group": "hparam",
        "description": "Batch size lớn (2048)",
        "loss": "ce",
        "optimizer": "sgd_momentum",
        "lr": 0.05,
        "batch": 2048,
        "epochs": 20,
        "hidden": (256, 128),
        "dropout": 0.0,
        "init": "he",
        "clip_norm": None,
        "precision": "fp32",
        "seed": 1,
        "hypothesis": "Batch 2048 có ít bước cập nhật hơn trong 20 epoch (chỉ 182 bước/epoch) nên với cùng lr=0.05 sẽ hội tụ chậm hơn.",
    },
    {
        "exp_id": "hp-mwide",
        "group": "hparam",
        "description": "M-wide (512 -> 256 -> 7, 161,287 params)",
        "loss": "ce",
        "optimizer": "sgd_momentum",
        "lr": 0.05,
        "batch": 512,
        "epochs": 20,
        "hidden": (512, 256),
        "dropout": 0.0,
        "init": "he",
        "clip_norm": None,
        "precision": "fp32",
        "seed": 1,
        "hypothesis": "Mạng rộng hơn tăng đáng kể năng lực biểu diễn dữ liệu, giúp giảm loss sâu hơn và tăng macro-F1 vượt ngưỡng nhiễu 2σ.",
    },
    {
        "exp_id": "hp-mdeep",
        "group": "hparam",
        "description": "M-deep (256 -> 128 -> 64 -> 7, 55,687 params)",
        "loss": "ce",
        "optimizer": "sgd_momentum",
        "lr": 0.05,
        "batch": 512,
        "epochs": 20,
        "hidden": (256, 128, 64),
        "dropout": 0.0,
        "init": "he",
        "clip_norm": None,
        "precision": "fp32",
        "seed": 1,
        "hypothesis": "Mạng sâu hơn cho phép học các biểu diễn phân cấp trừu tượng hơn, đạt kết quả tốt hơn M-base với số tham số tăng không đáng kể.",
    },

    # --- CHỦ ĐỀ 4: DROPOUT ---
    {
        "exp_id": "drop-0.1",
        "group": "dropout",
        "description": "Dropout q=0.1",
        "loss": "ce",
        "optimizer": "sgd_momentum",
        "lr": 0.05,
        "batch": 512,
        "epochs": 20,
        "hidden": (256, 128),
        "dropout": 0.1,
        "init": "he",
        "clip_norm": None,
        "precision": "fp32",
        "seed": 1,
        "hypothesis": "Dropout nhẹ 0.1 giúp ngăn co-adaptation của các nơ-ron, thu hẹp khoảng cách train-val loss.",
    },
    {
        "exp_id": "drop-0.3",
        "group": "dropout",
        "description": "Dropout q=0.3",
        "loss": "ce",
        "optimizer": "sgd_momentum",
        "lr": 0.05,
        "batch": 512,
        "epochs": 20,
        "hidden": (256, 128),
        "dropout": 0.3,
        "init": "he",
        "clip_norm": None,
        "precision": "fp32",
        "seed": 1,
        "hypothesis": "Do dữ liệu rất lớn (370k mẫu) trong khi mạng nhỏ (47k params) chưa bị quá khớp nặng, dropout cao 0.3 sẽ làm chậm việc học và giảm nhẹ F1.",
    },

    # --- CHỦ ĐỀ 5: GRADIENT CLIPPING ---
    {
        "exp_id": "clip-highlr-noclip",
        "group": "clipping",
        "description": "LR cao (0.5) Không clip gradient",
        "loss": "ce",
        "optimizer": "sgd_momentum",
        "lr": 0.5,
        "batch": 512,
        "epochs": 20,
        "hidden": (256, 128),
        "dropout": 0.0,
        "init": "he",
        "clip_norm": None,
        "precision": "fp32",
        "seed": 1,
        "hypothesis": "Với lr quá lớn (0.5), gradient không bị cắt sẽ gây bước nhảy quá lớn, làm đường loss dao động dữ dội hoặc vỡ hội tụ.",
    },
    {
        "exp_id": "clip-highlr-clip1.0",
        "group": "clipping",
        "description": "LR cao (0.5) Có clip norm=1.0",
        "loss": "ce",
        "optimizer": "sgd_momentum",
        "lr": 0.5,
        "batch": 512,
        "epochs": 20,
        "hidden": (256, 128),
        "dropout": 0.0,
        "init": "he",
        "clip_norm": 1.0,
        "precision": "fp32",
        "seed": 1,
        "hypothesis": "Cắt gradient ở mức 1.0 sẽ chặn các gai gradient lớn, giúp mô hình ổn định ngay cả khi dùng learning rate cao.",
    },

    # --- CHỦ ĐỀ 6: MIXED PRECISION (AMP) ---
    {
        "exp_id": "amp-bf16",
        "group": "amp",
        "description": "Mixed Precision BFloat16",
        "loss": "ce",
        "optimizer": "sgd_momentum",
        "lr": 0.05,
        "batch": 512,
        "epochs": 2,
        "hidden": (256, 128),
        "dropout": 0.0,
        "init": "he",
        "clip_norm": None,
        "precision": "bf16",
        "seed": 1,
        "hypothesis": "BFloat16 duy trì cùng khoảng biểu diễn exponent như FP32 nên độ chính xác gần như tương đương FP32.",
    },

    # --- CHỦ ĐỀ 7: KHỞI TẠO THAM SỐ (INIT) ---
    {
        "exp_id": "init-zeros",
        "group": "init",
        "description": "Khởi tạo tất cả trọng số bằng 0 (Zeros)",
        "loss": "ce",
        "optimizer": "sgd_momentum",
        "lr": 0.05,
        "batch": 512,
        "epochs": 20,
        "hidden": (256, 128),
        "dropout": 0.0,
        "init": "zeros",
        "clip_norm": None,
        "precision": "fp32",
        "seed": 1,
        "hypothesis": "Khởi tạo bằng 0 khiến mọi nơ-ron trong cùng một lớp có gradient y hệt nhau (tính đối xứng), mô hình mất hoàn toàn năng lực học (accuracy đứng yên ở mốc đoán đa số ~48.76%).",
    },
    {
        "exp_id": "init-normal",
        "group": "init",
        "description": "Khởi tạo Normal N(0, 0.01^2)",
        "loss": "ce",
        "optimizer": "sgd_momentum",
        "lr": 0.05,
        "batch": 512,
        "epochs": 20,
        "hidden": (256, 128),
        "dropout": 0.0,
        "init": "normal",
        "clip_norm": None,
        "precision": "fp32",
        "seed": 1,
        "hypothesis": "Phương sai 0.01^2 quá nhỏ khiến tín hiệu bị triệt tiêu qua các lớp sâu, gradient ban đầu rất nhỏ làm mô hình học rất chậm.",
    },
    {
        "exp_id": "init-xavier",
        "group": "init",
        "description": "Khởi tạo Xavier Normal",
        "loss": "ce",
        "optimizer": "sgd_momentum",
        "lr": 0.05,
        "batch": 512,
        "epochs": 20,
        "hidden": (256, 128),
        "dropout": 0.0,
        "init": "xavier",
        "clip_norm": None,
        "precision": "fp32",
        "seed": 1,
        "hypothesis": "Xavier duy trì phương sai tốt cho hàm tanh/sigmoid, nhưng với ReLU sẽ hơi bị giảm phương sai (do nửa âm bị triệt tiêu), kết quả kém nhẹ so với He.",
    },
]


def run_all_or_group(target_group: str = "all"):
    """Thực hiện chạy các thí nghiệm và vẽ đồ thị so sánh nhóm."""
    base_dir = Path(__file__).resolve().parent.parent
    res_dir = base_dir / "results"
    fig_dir = base_dir / "figures"
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    data = prepare_data(device="cpu")

    # Nạp base-s1 để so sánh
    base_s1_path = res_dir / "base-s1.json"
    base_s1 = None
    if base_s1_path.exists():
        import json
        with open(base_s1_path, "r", encoding="utf-8") as f:
            base_s1 = json.load(f)

    # Lọc thí nghiệm theo group
    if target_group.lower() == "all":
        selected_exps = EXPERIMENTS
    else:
        selected_exps = [e for e in EXPERIMENTS if e["group"].lower() == target_group.lower()]

    print("=" * 70)
    print(f"BẮT ĐẦU CHẠY {len(selected_exps)} THÍ NGHIỆM CHO NHÓM: '{target_group.upper()}'")
    print("=" * 70)

    executed_results = {}
    for i, exp in enumerate(selected_exps, start=1):
        exp_id = exp["exp_id"]
        json_file = res_dir / f"{exp_id}.json"

        print(f"\n[{i}/{len(selected_exps)}] >>> THÍ NGHIỆM: {exp_id} ({exp['description']})")
        print(f"  * Dự đoán lý thuyết: {exp['hypothesis']}")

        # Nếu đã chạy rồi thì đọc lại từ file để tiết kiệm thời gian (hoặc chạy mới nếu chưa có)
        if json_file.exists():
            import json
            print(f"  * Đã tìm thấy kết quả đã lưu tại {json_file.name}, đang nạp lại...")
            with open(json_file, "r", encoding="utf-8") as f:
                res = json.load(f)
        else:
            cfg_to_run = {k: v for k, v in exp.items() if k != "hypothesis"}
            res = run_experiment(cfg_to_run, data, verbose=True)
            save_result(res, str(res_dir))

        # Lưu ảnh riêng cho thí nghiệm
        fig_path = str(fig_dir / f"{exp_id}.png")
        plot_run(res, fig_path)

        executed_results[exp_id] = res

        # In tóm tắt kết quả
        s = res["summary"]
        print(f"  * KẾT QUẢ: Val Acc = {s['val_acc']:.4%}, Val Macro-F1 = {s['val_macro_f1']:.4f}, Best Loss = {s['best_val_loss']:.4f}")

    # Vẽ biểu đồ so sánh nhóm
    print("\n" + "=" * 70)
    print("VẼ CÁC BIỂU ĐỒ SO SÁNH THEO NHÓM (COMPARE FIGURES)...")
    print("=" * 70)

    all_res = load_results(str(res_dir))
    groups = set(e["group"] for e in EXPERIMENTS)
    if target_group.lower() != "all":
        groups = {target_group.lower()}

    for grp in sorted(groups):
        # Lấy các thí nghiệm trong nhóm + baseline base-s1 để đối chiếu
        grp_runs = [r for r in all_res if r["cfg"].get("group") == grp]
        if base_s1 and not any(r["cfg"]["exp_id"] == "base-s1" for r in grp_runs):
            grp_runs = [base_s1] + grp_runs

        if len(grp_runs) > 1:
            cmp_path = str(fig_dir / f"compare_{grp}.png")
            # Vẽ so sánh Macro-F1 (hoặc Loss)
            plot_compare(grp_runs, metric="val_macro_f1", path=cmp_path,
                         title=f"So sánh Val Macro-F1: Nhóm {grp.upper()} vs Baseline")
            print(f"[✓] Đã tạo biểu đồ so sánh nhóm '{grp}': {cmp_path}")

    print("\n" + "=" * 70)
    print(">>> HOÀN THÀNH TẤT CẢ THÍ NGHIỆM VÀ ĐỒ THỊ SO SÁNH CHO PART 3! <<<")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=str, default="all", help="all hoặc tên nhóm: loss, optimizer, hparam, dropout, clipping, amp, init")
    args = parser.parse_args()
    run_all_or_group(args.group)
