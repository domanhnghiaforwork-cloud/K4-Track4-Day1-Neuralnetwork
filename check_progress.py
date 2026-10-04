import sys
import json
import glob
from pathlib import Path

# Bảo đảm in được tiếng Việt/Unicode trên terminal Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

res_dir = Path("submission_2A202602971/results")
files = sorted(res_dir.glob("*.json"))

TOTAL_EXPERIMENTS = 17
part3_count = sum(1 for f in files if not f.stem.startswith("base-") and f.stem != "test-base-part2")

print("=" * 85)
print(f"TIẾN ĐỘ THÍ NGHIỆM: Đã hoàn thành {part3_count}/{TOTAL_EXPERIMENTS} thí nghiệm Part 3 (+ 3 seed Baseline)")
print("=" * 85)
print(f"{'EXP_ID':<20} | {'GROUP':<10} | {'BEST_LOSS':<10} | {'VAL_ACC':<10} | {'VAL_F1':<10} | {'TIME/EP':<10}")
print("-" * 85)
for f in files:
    with open(f, "r", encoding="utf-8") as fp:
        d = json.load(fp)
    cfg = d.get("cfg", {})
    s = d.get("summary", {})
    exp_id = cfg.get("exp_id", f.stem)
    grp = cfg.get("group", "")
    loss = s.get("best_val_loss", 0.0)
    acc = s.get("val_acc", 0.0)
    f1 = s.get("val_macro_f1", 0.0)
    time_s = s.get("time_per_epoch_s", 0.0)
    print(f"{exp_id:<20} | {grp:<10} | {loss:<10.4f} | {acc*100:<9.2f}% | {f1:<10.4f} | {time_s:<9.1f}s")
print("=" * 85)
