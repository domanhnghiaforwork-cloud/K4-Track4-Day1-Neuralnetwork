"""Cách chấm điểm chính thức trên tập eval.

Bạn nộp một file dự đoán cho TOÀN BỘ tập eval; script này tính điểm. Giảng viên dùng
đúng script này để chấm, nên hãy chạy nó trước khi nộp và ghi kết quả vào báo cáo.

    python scripts/evaluate.py --pred submission_<MSSV>/predictions_eval.csv
    python scripts/evaluate.py --pred ... --out submission_<MSSV>/eval_result.json

File dự đoán (CSV, có dòng tiêu đề):
    row_id,pred
    123,2
    ...
  - row_id : số nguyên, lấy từ data/processed/eval.npz (khoá row_id)
  - pred   : nhãn dự đoán là số nguyên 0..6 (CÙNG hệ nhãn với y: đã trừ 1)
  - phải có đủ 116 203 row_id của tập eval, mỗi row_id đúng một lần, thứ tự tuỳ ý

Chỉ số:
  - accuracy  = số dự đoán đúng / số mẫu
  - macro-F1  = trung bình cộng (không trọng số) của F1 từng lớp trong 7 lớp
        F1_c = 2 * P_c * R_c / (P_c + R_c)   (bằng 0 nếu P_c + R_c = 0)
        P_c = TP_c / (TP_c + FP_c),  R_c = TP_c / (TP_c + FN_c)
  - Chỉ số CHÍNH để xếp mức điểm là macro-F1 (dữ liệu mất cân bằng: lớp 2 chiếm ~49%,
    lớp 4 chỉ ~0,5%). Accuracy là chỉ số phụ.
"""
import argparse
import json
import sys

# Bảo đảm in được tiếng Việt/Unicode trên terminal Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd

LABEL = "Cover_Type"
N_CLASSES = 7


def load_eval_labels(data_path, meta_path):
    meta = pd.read_csv(meta_path)
    y_all = pd.read_csv(data_path, usecols=[LABEL])[LABEL].to_numpy() - 1
    ids = meta.loc[meta["split"] == "eval", "row_id"].to_numpy()
    return pd.Series(y_all[ids], index=ids, name="y")


def confusion(y_true, y_pred, k=N_CLASSES):
    cm = np.zeros((k, k), dtype=np.int64)          # hàng = nhãn thật, cột = dự đoán
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def scores(cm):
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return prec, rec, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="CSV với cột row_id,pred")
    ap.add_argument("--data", default="data/covtype.csv.gz")
    ap.add_argument("--meta", default="data/split_metadata.csv")
    ap.add_argument("--out", default=None, help="ghi kết quả ra JSON (tuỳ chọn)")
    args = ap.parse_args()

    truth = load_eval_labels(args.data, args.meta)
    pred = pd.read_csv(args.pred)

    # ---- kiểm tra hợp lệ (lỗi ở đây nghĩa là file nộp không được chấm)
    errs = []
    if list(pred.columns) != ["row_id", "pred"]:
        errs.append(f"cột phải là đúng 'row_id,pred', hiện là {list(pred.columns)}")
    else:
        if pred["row_id"].isna().any() or pred["pred"].isna().any():
            errs.append("có giá trị trống")
        elif not (np.array_equal(pred["row_id"], pred["row_id"].astype(int)) and
                  np.array_equal(pred["pred"], pred["pred"].astype(int))):
            errs.append("row_id và pred phải là số nguyên")
        else:
            if pred["row_id"].duplicated().any():
                errs.append(f"có {int(pred['row_id'].duplicated().sum())} row_id bị lặp")
            extra = set(pred["row_id"]) - set(truth.index)
            missing = set(truth.index) - set(pred["row_id"])
            if extra:
                errs.append(f"{len(extra)} row_id không thuộc tập eval (nhầm sang train?)")
            if missing:
                errs.append(f"thiếu {len(missing)} row_id của tập eval (cần đủ {len(truth)})")
            if ((pred["pred"] < 0) | (pred["pred"] > N_CLASSES - 1)).any():
                errs.append("pred phải nằm trong 0..6 (nhớ đã trừ 1 so với Cover_Type gốc)")
    if errs:
        print("FILE DỰ ĐOÁN KHÔNG HỢP LỆ:\n  - " + "\n  - ".join(errs))
        sys.exit(1)

    pred = pred.set_index("row_id")["pred"].astype(int).loc[truth.index]
    cm = confusion(truth.to_numpy(), pred.to_numpy())
    prec, rec, f1 = scores(cm)
    acc = float(np.trace(cm) / cm.sum())
    macro_f1 = float(f1.mean())

    print(f"n_eval = {int(cm.sum())}")
    print(f"accuracy = {acc:.4f}")
    print(f"macro_f1 = {macro_f1:.4f}   <- chỉ số chính")
    print("\nlớp  support  precision  recall    f1")
    for c in range(N_CLASSES):
        print(f"{c:>3d}  {int(cm[c].sum()):7d}  {prec[c]:9.4f}  {rec[c]:6.4f}  {f1[c]:6.4f}")
    print("\nma trận nhầm lẫn (hàng = thật, cột = dự đoán):")
    print(pd.DataFrame(cm).to_string())

    if args.out:
        res = dict(n_eval=int(cm.sum()), accuracy=acc, macro_f1=macro_f1,
                   per_class=[dict(cls=c, support=int(cm[c].sum()), precision=float(prec[c]),
                                   recall=float(rec[c]), f1=float(f1[c])) for c in range(N_CLASSES)],
                   confusion_matrix=cm.tolist())
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print("\nđã ghi", args.out)


if __name__ == "__main__":
    main()
