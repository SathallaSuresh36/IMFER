import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def compute_metrics_from_cm(cm: np.ndarray) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    num_classes = cm.shape[0]
    total = int(cm.sum())
    correct = int(np.trace(cm))
    accuracy = safe_div(correct, total)

    class_rows: List[Dict[str, float]] = []
    precisions, recalls, f1s, supports = [], [], [], []

    for i in range(num_classes):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        support = int(cm[i, :].sum())

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)

        class_rows.append(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

    support_arr = np.array(supports, dtype=np.float64)
    weights = support_arr / support_arr.sum() if support_arr.sum() else np.zeros_like(support_arr)

    macro = {
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "f1": float(np.mean(f1s)) if f1s else 0.0,
        "support": int(total),
    }
    weighted = {
        "precision": float(np.sum(np.array(precisions) * weights)) if precisions else 0.0,
        "recall": float(np.sum(np.array(recalls) * weights)) if recalls else 0.0,
        "f1": float(np.sum(np.array(f1s) * weights)) if f1s else 0.0,
        "support": int(total),
    }

    overall = {
        "accuracy": float(accuracy),
        "support": int(total),
    }

    return class_rows, {"macro": macro, "weighted": weighted, "overall": overall}


def read_predictions(predictions_csv: str) -> Tuple[np.ndarray, np.ndarray]:
    y_true, y_pred = [], []
    with open(predictions_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y_true.append(int(row["y_true"]))
            y_pred.append(int(row["y_pred"]))
    return np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)


def render_report_text(class_names: List[str], class_rows: List[Dict[str, float]], summary: Dict[str, Dict[str, float]]) -> str:
    lines = []
    lines.append("Classification report")
    lines.append("" )
    lines.append(f"{'class':<12} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
    lines.append("-" * 58)

    for idx, name in enumerate(class_names):
        row = class_rows[idx]
        lines.append(
            f"{name:<12} {row['precision']*100:>9.2f}% {row['recall']*100:>9.2f}% {row['f1']*100:>9.2f}% {row['support']:>10d}"
        )

    lines.append("-" * 58)
    lines.append(
        f"{'accuracy':<12} {'':>10} {'':>10} {summary['overall']['accuracy']*100:>9.2f}% {summary['overall']['support']:>10d}"
    )
    lines.append(
        f"{'macro avg':<12} {summary['macro']['precision']*100:>9.2f}% {summary['macro']['recall']*100:>9.2f}% {summary['macro']['f1']*100:>9.2f}% {summary['macro']['support']:>10d}"
    )
    lines.append(
        f"{'weighted avg':<12} {summary['weighted']['precision']*100:>9.2f}% {summary['weighted']['recall']*100:>9.2f}% {summary['weighted']['f1']*100:>9.2f}% {summary['weighted']['support']:>10d}"
    )
    return "\n".join(lines)


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: str, dataset_name: str = "Dataset"):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_title(f"{dataset_name.upper()} Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    threshold = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=8,
            )

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_per_class_f1(class_names: List[str], class_rows: List[Dict[str, float]], save_path: str, dataset_name: str = "Dataset"):
    f1_values = [row["f1"] * 100.0 for row in class_rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(class_names))
    bars = ax.bar(x, f1_values, color="#e67e22", alpha=0.9)
    ax.set_ylim(0, 100)
    ax.set_ylabel("F1 (%)")
    ax.set_title(f"Per-Class F1 ({dataset_name.upper()})")
    ax.grid(axis="y", alpha=0.25)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")

    for b, v in zip(bars, f1_values):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Print classification report and draw classification images")
    parser.add_argument("--predictions_csv", required=True)
    parser.add_argument("--class_names", required=True, help="Comma-separated class names in label-id order")
    parser.add_argument("--dataset_name", default="dataset", help="Name of the dataset for titles and labels")
    parser.add_argument("--output_dir", default="./figures")
    parser.add_argument("--report_out", default="./artifacts/aggregate/classification_report.txt")
    parser.add_argument("--json_out", default="./artifacts/aggregate/classification_report.json")
    args = parser.parse_args()

    class_names = [x.strip() for x in args.class_names.split(",") if x.strip()]
    y_true, y_pred = read_predictions(args.predictions_csv)

    if y_true.size == 0:
        raise ValueError("No rows found in predictions CSV")

    num_classes = len(class_names)
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=num_classes)
    class_rows, summary = compute_metrics_from_cm(cm)

    report_text = render_report_text(class_names, class_rows, summary)
    print(report_text)

    os.makedirs(os.path.dirname(args.report_out), exist_ok=True)
    with open(args.report_out, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")

    payload = {
        "class_names": class_names,
        "class_metrics": class_rows,
        "summary": summary,
        "confusion_matrix": cm.tolist(),
    }
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    f1_path = os.path.join(args.output_dir, "per_class_f1.png")
    plot_confusion_matrix(cm, class_names, cm_path, dataset_name=args.dataset_name)
    plot_per_class_f1(class_names, class_rows, f1_path, dataset_name=args.dataset_name)

    print(f"\nSaved report text: {args.report_out}")
    print(f"Saved report json: {args.json_out}")
    print(f"Saved image: {cm_path}")
    print(f"Saved image: {f1_path}")


if __name__ == "__main__":
    main()
