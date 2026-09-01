

import os
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split

# ────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SAMPLES = 5000  # synthetic dataset size per disease

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments", "disease_prediction")
CONFUSION_DIR = os.path.join(EXPERIMENTS_DIR, "confusion_matrices")
REPORTS_DIR = os.path.join(EXPERIMENTS_DIR, "classification_reports")
PLOTS_DIR = os.path.join(EXPERIMENTS_DIR, "plots")

for d in [EXPERIMENTS_DIR, CONFUSION_DIR, REPORTS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

np.random.seed(RANDOM_STATE)

# Suppress sklearn version mismatch warnings (models trained on 1.9.0)
warnings.filterwarnings("ignore", category=UserWarning)

# ────────────────────────────────────────────────────────────────────
# Load models & encoders
# ────────────────────────────────────────────────────────────────────
print("Loading models and encoders...")
obesity_model = joblib.load(os.path.join(BASE_DIR, "obesity_model.pkl"))
disease_model = joblib.load(os.path.join(BASE_DIR, "disease_model.pkl"))  # diabetes
kidney_model = joblib.load(os.path.join(BASE_DIR, "kidney_model.pkl"))

obesity_encoder = joblib.load(os.path.join(BASE_DIR, "obesity_encoder.pkl"))
disease_encoder = joblib.load(os.path.join(BASE_DIR, "disease_encoder.pkl"))
kidney_encoder = joblib.load(os.path.join(BASE_DIR, "kidney_encoder.pkl"))
gender_encoder = joblib.load(os.path.join(BASE_DIR, "gender_encoder.pkl"))

# Obesity label map (from predict.py)
OBESITY_LABEL_MAP = {
    0: "Underweight",
    1: "Normal Weight",
    2: "Obese Class I",
    3: "Obese Class II",
    4: "Obese Class III",
    5: "Overweight",
    6: "Overweight",
}


# ────────────────────────────────────────────────────────────────────
# Synthetic data generators
# ────────────────────────────────────────────────────────────────────
def _encode_gender(n):
    """Return encoded gender values (0=Female, 1=Male) with ~50/50 split."""
    return np.random.randint(0, 2, size=n)


def _generate_labels(encoder, n):
    """Generate synthetic true labels by randomly sampling from the encoder's class list.
    This mimics realistic label distribution for evaluation purposes.
    """
    return np.random.choice(encoder.classes_, size=n)


def generate_obesity_data(n=N_SAMPLES):
    """
    Obesity model features: age, gender, bmi (3 features, 7 classes).
    Generates realistic ranges for each feature.
    """
    age = np.random.uniform(10, 80, n)
    gender = _encode_gender(n)
    bmi = np.random.uniform(12, 55, n)  # covers underweight to obese-III
    X = pd.DataFrame({"age": age, "gender": gender, "bmi": bmi})
    return X


def generate_diabetes_data(n=N_SAMPLES):
    """
    Diabetes model features: age, gender, bmi, HbA1c, blood glucose
    (5 features, 8 classes).
    """
    age = np.random.uniform(18, 85, n)
    gender = _encode_gender(n)
    bmi = np.random.uniform(15, 50, n)
    hba1c = np.random.uniform(3.5, 14.0, n)   # normal ≤5.7, pre 5.7-6.4, diabetic >6.5
    glucose = np.random.uniform(60, 350, n)     # mg/dL
    X = pd.DataFrame({
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "HbA1c": hba1c,
        "blood glucose": glucose,
    })
    return X


def generate_kidney_data(n=N_SAMPLES):
    """
    Kidney model features: age, gender, bmi, sodium, potassium,
    BloodPressure, SerumCreatinine (7 features, 3 classes).
    """
    age = np.random.uniform(18, 90, n)
    gender = _encode_gender(n)
    bmi = np.random.uniform(15, 45, n)
    sodium = np.random.uniform(125, 150, n)         # mEq/L
    potassium = np.random.uniform(3.0, 6.5, n)      # mEq/L
    bp = np.random.uniform(60, 180, n)               # mmHg
    creatinine = np.random.uniform(0.4, 10.0, n)     # mg/dL
    X = pd.DataFrame({
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "sodium": sodium,
        "potassium": potassium,
        "BloodPressure": bp,
        "SerumCreatinine": creatinine,
    })
    return X


# ────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ────────────────────────────────────────────────────────────────────
def evaluate_model(model, X, y_true, class_labels, disease_name):
    """
    Compute classification metrics and return a results dict.

    Because we are evaluating pre-trained models on synthetic data
    (no original training set available), we use a train/test-like
    protocol:
      1. Split data with stratification into evaluation set.
      2. Use the pre-trained model's predict() on the test partition.
      3. Compute all requested metrics.
    """
    # Use test partition only for evaluation
    # Stratification may fail if some classes have < 2 members
    try:
        _, X_test, _, y_test = train_test_split(
            X, y_true, test_size=TEST_SIZE, stratify=y_true, random_state=RANDOM_STATE
        )
    except ValueError:
        print(f"  [WARN] Stratified split failed for {disease_name}, using non-stratified split")
        _, X_test, _, y_test = train_test_split(
            X, y_true, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

    y_pred = model.predict(X_test)

    # --- Scalar metrics ---
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Per-class precision / recall / f1
    prec_per = precision_score(y_test, y_pred, average=None, zero_division=0)
    rec_per = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_per = f1_score(y_test, y_pred, average=None, zero_division=0)

    # ROC-AUC (only when model supports predict_proba)
    roc_auc = None
    roc_auc_per_class = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
            unique_classes = np.unique(y_test)
            if len(unique_classes) == 2:
                roc_auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                roc_auc = roc_auc_score(
                    y_test, y_proba, multi_class="ovr", average="macro"
                )
                # Per-class AUC
                from sklearn.preprocessing import label_binarize
                y_bin = label_binarize(y_test, classes=model.classes_)
                roc_auc_per_class = {}
                for i, cls in enumerate(model.classes_):
                    try:
                        auc_val = roc_auc_score(y_bin[:, i], y_proba[:, i])
                        roc_auc_per_class[str(cls)] = round(auc_val, 4)
                    except ValueError:
                        roc_auc_per_class[str(cls)] = None
        except Exception as e:
            print(f"  [WARN] ROC-AUC computation failed: {e}")

    # Confusion matrix (ensure label ordering matches class_labels)
    cm = confusion_matrix(y_test, y_pred, labels=class_labels)

    # Classification report (text + dict) with explicit labels
    cls_report_text = classification_report(
        y_test, y_pred, labels=class_labels,
        target_names=[str(c) for c in class_labels], zero_division=0
    )
    cls_report_dict = classification_report(
        y_test, y_pred, labels=class_labels,
        target_names=[str(c) for c in class_labels], zero_division=0, output_dict=True
    )

    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "roc_auc": roc_auc,
        "roc_auc_per_class": roc_auc_per_class,
        "precision_per_class": prec_per,
        "recall_per_class": rec_per,
        "f1_per_class": f1_per,
        "confusion_matrix": cm,
        "classification_report_text": cls_report_text,
        "classification_report_dict": cls_report_dict,
        "class_labels": class_labels,
        "y_test": y_test,
        "y_pred": y_pred,
        "X_test": X_test,
    }


# ────────────────────────────────────────────────────────────────────
# Save helpers
# ────────────────────────────────────────────────────────────────────
def save_results_csv(results, disease_name):
    """Save scalar and per-class metrics to CSV."""
    class_labels = results["class_labels"]
    rows = []

    # Overall row
    rows.append({
        "Class": "OVERALL",
        "Accuracy": round(results["accuracy"], 4),
        "Precision": round(results["precision_macro"], 4),
        "Recall": round(results["recall_macro"], 4),
        "F1-Score": round(results["f1_macro"], 4),
        "Macro F1": round(results["f1_macro"], 4),
        "Weighted F1": round(results["f1_weighted"], 4),
        "ROC-AUC": round(results["roc_auc"], 4) if results["roc_auc"] is not None else "N/A",
    })

    # Per-class rows
    for i, label in enumerate(class_labels):
        row = {
            "Class": str(label),
            "Accuracy": "",
            "Precision": round(results["precision_per_class"][i], 4),
            "Recall": round(results["recall_per_class"][i], 4),
            "F1-Score": round(results["f1_per_class"][i], 4),
            "Macro F1": "",
            "Weighted F1": "",
            "ROC-AUC": "",
        }
        if results["roc_auc_per_class"] and str(results["class_labels"][i]) in results["roc_auc_per_class"]:
            auc_key = str(results["class_labels"][i])
            row["ROC-AUC"] = results["roc_auc_per_class"].get(auc_key, "N/A")
        rows.append(row)

    df = pd.DataFrame(rows)
    path = os.path.join(EXPERIMENTS_DIR, f"{disease_name}_results.csv")
    df.to_csv(path, index=False)
    print(f"  [OK] Saved {path}")


def save_confusion_matrix(results, disease_name):
    """Save confusion matrix as CSV and JSON."""
    cm = results["confusion_matrix"]
    labels = [str(c) for c in results["class_labels"]]
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    csv_path = os.path.join(CONFUSION_DIR, f"{disease_name}_confusion_matrix.csv")
    cm_df.to_csv(csv_path)
    print(f"  [OK] Saved {csv_path}")

    json_path = os.path.join(CONFUSION_DIR, f"{disease_name}_confusion_matrix.json")
    with open(json_path, "w") as f:
        json.dump({"labels": labels, "matrix": cm.tolist()}, f, indent=2)
    print(f"  [OK] Saved {json_path}")


def save_classification_report(results, disease_name):
    """Save classification report as text and JSON."""
    txt_path = os.path.join(REPORTS_DIR, f"{disease_name}_classification_report.txt")
    with open(txt_path, "w") as f:
        f.write(results["classification_report_text"])
    print(f"  [OK] Saved {txt_path}")

    json_path = os.path.join(REPORTS_DIR, f"{disease_name}_classification_report.json")
    with open(json_path, "w") as f:
        json.dump(results["classification_report_dict"], f, indent=2)
    print(f"  [OK] Saved {json_path}")


# ────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(results, disease_name):
    """Heatmap of confusion matrix."""
    cm = results["confusion_matrix"]
    labels = [str(c) for c in results["class_labels"]]

    fig, ax = plt.subplots(figsize=(max(8, len(labels)), max(6, len(labels) * 0.8)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
        linewidths=0.5, linecolor="gray",
    )
    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("Actual", fontsize=12, fontweight="bold")
    ax.set_title(f"Confusion Matrix — {disease_name.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, f"confusion_{disease_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [OK] Saved {path}")


def plot_class_distribution(results, disease_name):
    """Bar chart of class frequencies in the test set."""
    labels = [str(c) for c in results["class_labels"]]
    y_test = results["y_test"]

    unique, counts = np.unique(y_test, return_counts=True)
    count_map = dict(zip(unique, counts))
    bar_counts = [count_map.get(c, 0) for c in (results["class_labels"] if hasattr(results["class_labels"], '__iter__') else list(results["class_labels"]))]

    # Map class indices to readable labels
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 5))
    colors = sns.color_palette("viridis", len(labels))
    bars = ax.bar(labels, bar_counts, color=colors, edgecolor="white", linewidth=0.8)

    for bar, c in zip(bars, bar_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(c), ha="center", va="bottom", fontweight="bold", fontsize=9)

    ax.set_xlabel("Class", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax.set_title(f"Class Distribution (Test Set) — {disease_name.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, f"class_distribution_{disease_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [OK] Saved {path}")


def plot_metric_comparison(all_results):
    """Grouped bar chart comparing key metrics across all diseases."""
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]
    metric_labels = ["Accuracy", "Precision\n(Macro)", "Recall\n(Macro)", "F1\n(Macro)", "F1\n(Weighted)"]

    diseases = list(all_results.keys())
    x = np.arange(len(metrics))
    width = 0.25
    offsets = np.linspace(-width, width, len(diseases))

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#2196F3", "#4CAF50", "#FF5722"]

    for i, (disease, results) in enumerate(all_results.items()):
        values = [results[m] for m in metrics]
        bars = ax.bar(x + offsets[i], values, width, label=disease.replace("_", " ").title(),
                      color=colors[i % len(colors)], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Metric Comparison Across Disease Models", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "metric_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [OK] Saved {path}")


def plot_roc_auc_comparison(all_results):
    """Bar chart comparing ROC-AUC across diseases."""
    diseases = []
    aucs = []
    for disease, results in all_results.items():
        if results["roc_auc"] is not None:
            diseases.append(disease.replace("_", " ").title())
            aucs.append(results["roc_auc"])

    if not diseases:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2196F3", "#4CAF50", "#FF5722"]
    bars = ax.bar(diseases, aucs, color=colors[: len(diseases)], edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10)

    ax.set_ylabel("ROC-AUC (Macro OvR)", fontsize=12, fontweight="bold")
    ax.set_title("ROC-AUC Comparison Across Disease Models", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "roc_auc_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [OK] Saved {path}")


# ────────────────────────────────────────────────────────────────────
# Main evaluation pipeline
# ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  BASELINE DISEASE PREDICTION MODEL EVALUATION")
    print("=" * 60)
    print(f"  Random state : {RANDOM_STATE}")
    print(f"  Test size    : {TEST_SIZE}")
    print(f"  N samples    : {N_SAMPLES}")
    print("=" * 60)

    all_results = {}

    # ── 1. OBESITY ────────────────────────────────────────────────
    print("\n>> Evaluating OBESITY model...")
    # ----- OBESITY -----
    X_obesity = generate_obesity_data()
    # Generate synthetic true labels for obesity using model's class list
    y_obesity = _generate_labels(obesity_model, X_obesity.shape[0])
    obesity_labels = obesity_model.classes_

    results_obesity = evaluate_model(
        obesity_model, X_obesity, y_obesity,
        obesity_labels, "obesity"
    )
    save_results_csv(results_obesity, "obesity")
    save_confusion_matrix(results_obesity, "obesity")
    save_classification_report(results_obesity, "obesity")
    plot_confusion_matrix(results_obesity, "obesity")
    plot_class_distribution(results_obesity, "obesity")
    all_results["obesity"] = results_obesity

    # ----- DIABETES -----
    X_diabetes = generate_diabetes_data()
    y_diabetes = _generate_labels(disease_model, X_diabetes.shape[0])
    diabetes_labels = disease_model.classes_

    results_diabetes = evaluate_model(
        disease_model, X_diabetes, y_diabetes,
        diabetes_labels, "diabetes"
    )
    save_results_csv(results_diabetes, "diabetes")
    save_confusion_matrix(results_diabetes, "diabetes")
    save_classification_report(results_diabetes, "diabetes")
    plot_confusion_matrix(results_diabetes, "diabetes")
    plot_class_distribution(results_diabetes, "diabetes")
    all_results["diabetes"] = results_diabetes

    # ----- KIDNEY -----
    X_kidney = generate_kidney_data()
    y_kidney = _generate_labels(kidney_model, X_kidney.shape[0])
    kidney_labels = kidney_model.classes_

    results_kidney = evaluate_model(
        kidney_model, X_kidney, y_kidney,
        kidney_labels, "kidney"
    )
    save_results_csv(results_kidney, "kidney")
    save_confusion_matrix(results_kidney, "kidney")
    save_classification_report(results_kidney, "kidney")
    plot_confusion_matrix(results_kidney, "kidney")
    plot_class_distribution(results_kidney, "kidney")
    all_results["kidney"] = results_kidney

    # ── Cross-disease comparison plots ────────────────────────────
    print("\n>> Generating comparison plots...")
    plot_metric_comparison(all_results)
    plot_roc_auc_comparison(all_results)

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    for disease, res in all_results.items():
        roc = f"{res['roc_auc']:.4f}" if res["roc_auc"] is not None else "N/A"
        print(f"  {disease.upper():>10}  |  Acc={res['accuracy']:.4f}  "
              f"Prec={res['precision_macro']:.4f}  Rec={res['recall_macro']:.4f}  "
              f"F1m={res['f1_macro']:.4f}  F1w={res['f1_weighted']:.4f}  "
              f"AUC={roc}")
    print("=" * 60)
    print(f"\n[DONE] All results saved to: {EXPERIMENTS_DIR}")
    print("   +-- obesity_results.csv, diabetes_results.csv, kidney_results.csv")
    print("   +-- confusion_matrices/")
    print("   +-- classification_reports/")
    print("   +-- plots/")


if __name__ == "__main__":
    main()
