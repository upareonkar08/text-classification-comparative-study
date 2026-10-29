"""
Comparative Text Classification Experiment
Dataset: 20 Newsgroups, two clearly defined classes:
  1. sci.space
  2. rec.sport.hockey

Run:
    python run_experiment.py

The script downloads the labelled dataset through scikit-learn on first run,
uses one stratified train/test split, one TF-IDF representation, and evaluates
four classifiers. It saves actual results and graphs under ./outputs/.
"""
from pathlib import Path
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

warnings.filterwarnings("ignore", category=UserWarning)
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

CATEGORIES = ["sci.space", "rec.sport.hockey"]
LABEL_NAMES = ["sci.space", "rec.sport.hockey"]
RANDOM_STATE = 42
TEST_SIZE = 0.20

def main():
    print("Downloading/loading labelled 20 Newsgroups dataset...")
    dataset = fetch_20newsgroups(
        subset="all",
        categories=CATEGORIES,
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=RANDOM_STATE,
        download_if_missing=True,
    )
    texts = dataset.data
    y = dataset.target

    print(f"Documents: {len(texts)}")
    print("Class counts:")
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name}: {sum(y == i)}")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Identical preprocessing and numerical features for all classifiers.
    # TF-IDF is fitted on training data only to avoid test-data leakage.
    tfidf = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        max_features=30000,
        min_df=2,
        sublinear_tf=True,
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    print(f"Training TF-IDF shape: {X_train_tfidf.shape}")
    print(f"Test TF-IDF shape: {X_test_tfidf.shape}")

    models = {
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE, max_depth=None
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5, metric="cosine", algorithm="brute"),
        "SVM": LinearSVC(C=1.0, random_state=RANDOM_STATE, max_iter=10000),
        "Naive Bayes": MultinomialNB(alpha=1.0),
    }

    rows = []
    predictions = {}
    reports = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_tfidf, y_train)
        pred = model.predict(X_test_tfidf)
        predictions[name] = pred
        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision (weighted)": precision_score(y_test, pred, average="weighted", zero_division=0),
            "Recall (weighted)": recall_score(y_test, pred, average="weighted", zero_division=0),
            "F1-score (weighted)": f1_score(y_test, pred, average="weighted", zero_division=0),
        }
        rows.append(metrics)
        reports[name] = classification_report(
            y_test, pred, target_names=LABEL_NAMES, zero_division=0
        )
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES
        )
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title(f"Confusion Matrix: {name}")
        plt.tight_layout()
        plt.savefig(OUT / f"confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=180)
        plt.close()

    results = pd.DataFrame(rows).sort_values("F1-score (weighted)", ascending=False)
    results.to_csv(OUT / "model_comparison.csv", index=False)
    print("\n=== ACTUAL TEST RESULTS ===")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Main comparison graph
    plot_df = results.set_index("Model")[[
        "Accuracy", "Precision (weighted)", "Recall (weighted)", "F1-score (weighted)"
    ]]
    ax = plot_df.plot(kind="bar", figsize=(12, 6), ylim=(0, 1.05))
    ax.set_ylabel("Score")
    ax.set_title("Classifier Comparison on the Same TF-IDF Test Set")
    ax.set_xlabel("")
    plt.xticks(rotation=0)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT / "metric_comparison.png", dpi=180)
    plt.close()

    with open(OUT / "classification_reports.txt", "w", encoding="utf-8") as f:
        for name, report in reports.items():
            f.write(f"\n{'='*65}\n{name}\n{'='*65}\n{report}\n")

    # What-if experiment: change only the SVM regularization parameter C.
    # Same train/test split and same TF-IDF features.
    what_if_rows = []
    for c_value in [0.1, 1.0, 10.0]:
        svm = LinearSVC(C=c_value, random_state=RANDOM_STATE, max_iter=10000)
        svm.fit(X_train_tfidf, y_train)
        pred = svm.predict(X_test_tfidf)
        what_if_rows.append({
            "SVM C": c_value,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision (weighted)": precision_score(y_test, pred, average="weighted", zero_division=0),
            "Recall (weighted)": recall_score(y_test, pred, average="weighted", zero_division=0),
            "F1-score (weighted)": f1_score(y_test, pred, average="weighted", zero_division=0),
        })
    what_if = pd.DataFrame(what_if_rows)
    what_if.to_csv(OUT / "what_if_svm_C.csv", index=False)
    print("\n=== WHAT-IF: SVM REGULARIZATION C ===")
    print(what_if.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    ax = what_if.set_index("SVM C")[["Accuracy", "F1-score (weighted)"]].plot(
        marker="o", figsize=(8, 5), ylim=(0, 1.05)
    )
    ax.set_title("What-if Experiment: Effect of SVM C")
    ax.set_ylabel("Score")
    plt.tight_layout()
    plt.savefig(OUT / "what_if_svm_C.png", dpi=180)
    plt.close()

    best = results.iloc[0]
    summary = {
        "dataset": "20 Newsgroups (sci.space vs rec.sport.hockey)",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "train_documents": len(X_train),
        "test_documents": len(X_test),
        "tfidf_features": int(X_train_tfidf.shape[1]),
        "best_model_by_weighted_f1": str(best["Model"]),
        "best_weighted_f1": float(best["F1-score (weighted)"]),
        "metrics": results.to_dict(orient="records"),
        "what_if_svm_C": what_if.to_dict(orient="records"),
    }
    with open(OUT / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Build a report from real metrics generated by this run.
    report_lines = [
        "# Comparative Study of Text Classification Algorithms",
        "",
        "## 1. Objective",
        "Compare Decision Tree, K-Nearest Neighbours (KNN), Support Vector Machine (SVM), "
        "and Naive Bayes using the same labelled text dataset, TF-IDF representation, "
        "train/test split, and evaluation metrics.",
        "",
        "## 2. Dataset and classes",
        "The experiment uses scikit-learn's 20 Newsgroups dataset, restricted to two classes:",
        "- `sci.space`: posts about space, astronomy, and space exploration.",
        "- `rec.sport.hockey`: posts about hockey and related sports discussion.",
        f"- Total documents loaded: {len(texts)}",
        f"- Training documents: {len(X_train)}",
        f"- Test documents: {len(X_test)}",
        "",
        "## 3. Preprocessing and features",
        "Headers, footers, and quoted text were removed using the dataset loader. "
        "TF-IDF vectorization used lowercase text, Unicode accent stripping, English stop-word "
        "removal, unigrams and bigrams, `min_df=2`, `max_features=30000`, and sublinear term frequency. "
        "The vectorizer was fitted on the training split only, then applied to the test split.",
        f"- TF-IDF vocabulary size after filtering: {X_train_tfidf.shape[1]}",
        "- Split: stratified 80% training / 20% test, random_state=42.",
        "",
        "## 4. Algorithms",
        "- Decision Tree: `DecisionTreeClassifier(random_state=42)`.",
        "- KNN: 5 neighbours, cosine distance, brute-force search.",
        "- SVM: linear SVM (`LinearSVC`, C=1.0).",
        "- Naive Bayes: Multinomial Naive Bayes (`alpha=1.0`).",
        "",
        "## 5. Results",
        results.to_markdown(index=False, floatfmt=".4f"),
        "",
        "The full classification reports and individual confusion matrices are saved in the outputs folder. "
        "See `metric_comparison.png` for the metric comparison graph.",
        "",
        "## 6. Confusion matrix analysis",
    ]
    for name in models:
        cm = confusion_matrix(y_test, predictions[name], labels=[0, 1])
        report_lines.append(
            f"- **{name}:** rows are true labels and columns are predicted labels; "
            f"label order is `{LABEL_NAMES}`. Matrix: `[[{cm[0,0]}, {cm[0,1]}], [{cm[1,0]}, {cm[1,1]}]]`."
        )
    report_lines += [
        "",
        "## 7. What-if experiment: SVM parameter C",
        "Only the SVM regularization parameter C was changed (0.1, 1.0, 10.0). "
        "The dataset split and TF-IDF features remained unchanged.",
        what_if.to_markdown(index=False, floatfmt=".4f"),
        "",
        "See `what_if_svm_C.png` for the graph. Explain the observed trend using the values above: "
        "C controls the penalty for classification errors, so changing it changes the margin/error trade-off. "
        "Do not claim that performance increased or decreased unless the table shows it.",
        "",
        "## 8. Best model and recommendation",
        f"Using weighted F1-score as the primary selection metric, the best model in this run is **{best['Model']}** "
        f"(weighted F1 = {best['F1-score (weighted)']:.4f}; accuracy = {best['Accuracy']:.4f}).",
        "Recommend this model for this dataset based on the observed test metrics, while noting that results depend "
        "on the dataset, split, feature representation, and chosen hyperparameters. Weighted metrics account for "
        "class support; inspect the confusion matrices and per-class report before making a broader claim.",
        "",
        "## 9. Why the models may differ",
        "- Linear SVM often works well with sparse, high-dimensional TF-IDF features because it learns a separating boundary.",
        "- Multinomial Naive Bayes is efficient for word-count-like features and models class-conditional token evidence, "
        "but its conditional-independence assumption is simplified.",
        "- KNN relies on document similarity; sparse text vectors can make neighbourhood quality and prediction cost important.",
        "- A Decision Tree can capture nonlinear rules but may overfit sparse, high-dimensional text unless controlled.",
        "",
        "## 10. Limitations",
        "This is a two-class subset of a larger news dataset. Results should not be generalized to all text-classification "
        "tasks without further testing. The split is fixed for reproducibility; repeated cross-validation could give a "
        "more stable estimate.",
        "",
        "## 11. Reproducibility",
        "Run `python run_experiment.py`. The script saves CSV tables, JSON summary, classification reports, "
        "confusion matrices, and graphs under `outputs/`.",
    ]
    (OUT / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"\nBest model by weighted F1: {best['Model']} ({best['F1-score (weighted)']:.4f})")
    print(f"All actual outputs saved to: {OUT.resolve()}")
    print("IMPORTANT: Submit the generated results from your own run, not guessed numbers.")

if __name__ == "__main__":
    main()
