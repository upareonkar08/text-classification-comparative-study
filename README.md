# Text Classification Comparative Study

This project compares Decision Tree, KNN, SVM, and Naive Bayes on the same labelled dataset using TF-IDF features.

## Dataset and classes
Uses scikit-learn's 20 Newsgroups dataset, restricted to:
- `sci.space`: space/astronomy discussions
- `rec.sport.hockey`: hockey/sports discussions

The dataset is downloaded automatically on the first run. Internet access is required for the initial download.

## Run the experiment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies and run:
```bash
pip install -r requirements.txt
python run_experiment.py
```

## Outputs
The `outputs/` folder is generated after the experiment runs. It includes:
- `model_comparison.csv`: Accuracy, weighted Precision, weighted Recall, weighted F1
- `metric_comparison.png`: model comparison graph
- `confusion_matrix_*.png`: confusion matrix for each classifier
- `classification_reports.txt`: per-class metrics
- `what_if_svm_C.csv` and `what_if_svm_C.png`: what-if experiment changing SVM `C`
- `report.md`: report populated with metrics from the actual run
- `experiment_summary.json`: reproducibility summary

## Fair-comparison protocol
- Same labelled documents and stratified train/test split (`random_state=42`)
- Same train/test split for all four classifiers
- One TF-IDF vectorizer fitted on training documents only
- Same test set and same metrics for every classifier
- What-if experiment changes only SVM `C` while holding the split and TF-IDF features fixed

## Important academic integrity note
Do not put made-up metrics in your submission. Run the script yourself, inspect the generated tables/graphs/confusion matrices, and be prepared to explain the code and results in your viva. The results may vary slightly across software versions.
