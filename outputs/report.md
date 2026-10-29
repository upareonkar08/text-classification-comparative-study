# Comparative Study of Text Classification Algorithms

## 1. Objective
Compare Decision Tree, K-Nearest Neighbours (KNN), Support Vector Machine (SVM), and Naive Bayes using the same labelled text dataset, TF-IDF representation, train/test split, and evaluation metrics.

## 2. Dataset and classes
The experiment uses scikit-learn's 20 Newsgroups dataset, restricted to two classes:
- `sci.space`: posts about space, astronomy, and space exploration.
- `rec.sport.hockey`: posts about hockey and related sports discussion.
- Total documents loaded: 1986
- Training documents: 1588
- Test documents: 398

## 3. Preprocessing and features
Headers, footers, and quoted text were removed using the dataset loader. TF-IDF vectorization used lowercase text, Unicode accent stripping, English stop-word removal, unigrams and bigrams, `min_df=2`, `max_features=30000`, and sublinear term frequency. The vectorizer was fitted on the training split only, then applied to the test split.
- TF-IDF vocabulary size after filtering: 23356
- Split: stratified 80% training / 20% test, random_state=42.

## 4. Algorithms
- Decision Tree: `DecisionTreeClassifier(random_state=42)`.
- KNN: 5 neighbours, cosine distance, brute-force search.
- SVM: linear SVM (`LinearSVC`, C=1.0).
- Naive Bayes: Multinomial Naive Bayes (`alpha=1.0`).

## 5. Results
| Model         |   Accuracy |   Precision (weighted) |   Recall (weighted) |   F1-score (weighted) |
|:--------------|-----------:|-----------------------:|--------------------:|----------------------:|
| SVM           |     0.9573 |                 0.9582 |              0.9573 |                0.9573 |
| Naive Bayes   |     0.9548 |                 0.9552 |              0.9548 |                0.9548 |
| KNN           |     0.9422 |                 0.9422 |              0.9422 |                0.9422 |
| Decision Tree |     0.8819 |                 0.8848 |              0.8819 |                0.8817 |

The full classification reports and individual confusion matrices are saved in the outputs folder. See `metric_comparison.png` for the metric comparison graph.

## 6. Confusion matrix analysis
- **Decision Tree:** rows are true labels and columns are predicted labels; label order is `['sci.space', 'rec.sport.hockey']`. Matrix: `[[168, 32], [15, 183]]`.
- **KNN:** rows are true labels and columns are predicted labels; label order is `['sci.space', 'rec.sport.hockey']`. Matrix: `[[189, 11], [12, 186]]`.
- **SVM:** rows are true labels and columns are predicted labels; label order is `['sci.space', 'rec.sport.hockey']`. Matrix: `[[187, 13], [4, 194]]`.
- **Naive Bayes:** rows are true labels and columns are predicted labels; label order is `['sci.space', 'rec.sport.hockey']`. Matrix: `[[194, 6], [12, 186]]`.

## 7. What-if experiment: SVM parameter C
Only the SVM regularization parameter C was changed (0.1, 1.0, 10.0). The dataset split and TF-IDF features remained unchanged.
|   SVM C |   Accuracy |   Precision (weighted) |   Recall (weighted) |   F1-score (weighted) |
|--------:|-----------:|-----------------------:|--------------------:|----------------------:|
|  0.1000 |     0.9472 |                 0.9505 |              0.9472 |                0.9472 |
|  1.0000 |     0.9573 |                 0.9582 |              0.9573 |                0.9573 |
| 10.0000 |     0.9548 |                 0.9550 |              0.9548 |                0.9548 |

See `what_if_svm_C.png` for the graph. Explain the observed trend using the values above: C controls the penalty for classification errors, so changing it changes the margin/error trade-off. Do not claim that performance increased or decreased unless the table shows it.

## 8. Best model and recommendation
Using weighted F1-score as the primary selection metric, the best model in this run is **SVM** (weighted F1 = 0.9573; accuracy = 0.9573).
Recommend this model for this dataset based on the observed test metrics, while noting that results depend on the dataset, split, feature representation, and chosen hyperparameters. Weighted metrics account for class support; inspect the confusion matrices and per-class report before making a broader claim.

## 9. Why the models may differ
- Linear SVM often works well with sparse, high-dimensional TF-IDF features because it learns a separating boundary.
- Multinomial Naive Bayes is efficient for word-count-like features and models class-conditional token evidence, but its conditional-independence assumption is simplified.
- KNN relies on document similarity; sparse text vectors can make neighbourhood quality and prediction cost important.
- A Decision Tree can capture nonlinear rules but may overfit sparse, high-dimensional text unless controlled.

## 10. Limitations
This is a two-class subset of a larger news dataset. Results should not be generalized to all text-classification tasks without further testing. The split is fixed for reproducibility; repeated cross-validation could give a more stable estimate.

## 11. Reproducibility
Run `python run_experiment.py`. The script saves CSV tables, JSON summary, classification reports, confusion matrices, and graphs under `outputs/`.