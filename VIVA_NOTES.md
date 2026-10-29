# Viva preparation (understand these points in your own words)

1. **Why TF-IDF?** It represents each document numerically and gives more weight to terms that are frequent in a document but less common across the corpus.
2. **Why fit TF-IDF only on training data?** To prevent information from the test set leaking into training.
3. **Why the same split and features?** To make the comparison fair: algorithm is the main changed factor.
4. **Accuracy:** fraction of all test predictions that are correct.
5. **Precision:** among documents predicted as a class, the fraction that truly belong to it.
6. **Recall:** among documents that truly belong to a class, the fraction correctly found.
7. **F1-score:** harmonic mean of precision and recall.
8. **Weighted average:** averages class metrics weighted by the number of true examples in each class.
9. **Confusion matrix:** rows are actual labels and columns are predicted labels in this project.
10. **Decision Tree:** learns word-feature split rules; can overfit a sparse, high-dimensional representation.
11. **KNN:** predicts from the labels of the nearest documents; cosine distance is used for TF-IDF vectors.
12. **Linear SVM:** finds a decision boundary with a large margin; often strong on high-dimensional sparse text.
13. **Multinomial Naive Bayes:** applies Bayes' rule with a simplifying conditional-independence assumption; commonly used for text.
14. **What-if experiment:** SVM `C` is varied across 0.1, 1.0, and 10.0. Smaller C applies stronger regularization; larger C penalizes training errors more strongly. Use the actual CSV values to explain what happened.
15. **How to defend the conclusion:** name the best model according to the actual weighted F1 result, quote its actual metrics, and mention the confusion matrix/per-class errors. Never memorize invented values.
