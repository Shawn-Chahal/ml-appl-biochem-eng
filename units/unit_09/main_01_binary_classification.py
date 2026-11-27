# ============================================================================
#   Binary Classification
# ============================================================================
#
#   * This script evaluates several machine learning classifiers
#     for identifying mycoplasma contamination from bioprocess
#     measurements.
#
#   * Candidate models include logistic regression, support vector
#     machine, k-nearest neighbors, Gaussian naive Bayes,
#     decision tree, and random forest.
#
#   * The selected model is evaluated on a separate test dataset
#     using accuracy and Matthews correlation coefficient (MCC).
#
#   * A confusion matrix is generated to visualize classification
#     performance, and the optimized decision tree is plotted to
#     illustrate an interpretable machine learning model.
#
# ============================================================================

import os

import pandas as pd
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.model_selection import train_test_split

from units.unit_09.src.functions import generate_model_dictionary, plot_confusion_matrix, plot_decision_tree

# Load the dataset
df = pd.read_csv(os.path.join("datasets", "dataset_contamination.csv"))

# Extract features and target
features = df.columns[:-1]
target = df.columns[-1]

# Convert DataFrame slices to numpy arrays
x = df.loc[:, features].to_numpy()
y = df.loc[:, target].to_numpy()

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=0)

# Generate the dictionary of classifier models
model_dict = generate_model_dictionary()

# Train each GridSearchCV model and keep track of the best classifier
best_classifier = None
best_score = None
for classifier in model_dict.keys():
    model = model_dict[classifier]
    model.fit(x_train, y_train)
    score = model.best_score_

    print(f"{classifier:25s} | Accuracy (Valid): {score:6.1%} | {model.best_params_}")
    if (best_score is None) or (score > best_score):
        best_classifier = classifier
        best_score = score

# Test the model with the best GridSearchCV score
model = model_dict[best_classifier]
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

print()
print("Best model:")
print(f"{best_classifier:25s} | Accuracy  (Test): {acc:.1%} | MCC (Test): {mcc:.3f} | {model.best_params_}")
print()

plot_confusion_matrix(
    y_true=y_test,
    y_pred=model.predict(x_test),
    title=f"Identifying mycoplasma contamination using:\n{best_classifier} {model.best_params_}\nAccuracy: {acc:.1%} | MCC: {mcc:.3f}",
    filepath=os.path.join("figures", "figure_01_confusion_matrix.png")
)

plot_decision_tree(
    decision_tree=model_dict["Decision Tree"].best_estimator_,
    feature_names=features,
    class_names=model_dict["Decision Tree"].best_estimator_.classes_,
    fontsize=8,
    filepath=os.path.join("figures", "figure_01_decision_tree.png")
)
