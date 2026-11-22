import os

import pandas as pd
from sklearn.model_selection import train_test_split

from units.unit_09.src.functions import generate_model_dictionary

metrics = ["accuracy", "matthews_corrcoef"]
gscv_folds = 4

df = pd.read_csv(os.path.join("datasets", "dataset_contamination.csv"))

features = df.columns[:-1]
target = df.columns[-1]

x = df.loc[:, features].to_numpy()
y = df.loc[:, target].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=0)

for metric in metrics:
    print(100 * "=")

    model_dict = generate_model_dictionary(metric, gscv_folds)

    best_classifier = None
    best_score = None
    for classifier in model_dict.keys():
        model = model_dict[classifier]
        model.fit(x_train, y_train)
        score = model.best_score_

        print(f"{classifier:25s} | {metric:20s} (Valid) | {score:6.3f}")
        if (best_score is None) or (score > best_score):
            best_classifier = classifier
            best_score = score

    model = model_dict[best_classifier]
    test_score = model.score(x_test, y_test)
    print(f"\nBest model:\n{best_classifier:25s} | {metric:20s}  (Test) | {test_score:6.3f}\n")
    # TODO: Plot Confusion matrix for best classifier predictions
