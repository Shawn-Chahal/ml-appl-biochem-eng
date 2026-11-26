import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.model_selection import train_test_split

from units.unit_09.src.classes import FCNNClassifier
from units.unit_09.src.functions import plot_confusion_matrix

# Set random seeds for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

df = pd.read_csv(os.path.join("datasets", "dataset_wine.csv"))

features = df.columns[1:]
target = df.columns[0]

x = df.loc[:, features].to_numpy()
y = df.loc[:, target].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=1)

model = FCNNClassifier(hidden_layers=(4,), random_state=0)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
y_pred_proba = model.predict_proba(x_test)
class_names = model.get_class_names()

acc = accuracy_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

print(f"Fully Connected Neural Network | Accuracy  (Test): {acc:.1%} | MCC (Test): {mcc:.3f}")

plot_confusion_matrix(
    y_true=y_test,
    y_pred=model.predict(x_test),
    title=f"Wine identification using:\nFully Connected Neural Network\nAccuracy: {acc:.1%} | MCC: {mcc:.3f}",
    filepath=os.path.join("figures", f"figure_03.png")
)

columns = ["y_test", "y_pred"] + list(class_names)

records = []
for i in range(y_test.shape[0]):
    record = [y_test[i], y_pred[i]]
    for j in range(y_pred_proba.shape[1]):
        record.append(f"{y_pred_proba[i, j]:6.1%}")

    records.append(tuple(record))

df = pd.DataFrame.from_records(records, columns=columns)

df.to_csv(os.path.join("tables", "table_03.csv"), index=False)
