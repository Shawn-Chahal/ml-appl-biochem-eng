import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


def generate_model_dictionary(cv=4, scoring="accuracy"):
    model_dict = {
        "Logistic Regression": GridSearchCV(
            estimator=Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(random_state=0))
            ]),
            param_grid=dict(
                clf__C=[10 ** i for i in range(-4, 5)]
            ),
            scoring=scoring,
            cv=cv
        ),
        "Support Vector Machine": GridSearchCV(
            estimator=Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LinearSVC(random_state=0))
            ]),
            param_grid=dict(
                clf__C=[10 ** i for i in range(-4, 5)],
            ),
            scoring=scoring,
            cv=cv
        ),
        "K-Nearest Neighbors": GridSearchCV(
            estimator=Pipeline([
                ('scaler', StandardScaler()),
                ('clf', KNeighborsClassifier())
            ]),
            param_grid=dict(
                clf__n_neighbors=[i for i in range(1, 21)],
                clf__weights=["uniform", "distance"]
            ),
            scoring=scoring,
            cv=cv
        ),
        "Gaussian Naive Bayes ": GridSearchCV(
            estimator=Pipeline([
                ('scaler', StandardScaler()),
                ('clf', GaussianNB())
            ]),
            param_grid=dict(),
            scoring=scoring,
            cv=cv
        ),
        "Decision Tree": GridSearchCV(
            estimator=Pipeline([
                ('scaler', StandardScaler()),
                ('clf', DecisionTreeClassifier(random_state=0))
            ]),
            param_grid=dict(
                clf__max_depth=[i for i in range(1, 11)] + [None]
            ),
            scoring=scoring,
            cv=cv
        ),
        "Random Forest": GridSearchCV(
            estimator=Pipeline([
                ('scaler', StandardScaler()),
                ('clf', RandomForestClassifier(random_state=0))
            ]),
            param_grid=dict(
                clf__n_estimators=[2 ** i for i in range(1, 10)]
            ),
            scoring=scoring,
            cv=cv
        ),
    }

    return model_dict


def plot_confusion_matrix(y_true, y_pred, title, filepath):
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray"]

    u_labels = np.unique(y_true)

    if "Positive" in u_labels:
        r_labels = ["Positive"]
        for label in u_labels:
            if label != "Positive":
                r_labels.append(label)

        u_labels = np.array(r_labels)

    label_true = np.copy(u_labels)
    label_pred = np.copy(u_labels)

    idx_true = {label: idx for idx, label in enumerate(label_true)}
    idx_pred = {label: idx for idx, label in enumerate(label_pred)}

    confusion_matrix = np.zeros((label_true.size, label_pred.size))

    for y_t, y_p in zip(y_true, y_pred):
        idx_t = idx_true[y_t]
        idx_p = idx_pred[y_p]
        confusion_matrix[idx_t, idx_p] += 1

    fontsize_label = 12
    width = 1

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14.39 / 2.54, 12.09 / 2.54), dpi=300, layout="constrained")

    for i in range(confusion_matrix.shape[0]):
        j = -1

        x = width * j
        y = - width * i

        x_text = x + width / 2
        y_text = y - width / 2

        ax.add_patch(Rectangle(xy=(x, y), width=width, height=-width, edgecolor="black", facecolor=colors[i]))
        ax.text(x_text, y_text, f"{label_true[i]}", color="black", ha="center", va="center", fontsize=fontsize_label)

        j = confusion_matrix.shape[1]

        x = width * j
        y = - width * i

        x_text = x + width / 2
        y_text = y - width / 2

        frac_row = confusion_matrix[i, i] / np.sum(confusion_matrix[i, :])

        color_r = 1 - frac_row
        color_g = 1 - frac_row
        color_b = 1
        color = np.array([color_r, color_g, color_b])
        color = color ** 0.5

        ax.add_patch(Rectangle(xy=(x, y), width=width, height=-width, edgecolor="black", facecolor=color))

        if color[0] > 0.5:
            color = "black"
        else:
            color = "white"

        ax.text(x_text, y_text, f"{frac_row:.1%}", color=color, ha="center", va="center", fontsize=fontsize_label)

    for j in range(confusion_matrix.shape[1]):
        i = -1

        x = width * j
        y = - width * i

        x_text = x + width / 2
        y_text = y - width / 2

        ax.add_patch(Rectangle(xy=(x, y), width=width, height=-width, edgecolor="black", facecolor=colors[j]))
        ax.text(x_text, y_text, f"{label_pred[j]}", color="black", ha="center", va="center", fontsize=fontsize_label)

        i = confusion_matrix.shape[0]

        x = width * j
        y = - width * i

        x_text = x + width / 2
        y_text = y - width / 2

        if j < confusion_matrix.shape[0]:
            frac_col = confusion_matrix[j, j] / np.sum(confusion_matrix[:, j])

            color_r = 1 - frac_col
            color_g = 1 - frac_col
            color_b = 1
            color = np.array([color_r, color_g, color_b])
            color = color ** 0.5

            ax.add_patch(Rectangle(xy=(x, y), width=width, height=-width, edgecolor="black", facecolor=color))

            if color[0] > 0.5:
                color = "black"
            else:
                color = "white"

            ax.text(x_text, y_text, f"{frac_col:.1%}", color=color, ha="center", va="center", fontsize=fontsize_label)

    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            x = width * j
            y = - width * i

            x_text = x + width / 2
            y_text = y - width / 2

            frac_row = confusion_matrix[i, j] / np.sum(confusion_matrix[i, :])

            if i == j:
                color_r = 1 - frac_row
                color_g = 1
                color_b = 1 - frac_row
                color = np.array([color_r, color_g, color_b])
                color = color ** 0.5

            else:
                color_r = 1
                color_g = 1 - frac_row
                color_b = 1 - frac_row
                color = np.array([color_r, color_g, color_b])
                color = color ** 2

            ax.add_patch(Rectangle(xy=(x, y), width=width, height=-width, edgecolor="black", facecolor=color))
            ax.text(x_text, y_text, f"{confusion_matrix[i, j]:.0f}", color="black", ha="center", va="center",
                    fontsize=12)

    ax.set_xlim(left=-width, right=width * (confusion_matrix.shape[1] + 1))
    ax.set_ylim(bottom=-width * (confusion_matrix.shape[0] + 1), top=width)

    ax.set_xticks([], [])
    ax.set_yticks([], [])

    ax.xaxis.set_label_position('top')
    ax.set_xlabel("Predicted Value", fontsize=14)
    ax.set_ylabel("Actual Value", fontsize=14)

    ax.set_title(f"{title}\n", fontsize=14)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    fig.savefig(filepath)
    plt.close(fig)
