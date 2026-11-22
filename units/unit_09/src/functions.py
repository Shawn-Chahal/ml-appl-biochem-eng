from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


def generate_model_dictionary(metric, gscv_folds):
    model_dict = {
        "Logistic Regression": GridSearchCV(
            estimator=Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(random_state=0))
            ]),
            param_grid=dict(
                clf__C=[10 ** i for i in range(-4, 5)]
            ),
            scoring=metric,
            cv=gscv_folds
        ),
        "Support Vector Machine": GridSearchCV(
            estimator=Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LinearSVC(random_state=0))
            ]),
            param_grid=dict(
                clf__C=[10 ** i for i in range(-4, 5)],
            ),
            scoring=metric,
            cv=gscv_folds
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
            scoring=metric,
            cv=gscv_folds
        ),
        "Gaussian Naive Bayes ": GridSearchCV(
            estimator=Pipeline([
                ('scaler', StandardScaler()),
                ('clf', GaussianNB())
            ]),
            param_grid=dict(

            ),
            scoring=metric,
            cv=gscv_folds
        ),
        "Decision Tree": GridSearchCV(
            estimator=Pipeline([
                ('scaler', StandardScaler()),
                ('clf', DecisionTreeClassifier(random_state=0))
            ]),
            param_grid=dict(
                clf__max_depth=[i for i in range(1, 11)] + [None]
            ),
            scoring=metric,
            cv=gscv_folds
        ),
        "Random Forest": GridSearchCV(
            estimator=Pipeline([
                ('scaler', StandardScaler()),
                ('clf', RandomForestClassifier(random_state=0))
            ]),
            param_grid=dict(
                clf__n_estimators=[2 ** i for i in range(1, 10)]
            ),
            scoring=metric,
            cv=gscv_folds
        ),
    }

    return model_dict
