import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV


def draw_confusion(df_cm, title, filename):
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt=".0f")
    plt.title(title)
    plt.savefig(f"{filename}-confusion.pdf")
    plt.cla()
    plt.clf()


def early_stopping_fit(model, x_train, y_train, x_val, y_val, patience=5):
    best_score = 0
    no_improvement_count = 0
    best_model = model
    for i in range(1, 1001):
        model.fit(x_train, y_train)
        y_val_pred = model.predict(x_val)
        score = accuracy_score(y_val, y_val_pred)
        if score > best_score:
            best_score = score
            best_model = model
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                break
    return best_model


def fit_and_evaluate(model, x_train, y_train, x_test, y_test, patience=5):
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    best_model = early_stopping_fit(model, x_train, y_train, x_val, y_val, patience=patience)
    y_pred = best_model.predict(x_test)
    accuracy_value = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(conf_matrix)
    return accuracy_value, report, df_cm


def grid_search_rf(x_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [10, 20, None],
        'max_features': [1, 2, 3],
        'min_samples_split': [2, 10],
        'min_samples_leaf': [1, 2]
    }
    rfc = RandomForestClassifier(random_state=42, bootstrap=True, criterion='entropy')
    grid_search = GridSearchCV(rfc, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(x_train, y_train)
    print("\nBest parameters for Random Forest:", grid_search.best_params_)
    return grid_search.best_estimator_


def grid_search_dt(x_train, y_train):
    param_grid = {
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 10],
        'min_samples_leaf': [1, 2],
        'max_leaf_nodes': [None, 3, 10]
    }
    dt = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(dt, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(x_train, y_train)
    print("\nBest parameters for Decision Tree:", grid_search.best_params_)
    return grid_search.best_estimator_


def grid_search_svc(x_train, y_train):
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'kernel': ['rbf']
    }
    svc = SVC()
    grid_search = GridSearchCV(svc, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(x_train, y_train)
    print("\nBest parameters for SVC:", grid_search.best_params_)
    return grid_search.best_estimator_


def grid_search_knn(x_train, y_train):
    param_grid = {
        'n_neighbors': [5, 11, 21],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(x_train, y_train)
    print("\nBest parameters for KNN:", grid_search.best_params_)
    return grid_search.best_estimator_


def RF_prediction(x_train, y_train, x_test, y_test, comparison=4):
    best_rfc = grid_search_rf(x_train, y_train)
    return fit_and_evaluate(best_rfc, x_train, y_train, x_test, y_test)


def DT_prediction(x_train, y_train, x_test, y_test):
    best_dt = grid_search_dt(x_train, y_train)
    return fit_and_evaluate(best_dt, x_train, y_train, x_test, y_test)


def SVC_prediction(x_train, y_train, x_test, y_test):
    best_svc = grid_search_svc(x_train, y_train)
    return fit_and_evaluate(best_svc, x_train, y_train, x_test, y_test)


def knn_prediction(x_train, y_train, x_test, y_test):
    best_knn = grid_search_knn(x_train, y_train)
    return fit_and_evaluate(best_knn, x_train, y_train, x_test, y_test)
