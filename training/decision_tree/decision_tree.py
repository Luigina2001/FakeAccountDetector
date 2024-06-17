from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from training.utils import *

model_filename = 'dt_model.pkl'
dataset_path = '../../dataset_normalizzato/user_fake_authentic_2class_cleaned.csv'

def train_model(X_train, y_train):
    param_grid = {
        'max_depth': [None, 10, 20, 30], # Profondità massima dell'albero
        'min_samples_split': [2, 5, 10], # Numero minimo di campioni richiesti per suddividere un nodo
        'min_samples_leaf': [1, 2, 4] # Numero minimo di campioni richiesti per essere in un nodo foglia
    }
    dt = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def main():
    X_train, X_test, y_train, y_test = load_data(dataset_path)
    model = load_model(model_filename)
    if model is None:
        model = train_model(X_train, y_train)
        save_model(model, model_filename)
    evaluate_model(model, X_test, y_test)
    plot_and_save_confusion_matrix(model, X_test, y_test, 'dt')

if __name__ == "__main__":
    main()