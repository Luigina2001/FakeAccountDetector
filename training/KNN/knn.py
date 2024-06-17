from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from training.utils import *

model_filename = 'knn_model.pkl'
dataset_path = '../../dataset_normalizzato/user_fake_authentic_2class_cleaned.csv'


def train_model(train, y_train):
    param_grid = {
        'n_neighbors': [3, 5, 11],  # Numero di vicini
        'weights': ['uniform', 'distance'],  # Funzione peso utilizzata nella previsione
        'metric': ['minkowski', 'euclidean', 'cosine', 'jaccard']  # Metrica da utilizzare per il calcolo della distanza
    }
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(train, y_train)
    return grid_search.best_estimator_


def main():
    train, test, y_train, y_test = load_data(dataset_path)
    model = load_model(model_filename)
    if model is None:
        model = train_model(train, y_train)
        save_model(model, model_filename)
    evaluate_model(model, test, y_test)
    plot_and_save_confusion_matrix(model, test, y_test, 'knn')


if __name__ == "__main__":
    main()
