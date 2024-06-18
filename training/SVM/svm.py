from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from training.utils import *

model_filename = 'svm_model.pkl'
dataset_path = '../../dataset_normalizzato/user_fake_authentic_2class_cleaned.csv'


def train_model(train, y_train):
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000], # Parametro di regolarizzazione
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001], # Coefficiente del kernel
        'kernel': ['rbf'] # Tipo di kernel
    }
    svm = SVC()
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, refit=True, verbose=3)
    grid_search.fit(train, y_train)
    return grid_search.best_estimator_


def main():
    train, test, y_train, y_test = load_data(dataset_path)
    model = load_model(model_filename)
    if model is None:
        model = train_model(train, y_train)
        save_model(model, model_filename)
    evaluate_model(model, test, y_test)
    plot_and_save_confusion_matrix(model, test, y_test, 'svm')


if __name__ == "__main__":
    main()
