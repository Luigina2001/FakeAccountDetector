from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import pandas as pd
from training.utils import *

model_filename = 'knn_model.pkl'
dataset_path = '../../dataset_normalizzato/user_fake_authentic_2class_cleaned.csv'


def train_model(train, y_train):
    param_grid = {
        'n_neighbors': [3, 5, 11],  # Numero di vicini
        'weights': ['uniform', 'distance'],  # Funzione peso utilizzata nella previsione
        'metric': ['minkowski', 'euclidean', 'cosine']  # Metrica da utilizzare per il calcolo della distanza
    }
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(train, y_train)
    return grid_search.best_estimator_


def plot_roc_curve(model, test, y_test):
    # Calcolo delle probabilità previste dal modello
    y_prob = model.predict_proba(test)[:, 1]

    # Calcolo dei valori di FPR e TPR per diverse soglie
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Plot della ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasso di Falsi Positivi')
    plt.ylabel('Tasso di Veri Positivi')
    plt.title('Curva ROC (Receiver Operating Characteristic)')
    plt.legend(loc='lower right')

    # Salvataggio della ROC curve come immagine
    plt.savefig('roc_curve.png')
    plt.show()


def load_data(dataset_path, label_column='class'):
    # Carica il dataset
    data = pd.read_csv(dataset_path)

    # Estrai le caratteristiche (features) e l'etichetta (label)
    X = data.drop(columns=[label_column])
    y = data[label_column]

    # Dividi il dataset in train e test set
    train, test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return train, test, y_train, y_test


def main():
    label_column = 'class'
    train, test, y_train, y_test = load_data(dataset_path, label_column=label_column)
    # train, test, y_train, y_test = load_data(dataset_path)

    # Stampa delle classi univoche presenti nel dataset
    print(f"Classi presenti nel dataset di addestramento: {set(y_train)}")
    print(f"Classi presenti nel dataset di test: {set(y_test)}")

    model = load_model(model_filename)
    if model is None:
        model = train_model(train, y_train)
        save_model(model, model_filename)
    evaluate_model(model, test, y_test)
    plot_and_save_confusion_matrix(model, test, y_test, 'knn')

    # Plot della ROC curve
    plot_roc_curve(model, test, y_test)


if __name__ == "__main__":
    main()
