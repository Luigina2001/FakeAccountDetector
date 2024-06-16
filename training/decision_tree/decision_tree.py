import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

model_filename = 'dt_model.pkl'
def load_data():
    df = pd.read_csv('../../dataset_normalizzato/user_fake_authentic_2class_cleaned.csv')
    df['class'] = df['class'].map({'r': 0, 'f': 1})
    X = df.drop('class', axis=1) # Variabili indipendenti
    y = df['class'] # Variabile dipendente
    return train_test_split(X, y, test_size=0.2, random_state=42)

def load_model(model_filename):
    if os.path.exists(model_filename):
        print("Modello salvato trovato. Caricamento del modello...")
        with open(model_filename, 'rb') as f:
            return pickle.load(f)
    return None

def train_model(X_train, y_train):
    param_grid = {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    dt = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def save_model(model, model_filename):
    try:
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        print("Modello salvato")
    except Exception as e:
        print(f"Errore nel salvataggio del modello: {e}")

def evaluate_model(model, X_test, y_test):
    y_pred_dt = model.predict(X_test)
    accuracy_dt = accuracy_score(y_test, y_pred_dt)
    print(f'Accuracy: {accuracy_dt}')
    print('Classification Report:')
    print(classification_report(y_test, y_pred_dt))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred_dt))

def plot_and_save_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           ylabel='True label',
           xlabel='Predicted label')

    ax.set_title('Confusion Matrix for Decision Tree Model')

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2. else "black")

    plt.savefig('dt_confusion_matrix.png')

def main():
    X_train, X_test, y_train, y_test = load_data()
    model = load_model(model_filename)
    if model is None:
        model = train_model(X_train, y_train)
        save_model(model, model_filename)
    evaluate_model(model, X_test, y_test)
    plot_and_save_confusion_matrix(model, X_test, y_test)

if __name__ == "__main__":
    main()