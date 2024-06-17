from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

def load_data(dataset_path):
    df = pd.read_csv(dataset_path)
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

def save_model(model, model_filename):
    try:
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        print("Modello salvato")
    except Exception as e:
        print(f"Errore nel salvataggio del modello: {e}")

def evaluate_model(model, X_test, y_test):
    y_pred_rf = model.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    print(f'Accuracy: {accuracy_rf}')
    print('Classification Report:')
    print(classification_report(y_test, y_pred_rf))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred_rf))

def plot_and_save_confusion_matrix(model, X_test, y_test, type_model):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           ylabel='True label',
           xlabel='Predicted label')

    ax.set_title('Confusion Matrix for Random Forest Model')

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2. else "black")

    plt.savefig(f'{type_model}_confusion_matrix.png')




