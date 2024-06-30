from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix)
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
        print("Modello salvato trovato. Caricamento del modello...\n")
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
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)
    precision = report['macro avg']['precision']
    recall = report['macro avg']['recall']
    f1 = report['macro avg']['f1-score']

    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    print(f'Accuracy: {round(accuracy, 2)}')
    print(f'Precision: {round(precision, 2)}')
    print(f'Recall: {round(recall, 2)}')
    print(f'F1 Score: {round(f1, 2)}\n')

    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))

    # salvataggio delle metriche in un file csv
    with open('metrics.csv', 'w') as f:
        f.write('accuracy, precision, recall, f1-score\n')
        f.write(f'{accuracy}, {precision}, {recall}, {f1}')

def plot_and_save_confusion_matrix(model, X_test, y_test, type_model):
    model_names = {'rf': 'Random Forest', 'dt': 'Decision Tree', 'svm': 'Support Vector Machine', 'knn': 'K-Nearest Neighbors'}
    full_model_name = model_names[type_model]

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           ylabel='True label',
           xlabel='Predicted label')

    ax.set_title(f'Confusion Matrix for {full_model_name} Model')

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2. else "black")

    plt.savefig(f'{type_model}_confusion_matrix.png')




