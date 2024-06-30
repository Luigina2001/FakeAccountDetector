import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def load_data(file_path, delimiter=','):
    """Carica i dati da un file CSV."""
    return pd.read_csv(file_path, delimiter=delimiter)

def rename_columns(data, new_column_names):
    """Rinomina le colonne di un DataFrame."""
    data.rename(columns=new_column_names, inplace=True)

def add_columns(data, new_columns):
    """Aggiungi nuove colonne a un DataFrame."""
    for column, value in new_columns.items():
        data[column] = value

def combine_data(data_list):
    """Combina più DataFrame in uno solo."""
    return pd.concat(data_list)

def plot_comparison(x, y, hue, data, title, ylabel, xlabel, filename):
    """Crea un grafico a barre e salvalo come file PNG."""
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.barplot(x=x, y=y, hue=hue, data=data, ax=ax, palette="viridis", errorbar=None)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
    fig.tight_layout()
    fig.savefig(filename)

def main():
    # Percorsi dei file
    file_paths = [
        '../training/classificazione_con_RFD/Results_instagram_bilanciato_tipo2.csv',
        '../training/decision_tree/metrics.csv',
        '../training/random_forest/metrics.csv',
        '../training/KNN/metrics.csv',
        '../training/SVM/metrics.csv'
    ]

    # Caricamento dei dati
    data = [load_data(file_path, delimiter=';') if i == 0 else load_data(file_path) for i, file_path in enumerate(file_paths)]

    # Ridenominazione delle celle della colonna 'model' da 'SVC' a 'SVM' nel primo DataFrame
    data[0]['model'] = data[0]['model'].replace('SVC', 'SVM')

    # Ridenominazione delle colonne
    rename_columns(data[0], {'macro avg.recall': 'recall', 'macro avg.precision': 'precision', 'macro avg.f1-score': 'f1-score'})

    # Aggiunta di una colonna per distinguere i dataset
    add_columns(data[0], {'RFD': 'With RFD'})
    for i in range(1, len(data)):
        add_columns(data[i], {'RFD': 'Without RFD', 'model': ['DT', 'RF', 'KNN', 'SVM'][i-1]})

    # Unione dei dati in un unico DataFrame
    data_combined = combine_data(data)

    # Creazione dei grafici
    plot_comparison('model', 'accuracy', 'RFD', data_combined, 'Confronto dell\'Accuratezza tra Modelli con e senza RFD', 'Accuratezza', 'Modello', 'accuracy_comparison.png')
    plot_comparison('model', 'precision', 'RFD', data_combined, 'Confronto della Precisione tra Modelli con e senza RFD', 'Precisione', 'Modello', 'precision_comparison.png')
    plot_comparison('model', 'recall', 'RFD', data_combined, 'Confronto della Recall tra Modelli con e senza RFD', 'Recall', 'Modello', 'recall_comparison.png')
    plot_comparison('model', 'f1-score', 'RFD', data_combined, 'Confronto dell\'F1-Score tra Modelli con e senza RFD', 'F1-Score', 'Modello', 'f1_comparison.png')

if __name__ == "__main__":
    main()
