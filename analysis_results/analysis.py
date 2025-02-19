import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def load_data(file_path, delimiter=','):
    """Carica i dati da un file CSV."""
    return pd.read_csv(file_path, delimiter=delimiter)

def rename_columns(data, new_column_names):
    """Rinomina le colonne di un DataFrame."""
    data.rename(columns=new_column_names, inplace=True)

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
    file_paths = {
        'With RFD': '../training/classificazione_con_RFD/Results_instagram_bilanciato_tipo2.csv',
        'DT': '../training/decision_tree/metrics.csv',
        'RF': '../training/random_forest/metrics.csv',
        'KNN': '../training/KNN/metrics.csv',
        'SVM': '../training/SVM/metrics.csv'
    }

    with open(file_paths['With RFD'], 'r') as f:
        content = f.read()
        content = content.replace(';', ',')
    with open(file_paths['With RFD'], 'w') as f:
        f.write(content)

    # Caricamento e preparazione dei dati
    data_with_rfd = load_data(file_paths['With RFD'], delimiter=',')
    data_with_rfd['model'] = data_with_rfd['model'].replace('SVC', 'SVM')
    rename_columns(data_with_rfd, {'macro avg.recall': 'recall', 'macro avg.precision': 'precision', 'macro avg.f1-score': 'f1-score'})
    data_with_rfd['RFD'] = 'With RFD'

    # Preparazione dei dati senza RFD
    data_without_rfd_list = []
    for model, file_path in file_paths.items():
        if model == 'With RFD':
            continue
        data = load_data(file_path)
        data['RFD'] = 'Without RFD'
        data['model'] = model
        data_without_rfd_list.append(data)

    # Unione dei dati senza RFD in un unico DataFrame
    data_without_rfd = pd.concat(data_without_rfd_list, ignore_index=True)

    # Creazione dei grafici
    plot_comparison('model', 'accuracy', 'RFD', pd.concat([data_with_rfd, data_without_rfd]),
                    'Confronto dell\'Accuratezza tra Modelli con e senza RFD', 'Accuratezza', 'Modello', 'accuracy_comparison.png')
    plot_comparison('model', 'precision', 'RFD', pd.concat([data_with_rfd, data_without_rfd]),
                    'Confronto della Precisione tra Modelli con e senza RFD', 'Precisione', 'Modello', 'precision_comparison.png')
    plot_comparison('model', 'recall', 'RFD', pd.concat([data_with_rfd, data_without_rfd]),
                    'Confronto della Recall tra Modelli con e senza RFD', 'Recall', 'Modello', 'recall_comparison.png')
    plot_comparison('model', 'f1-score', 'RFD', pd.concat([data_with_rfd, data_without_rfd]),
                    'Confronto dell\'F1-Score tra Modelli con e senza RFD', 'F1-Score', 'Modello', 'f1_comparison.png')

    #
    data_with_rfd_rf_4 = data_with_rfd[(data_with_rfd['dataset'].str.contains('threshold_0_EXTENDED_0.4')) & (data_with_rfd['model'] == 'RF')]
    data_with_rfd_dt_4 = data_with_rfd[(data_with_rfd['dataset'].str.contains('threshold_0_EXTENDED_0.4')) & (data_with_rfd['model'] == 'DT')]

    data_with_rfd_rf_5 = data_with_rfd[(data_with_rfd['dataset'].str.contains('threshold_0_EXTENDED_0.5')) & (data_with_rfd['model'] == 'RF')]
    data_with_rfd_dt_5 = data_with_rfd[(data_with_rfd['dataset'].str.contains('threshold_0_EXTENDED_0.5')) & (data_with_rfd['model'] == 'DT')]

    # Creazione dei grafici
    df_rf4 = pd.concat([data_with_rfd_rf_4, data_without_rfd[data_without_rfd['model'] == 'RF']])
    df_dt4 = pd.concat([data_with_rfd_dt_4, data_without_rfd[data_without_rfd['model'] == 'DT']])

    df_rf5 = pd.concat([data_with_rfd_rf_5, data_without_rfd[data_without_rfd['model'] == 'RF']])
    df_dt5 = pd.concat([data_with_rfd_dt_5, data_without_rfd[data_without_rfd['model'] == 'DT']])

    df_final4 = pd.concat([df_rf4, df_dt4])
    plot_comparison('model', 'accuracy', 'RFD', df_final4,
                    'Confronto dell\'Accuratezza tra Random Forest e Decision Tree \ln con e senza RFD (Threshold 0 - Extend 0.4)', 'Accuratezza', 'Modello', 'accuracy_comparison_thr0_ext4.png')

    plot_comparison('model', 'precision', 'RFD', df_final4,
                    'Confronto della Precisione tra Random Forest e Decision Tree con e senza RFD (Threshold 0 - Extend 0.4)', 'Precisione', 'Modello', 'precision_comparison_thr0_ext4.png')

    plot_comparison('model', 'recall', 'RFD', df_final4,
                    'Confronto della Recall tra Random Forest e Decision Tree con e senza RFD (Threshold 0 - Extend 0.4)', 'Recall', 'Modello', 'recall_comparison_thr0_ext4.png')

    plot_comparison('model', 'f1-score', 'RFD', df_final4,
                    'Confronto dell\'F1-Score tra Random Forest e Decision Tree con e senza RFD (Threshold 0 - Extend 0.4)', 'F1-Score', 'Modello', 'f1_comparison_thr0_ext4.png')

    df_final5 = pd.concat([df_rf5, df_dt5])
    plot_comparison('model', 'accuracy', 'RFD', df_final5,
                    'Confronto dell\'Accuratezza tra Random Forest e Decision Tree con e senza RFD (Threshold 0 - Extend 0.5)', 'Accuratezza', 'Modello', 'accuracy_comparison_thr0_ext5.png')

    plot_comparison('model', 'precision', 'RFD', df_final5,
                    'Confronto della Precisione tra Random Forest e Decision Tree con e senza RFD (Threshold 0 - Extend 0.5)', 'Precisione', 'Modello', 'precision_comparison_thr0_ext5.png')

    plot_comparison('model', 'recall', 'RFD', df_final5,
                    'Confronto della Recall tra Random Forest e Decision Tree con e senza RFD (Threshold 0 - Extend 0.5)', 'Recall', 'Modello', 'recall_comparison_thr0_ext5.png')

    plot_comparison('model', 'f1-score', 'RFD', df_final5,
                    'Confronto dell\'F1-Score tra Random Forest e Decision Tree con e senza RFD (Threshold 0 - Extend 0.5)', 'F1-Score', 'Modello', 'f1_comparison_thr0_ext5.png')


if __name__ == "__main__":
    main()
