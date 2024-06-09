import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', None)  # per mostrare tutte le colonne

general = "../../datasets/instagram2/user_fake_authentic_2class.csv"
ig_users = pd.read_csv(general, sep=',')


# print(dataset.head())

def disegna_grafico(data, columns, algoritmo):
    # figura e set di assi
    fig, ax = plt.subplots()

    # boxplot
    boxplot = ax.boxplot(data[columns].values, widths=0.5, patch_artist=True, showfliers=False)

    # modifiche grafiche
    colors = ['lightblue', 'lightgreen', 'lightpink']
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)
    for whisker in boxplot['whiskers']:
        whisker.set(color='gray', linewidth=0.5)
    for cap in boxplot['caps']:
        cap.set(color='gray', linewidth=0.5)
    for median in boxplot['medians']:
        median.set(color='red', linewidth=0.5)

    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle='dashed', linewidth=0.5, alpha=0.7)

    # etichette assi
    ax.set_xticklabels(columns)
    ax.set_ylabel('Valore')
    ax.set_title('Boxplot dopo ' + algoritmo + ' normalization')

    # dimensione della figura per evitare sovrapposizioni
    fig.tight_layout()
    plt.show()


print("---------------------------------------------------------------------------------------------------------")

nomi_campi = ig_users.columns
print(f"Nomi dei campi presenti nel dataset {nomi_campi}")

print("---------------------------------------------------------------------------------------------------------")

numero_campi = ig_users.shape[1]
print(f"Numero di campi presenti nel dataset: {numero_campi}")

print("---------------------------------------------------------------------------------------------------------")

lunghezza_dataset = ig_users.shape[0]
print(f"Lunghezza del dataset (numero di righe): {lunghezza_dataset}")

print("---------------------------------------------------------------------------------------------------------")

print("INFO")
ig_users.info()

print("---------------------------------------------------------------------------------------------------------")

statistiche = ig_users.describe(include='all')
print(f"STATISTICHE:\n{statistiche}")

print("---------------------------------------------------------------------------------------------------------")

colonna = "class"
valori_unici = ig_users[colonna].unique()  # calcolo i valori unici dalla colonna
print(f"Valori unici colonna {colonna}: {valori_unici}")

num_fake = ig_users[ig_users['class'] == 'f'].shape[0]
num_real = ig_users[ig_users['class'] == 'r'].shape[0]
print(f"Numero di account fake: {num_fake}")
print(f"Numero di account real: {num_real}")

print("---------------------------------------------------------------------------------------------------------")

# ------------------------------- DATA CLEANING -------------------------------

# calcolo i valori mancanti per colonna
missing_values = ig_users.isnull().sum()
print(missing_values)

# calcolo il numero totale di valori mancanti nel dataset
total_missing = ig_users.isnull().sum().sum()
print("Totale valori mancanti prima del data cleaning:", total_missing)

if total_missing == 0:
    print("Nessun valore risulta nullo all'interno del dataset")

# ------------------------------- FEATURE SCALING -------------------------------

# per ogni colonna della tabella mostra le statistiche (per verificare valori min e max di ogni campo della tabella)
for col in ig_users:
    print(col, "\n", ig_users[col].describe(), "\n\n")

numeric_columns = ig_users.select_dtypes(include=[np.number]).columns

# [1] Plot prima del feature scaling
plt.figure(figsize=(10, 6))
for col in numeric_columns:
    values = ig_users[col].values     # estraggo i valori della colonna
    plt.plot(values, label=col)      # creo un grafico a linea dei valori

plt.ylim(0, 70000)
plt.legend()
plt.title('Grafico a linee prima del feature scaling')
plt.show()

# Z-score normalization
scaler = StandardScaler()
ig_users[numeric_columns] = scaler.fit_transform(ig_users[numeric_columns])

# [2] Boxplot dopo z-score normalization
disegna_grafico(ig_users, numeric_columns, "z-score")

# ------------------------------- FEATURE SELECTION -------------------------------

non_numeric_columns = ig_users.select_dtypes(include=['object', 'bool']).columns

# oggetto VarianceThreshold con soglia di varianza zero
selector = VarianceThreshold(threshold=0)
# applico la selezione delle feature alle sole colonne numeriche del dataset
dataset_selected = selector.fit_transform(ig_users[numeric_columns])

# indici delle feature selezionate
feature_indices = selector.get_support(indices=True)
# nomi delle feature selezionate
selected_features = [feature for i, feature in enumerate(numeric_columns) if i in feature_indices]

# nomi delle feature selezionate
print("Feature selezionate:")
for feature in selected_features:
    print(feature)

# creo un nuovo dataset con le sole feature selezionate
dataset_new = ig_users[selected_features].copy()

# aggiungo le colonne non numeriche al nuovo DataFrame
for column in non_numeric_columns:
    dataset_new[column] = ig_users[column]

dataset = dataset_new
print(dataset.head())

dataset.to_csv("../../dataset_normalizzato/user_fake_authentic_2class_cleaned.csv", index=False)
