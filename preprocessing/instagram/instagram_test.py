import pandas as pd

pd.set_option('display.max_columns', None)  # per mostrare tutte le colonne

general = "../../datasets/instagram/test.csv"
ig_users = pd.read_csv(general, sep=',')
# print(dataset.head())

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

num_real = ig_users[ig_users['fake'] == 0].shape[0]
num_fake = ig_users[ig_users['fake'] == 1].shape[0]
print(f"Numero di account fake: {num_fake}")
print(f"Numero di account real: {num_real}")

print("---------------------------------------------------------------------------------------------------------")
