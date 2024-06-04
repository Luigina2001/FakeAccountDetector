import pandas as pd

pd.set_option('display.max_columns', None)  # per mostrare tutte le colonne

general = "../../datasets/weibo/fusers.csv"
fake_users = pd.read_csv(general, sep=',')
# print(dataset.head())

print("---------------------------------------------------------------------------------------------------------")

nomi_campi = fake_users.columns
print(f"Nomi dei campi presenti nel dataset {nomi_campi}")

print("---------------------------------------------------------------------------------------------------------")

numero_campi = fake_users.shape[1]
print(f"Numero di campi presenti nel dataset: {numero_campi}")

print("---------------------------------------------------------------------------------------------------------")

lunghezza_dataset = fake_users.shape[0]
print(f"Lunghezza del dataset (numero di righe): {lunghezza_dataset}")

print("---------------------------------------------------------------------------------------------------------")

print("INFO")
fake_users.info()

print("---------------------------------------------------------------------------------------------------------")

statistiche = fake_users.describe(include='all')
print(f"STATISTICHE:\n{statistiche}")

print("---------------------------------------------------------------------------------------------------------")

colonna = "url"
valori_unici = fake_users[colonna].unique()  # calcolo i valori unici dalla colonna
print(f"Valori unici colonna {colonna}: {len(valori_unici)}")

print("---------------------------------------------------------------------------------------------------------")
