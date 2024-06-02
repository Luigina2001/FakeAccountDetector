import pandas as pd

pd.set_option('display.max_columns', None)  # per mostrare tutte le colonne

general = "../datasets/facebook.csv"
fb_users = pd.read_csv(general, sep=',')
# 0 = real, 1 = fake

print("---------------------------------------------------------------------------------------------------------")

nomi_campi = fb_users.columns
print(f"Nomi dei campi presenti nel dataset {nomi_campi}")

print("---------------------------------------------------------------------------------------------------------")

numero_campi = fb_users.shape[1]
print(f"Numero di campi presenti nel dataset: {numero_campi}")

print("---------------------------------------------------------------------------------------------------------")

lunghezza_dataset = fb_users.shape[0]
print(f"Lunghezza del dataset (numero di righe): {lunghezza_dataset}")

print("---------------------------------------------------------------------------------------------------------")

print("INFO")
fb_users.info()

print("---------------------------------------------------------------------------------------------------------")

statistiche = fb_users.describe(include='all')
print(f"STATISTICHE:\n{statistiche}")

print("---------------------------------------------------------------------------------------------------------")

colonna = "Label"
valori_unici = fb_users[colonna].unique()  # calcolo i valori unici dalla colonna
print(f"Valori unici colonna {colonna}: {valori_unici}")

num_fake = fb_users[fb_users['Label'] == 1].shape[0]
num_real = fb_users[fb_users['Label'] == 0].shape[0]
print(f"Numero di account fake: {num_fake}")
print(f"Numero di account real: {num_real}")

print("---------------------------------------------------------------------------------------------------------")