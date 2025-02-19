import pandas as pd

pd.set_option('display.max_columns', None)  # per mostrare tutte le colonne

general = "../../datasets/instagram2/user_fake_authentic_4class.csv"
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

colonna = "class"
valori_unici = ig_users[colonna].unique()  # calcolo i valori unici dalla colonna
print(f"Valori unici colonna {colonna}: {valori_unici}")

num_active_fake = ig_users[ig_users['class'] == 'a'].shape[0]
num_inactive_fake = ig_users[ig_users['class'] == 'i'].shape[0]
num_spammer_fake = ig_users[ig_users['class'] == 's'].shape[0]
num_real = ig_users[ig_users['class'] == 'r'].shape[0]
print(f"Numero di account fake attivi: {num_active_fake}")
print(f"Numero di account fake inattivi: {num_inactive_fake}")
print(f"Numero di account spammer fake: {num_spammer_fake}")
print(f"Numero di account real: {num_real}")

print("---------------------------------------------------------------------------------------------------------")
