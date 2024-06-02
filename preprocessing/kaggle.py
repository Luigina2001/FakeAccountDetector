import pandas as pd

pd.set_option('display.max_columns', None)  # per mostrare tutte le colonne

general = "../datasets/kaggle.csv"
kaggle_users = pd.read_csv(general, sep=',')

print("---------------------------------------------------------------------------------------------------------")

nomi_campi = kaggle_users.columns
print(f"Nomi dei campi presenti nel dataset {nomi_campi}")

print("---------------------------------------------------------------------------------------------------------")

numero_campi = kaggle_users.shape[1]
print(f"Numero di campi presenti nel dataset: {numero_campi}")

print("---------------------------------------------------------------------------------------------------------")

lunghezza_dataset = kaggle_users.shape[0]
print(f"Lunghezza del dataset (numero di righe): {lunghezza_dataset}")

print("---------------------------------------------------------------------------------------------------------")

print("INFO")
kaggle_users.info()

print("---------------------------------------------------------------------------------------------------------")

statistiche = kaggle_users.describe(include='all')
print(f"STATISTICHE:\n{statistiche}")

print("---------------------------------------------------------------------------------------------------------")

colonna = "ISBOT"
valori_unici = kaggle_users[colonna].unique()  # calcolo i valori unici dalla colonna
print(f"Valori unici colonna {colonna}: {valori_unici}")

num_fake = kaggle_users[kaggle_users['ISBOT'] == True].shape[0]
num_real = kaggle_users[kaggle_users['ISBOT'] == False].shape[0]
num_nan = kaggle_users[kaggle_users['ISBOT'].isnull()].shape[0]
print(f"Numero di account fake: {num_fake}")
print(f"Numero di account real: {num_real}")
print(f"Numero di account con campo ISBOT nullo: {num_nan}")

print("---------------------------------------------------------------------------------------------------------")
