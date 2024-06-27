import os
import pandas as pd


def rename_columns_in_place(path, columns_map):
    """
    Questa funzione rinomina i campi di un file CSV esistente sovrascrivendolo.

    :param path: Percorso del file CSV da modificare.
    :param columns_map: Dizionario che mappa i nomi dei campi attuali ai nuovi nomi.
                       Esempio: {'vecchio_nome1': 'nuovo_nome1', 'vecchio_nome2': 'nuovo_nome2'}
    """

    df = pd.read_csv(path, sep=";")
    df.rename(columns=columns_map, inplace=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    main_dir = '../../DatasetAfterFeatureSelection'
    paths = []
    for file_name in os.listdir(main_dir):
        paths.append(file_name)

    # print(paths)
    # print(len(paths))

    i = 1
    for path in paths:
        csv_path = main_dir + "/" + path

        columns_to_rename = {
            'Attr0': 'pos',
            'Attr1': 'flw',
            'Attr2': 'flg',
            'Attr3': 'bl',
            'Attr4': 'pic',
            'Attr5': 'lin',
            'Attr6': 'cl',
            'Attr7': 'cz',
            'Attr8': 'ni',
            'Attr9': 'erl',
            'Attr10': 'erc',
            'Attr11': 'lt',
            'Attr12': 'hc',
            'Attr13': 'pr',
            'Attr14': 'fo',
            'Attr15': 'cs',
            'Attr16': 'pi'
        }

        rename_columns_in_place(csv_path, columns_to_rename)
        print(f"{i}. I campi del file CSV sono stati rinominati in {csv_path}")
        i = i + 1
