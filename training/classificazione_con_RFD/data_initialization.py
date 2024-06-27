import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def initialize_data(dataset, tipo, threshold, percentage=100, dataset_num_column=19):
    train_path = '../../DatasetAfterFeatureSelection/' + dataset
    train_df = pd.read_csv(train_path, sep=',', encoding="ISO-8859-1")
    train_df = train_df.fillna(1)

    last_n_columns = train_df.iloc[:, -(len(train_df.axes[1]) - dataset_num_column):]
    native_columns = train_df.iloc[:, :dataset_num_column]

    last_n_columns = last_n_columns.sort_values(by=len(train_df) - 1, axis=1)
    first_selected_columns = last_n_columns.iloc[:, :(int((len(last_n_columns.axes[1]) * percentage) / 100))]

    number_of_columns_to_select = 0
    occurrence = False
    for i in range(0, len(first_selected_columns.axes[1])):
        if first_selected_columns.iloc[len(train_df) - 1, i] > threshold:
            occurrence = True
            number_of_columns_to_select = i - 1
            break

    if occurrence:
        final_selected_columns = first_selected_columns.iloc[:, :number_of_columns_to_select]
    else:
        final_selected_columns = first_selected_columns

    label_encoder = preprocessing.LabelEncoder()
    if tipo == 1:
        train_df = pd.concat([native_columns, final_selected_columns], axis=1).iloc[:-1, :]
    elif tipo == 0:
        train_df = pd.concat([final_selected_columns, native_columns['class']], axis=1).iloc[:-1, :]
    else:
        train_df = native_columns.iloc[:-1, :]

    Y = label_encoder.fit_transform(train_df["class"])
    train_df.drop(columns=["class"], inplace=True)

    x_train, x_test, y_train, y_test = train_test_split(train_df, Y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test
