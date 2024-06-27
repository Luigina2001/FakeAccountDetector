import os
import pandas as pd
from tqdm import tqdm
from data_initialization import initialize_data
from model_training import RF_prediction, DT_prediction, SVC_prediction, knn_prediction


def main():
    full_datasets = [f'{x}' for x in os.listdir('../../DatasetAfterFeatureSelection/')]
    results_dict = {}

    for dataset_name in tqdm(full_datasets, desc="Datasets"):
        tipo = 1
        comparison_thr = dataset_name.split("_")[1] if tipo != 2 else -1

        for thr in tqdm([0.1, 0.2, 0.3, 0.4, 0.5], desc=f"Thresholds for {dataset_name}"):
            x_train, x_test, y_train, y_test = initialize_data(dataset_name, tipo, thr)

            if x_train.shape[1] < 3:
                accuracy_value_RF, report_RF, df_cm_RF = RF_prediction(x_train, y_train, x_test, y_test,
                                                                       comparison=x_train.shape[1])
            else:
                accuracy_value_RF, report_RF, df_cm_RF = RF_prediction(x_train, y_train, x_test, y_test)

            results_dict[f'RF_{dataset_name}_{str(thr).replace(".", "-")}_{tipo}'] = report_RF
            report_RF.update(
                {'dataset': dataset_name, 'comparison_thr': comparison_thr, 'model': 'RF', 'thr': thr, 'tipo': tipo,
                 'accuracy_value': accuracy_value_RF, 'TP': df_cm_RF[0][0], 'FP': df_cm_RF[0][1], 'FN': df_cm_RF[1][0],
                 'TN': df_cm_RF[1][1]})

            accuracy_value_DT, report_DT, df_cm_DT = DT_prediction(x_train, y_train, x_test, y_test)
            results_dict[f'DT_{dataset_name}_{str(thr).replace(".", "-")}_{tipo}'] = report_DT
            report_DT.update(
                {'dataset': dataset_name, 'comparison_thr': comparison_thr, 'model': 'DT', 'thr': thr, 'tipo': tipo,
                 'accuracy_value': accuracy_value_DT, 'TP': df_cm_DT[0][0], 'FP': df_cm_DT[0][1], 'FN': df_cm_DT[1][0],
                 'TN': df_cm_DT[1][1]})

            accuracy_value_SVC, report_SVC, df_cm_SVC = SVC_prediction(x_train, y_train, x_test, y_test)
            results_dict[f'SVC_{dataset_name}_{str(thr).replace(".", "-")}_{tipo}'] = report_SVC
            report_SVC.update(
                {'dataset': dataset_name, 'comparison_thr': comparison_thr, 'model': 'SVC', 'thr': thr, 'tipo': tipo,
                 'accuracy_value': accuracy_value_SVC, 'TP': df_cm_SVC[0][0], 'FP': df_cm_SVC[0][1],
                 'FN': df_cm_SVC[1][0], 'TN': df_cm_SVC[1][1]})

            accuracy_value_KNN, report_KNN, df_cm_KNN = knn_prediction(x_train, y_train, x_test, y_test)
            results_dict[f'KNN_{dataset_name}_{str(thr).replace(".", "-")}_{tipo}'] = report_KNN
            report_KNN.update(
                {'dataset': dataset_name, 'comparison_thr': comparison_thr, 'model': 'KNN', 'thr': thr, 'tipo': tipo,
                 'accuracy_value': accuracy_value_KNN, 'TP': df_cm_KNN[0][0], 'FP': df_cm_KNN[0][1],
                 'FN': df_cm_KNN[1][0], 'TN': df_cm_KNN[1][1]})

    dfs = [pd.json_normalize(results_dict[val]) for val in results_dict]
    dftmp = pd.concat(dfs)
    dftmp.to_csv('Results_instagram_bilanciato_tipo2.csv', index=False, sep=';')


if __name__ == "__main__":
    main()
