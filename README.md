<div align="center">

# Fake Account Detector

</div>

# Indice

1. [Introduzione](#introduzione)
2. [Risultati](#risultati)
3. [Guida all'Installazione](#guida-allinstallazione)
   - [Installare Python](#installare-python)
   - [Clonare il Repository](#clonare-il-repository)
   - [Creare l'Ambiente Virtuale](#creare-lambiente-virtuale)
   - [Installare i Requisiti](#installare-i-requisiti)
4. [Struttura del Progetto](#struttura-del-progetto)

# Introduzione 

Al giorno d'oggi, i social media rappresentano una parte integrante della vita quotidiana di milioni di persone. Piattaforme come Instagram e Facebook hanno rivoluzionato il modo in cui comunichiamo, condividiamo informazioni e interagiamo con gli altri. Tuttavia, con la crescente popolarità di tali piattaforme è emerso un problema significativo, ovvero, la proliferazione di account falsi. Questi account non autentici possono essere utilizzati per vari scopi dannosi, tra cui la diffusione di informazioni false, l’aumento artificiale dei follower e delle interazioni, e persino per attività illegali. La presenza di account falsi non solo compromette l’integrità delle piattaforme di social media, ma può anche avere conseguenze negative per gli utenti reali, le aziende e la società nel suo complesso. Ad esempio, i fake account possono essere utilizzati per manipolare l’opinione pubblica, influenzare elezioni politiche, danneggiare la reputazione delle persone e delle aziende, e creare un ambiente online meno sicuro. Di fronte a queste sfide, il rilevamento e la rimozione di tali account sono diventati una priorità cruciale per le piattaforme di social media e per i ricercatori nel campo della data science e del machine learning.

In questo progetto, l'obiettivo è il rilevamento di account falsi cercando di dedurre caratteristiche peculiari, in termini di correlazioni all’interno dei dati di un dataset di profilo di social network, con l’obiettivo di migliorare le capacità dei metodi di machine learning di discriminarli. In particolare, viene usata una strategia di feature engineering che sfrutta metadati di profilazione, rappresentati in termini di Dipendenze Funzionali Rilassate (RFDs), che possono essere automaticamente dedotte dai dati.

# Risultati 
I risultati ottenuti dalle varie tecniche di machine learning applicate ai dataset sono riassunti nella directory `analysis_results`, dove sono presenti grafici di confronto delle metriche di accuratezza, precisione, recall e F1-score.

# Guida all'Installazione

Per installare i requisiti necessari al progetto, seguire i passaggi seguenti.

## Installare Python

Verificare di avere Python installato sulla propria macchina. Il progetto è compatibile con Python `3.9`.

Se Python non è installato, fare riferimento alla [Guida ufficiale Python](https://www.python.org/downloads/).

## Clonare il Repository 

Per clonare questo repository, scaricare e estrarre i file del progetto `.zip` utilizzando il pulsante `<Code>` in alto a destra o eseguire il seguente comando nel terminale:
```shell 
git clone https://github.com/Luigina2001/FakeAccountDetector.git
```

## Creare l'Ambiente Virtuale 

È fortemente raccomandato creare un ambiente virtuale per il progetto e attivarlo prima di procedere. Per creare l'ambiente virtuale si consiglia di utilizzare `pip`. Fare riferimento a [Creare un ambiente virtuale](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment).

## Installare i Requisiti

Per installare i requisiti, si prega di:
1. Assicurarsi di aver **attivato l'ambiente virtuale dove sono stati installati i requisiti del progetto**. Se attivato, il terminale dovrebbe apparire come segue: ``(nome-del-tuo-ambiente-virtuale) user@user path``

2. Installare i requisiti del progetto usando `pip`:
```shell 
pip install -r requirements.txt
```

# Struttura del Progetto

## Citazione del Dataset

Per ulteriori informazioni circa il dataset utilizzato in questo progetto (`instagram2/user_fake_authentic_2class.csv`), fare riferimento al seguente articolo:
@article{purba2020classification,
title={Classification of instagram fake users using supervised machine learning algorithms},
author={Purba, Kristo Radion and Asirvatham, David and Murugesan, Raja Kumar},
journal={International Journal of Electrical and Computer Engineering},
volume={10},
number={3},
pages={2763},
year={2020},
publisher={IAES Institute of Advanced Engineering and Science}
}

## Directory `datasets`

- `facebook.csv`: Dataset degli utenti di Facebook.
- `instagram/test.csv`: Dataset di test per Instagram.
- `instagram/train.csv`: Dataset di addestramento per Instagram.
- `instagram2/user_fake_authentic_2class.csv`: Dataset Instagram con classificazione binaria (citazione: purba2020classification).
- `instagram2/user_fake_authentic_4class.csv`: Dataset Instagram con classificazione a quattro classi (citazione: purba2020classification).
- `kaggle.csv`: Dataset degli utenti di Kaggle.
- `weibo/fusers.csv`: Dataset degli utenti falsi di Weibo.
- `weibo/users.csv`: Dataset degli utenti di Weibo.

## Directory `preprocessing`

- `instagram/instagram2_2class.py`: Script per preprocessare il dataset Instagram con classificazione binaria.
- `instagram/instagram2_4class.py`: Script per analizzare il dataset Instagram con classificazione a quattro classi.
- `instagram/instagram_test.py`: Script per analizzare il dataset di test di Instagram.
- `instagram/instagram_train.py`: Script per analizzare il dataset di addestramento di Instagram.
- `weibo/f_users.py`: Script per analizzare il dataset degli utenti falsi di Weibo.
- `weibo/users.py`: Script per analizzare il dataset degli utenti di Weibo.
- `facebook.py`: Script per analizzare il dataset degli utenti di Facebook.

## Directory `dataset_normalizzato`

- `user_fake_authentic_2class_cleaned.csv`: Dataset Instagram normalizzato con classificazione binaria.

## Directory `training`

- `decision_tree/decision_tree.py`: Script per addestrare un modello Decision Tree.
- `decision_tree/dt_confusion_matrix.png`: Matrice di confusione del modello Decision Tree.
- `decision_tree/dt_model.pkl`: Modello Decision Tree salvato.
- `decision_tree/dt_roc_curve.png`: Curva ROC del modello Decision Tree.
- `decision_tree/metrics.csv`: Metriche di valutazione del modello Decision Tree.

- `KNN/knn.py`: Script per addestrare un modello K-Nearest Neighbors.
- `KNN/knn_confusion_matrix.png`: Matrice di confusione del modello K-Nearest Neighbors.
- `KNN/knn_model.pkl`: Modello K-Nearest Neighbors salvato.
- `KNN/knn_roc_curve.png`: Curva ROC del modello K-Nearest Neighbors.
- `KNN/metrics.csv`: Metriche di valutazione del modello K-Nearest Neighbors.

- `random_forest/random_forest.py`: Script per addestrare un modello Random Forest.
- `random_forest/rf_confusion_matrix.png`: Matrice di confusione del modello Random Forest.
- `random_forest/rf_model.pkl`: Modello Random Forest salvato.
- `random_forest/rf_roc_curve.png`: Curva ROC del modello Random Forest.
- `random_forest/metrics.csv`: Metriche di valutazione del modello Random Forest.

- `SVM/svm.py`: Script per addestrare un modello Support Vector Machine.
- `SVM/svm_confusion_matrix.png`: Matrice di confusione del modello Support Vector Machine.
- `SVM/svm_model.pkl`: Modello Support Vector Machine salvato.
- `SVM/svm_roc_curve.png`: Curva ROC del modello Support Vector Machine.
- `SVM/metrics.csv`: Metriche di valutazione del modello Support Vector Machine.

- `utils.py`: Script di utilità per varie funzioni di supporto.

- `classificazione_con_RFD/data_initialization.py`: Script per l'inizializzazione dei dati.
- `classificazione_con_RFD/main.py`: Script principale per la classificazione con RFD.
- `classificazione_con_RFD/model_training.py`: Script per l'addestramento dei modello con RFD.
- `classificazione_con_RFD/rename_columns_RFD_dataset.py`: Script per rinominare le colonne del dataset RFD.
- `classificazione_con_RFD/Results_instagram_bilanciato_tipo2.csv`: Risultati della classificazione relativa al dataset con RFD

## Directory `DatasetAfterFeatureSelection`

Contiene i file CSV relativi ai dataset ottenuti a seguito del processo di aggiunta delle RFD ibride al dataset originale normalizzato al variare delle soglie di attribute comparison ed extent.

## Directory `analysis_results`

- `analysis.py`: Script per effettuare l'analisi dei risultati.

Immagini dei grafici relativi le metriche ottenute in `analysis.py`:
- `accuracy_comparison.png`
- `accuracy_comparison_thr0_ext4.png`
- `accuracy_comparison_thr0_ext5.png`
- `f1_comparison.png`
- `f1_comparison_thr0_ext4.png`
- `f1_comparison_thr0_ext5.png`
- `precision_comparison.png`
- `precision_comparison_thr0_ext4.png`
- `precision_comparison_thr0_ext5.png`
- `recall_comparison.png`
- `recall_comparison_thr0_ext4.png`
- `recall_comparison_thr0_ext5.png`
