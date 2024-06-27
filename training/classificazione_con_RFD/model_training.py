import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def draw_confusion(df_cm, title, filename):
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt=".0f")
    plt.title(title)
    plt.savefig(f"{filename}-confusion.pdf")
    plt.cla()
    plt.clf()


def early_stopping_fit(model, x_train, y_train, x_val, y_val, patience=5):
    """
    Questa funzione implementa un ciclo di addestramento iterativo con early stopping.
    Viene utilizzata per interrompere l'addestramento se non si osserva un miglioramento significativo delle prestazioni
    sul set di validazione dopo un certo numero di iterazioni.
    :param model: Modello di machine learning da addestrare.
    :param x_train: Feature del set di addestramento.
    :param y_train: Etichette del set di addestramento.
    :param x_val: Feature del set di validazione.
    :param y_val: Etichette del set di validazione.
    :param patience: Numero di iterazioni per cui l'addestramento continua senza miglioramento prima di interrompersi.
    :return: Miglior modello trovato durante il processo di early stopping.
    """

    best_score = 0
    no_improvement_count = 0
    best_model = model
    for i in range(1, 1001):  # Arbitrary large number of iterations
        model.fit(x_train, y_train)
        y_val_pred = model.predict(x_val)
        score = accuracy_score(y_val, y_val_pred)
        if score > best_score:
            best_score = score
            best_model = model
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                break
    return best_model


def fit_and_evaluate(model, x_train, y_train, x_test, y_test, patience=5):
    """
    Questa funzione è stata modificata per includere una suddivisione del set di addestramento
    in un set di addestramento e uno di validazione, e per applicare l'early stopping.
    :param model: Modello di machine learning da addestrare e valutare.
    :param x_train: Feature del set di addestramento.
    :param y_train: Etichette del set di addestramento.
    :param x_test: Feature del set di test.
    :param y_test: Etichette del set di test.
    :param patience: Numero di iterazioni per cui l'addestramento continua senza miglioramento prima di interrompersi.
    :return: Tupla contenente l'accuratezza del modello, il report di classificazione e la matrice di confusione.
    """

    # Split train data to create a validation set
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    # Apply early stopping
    best_model = early_stopping_fit(model, x_train, y_train, x_val, y_val, patience=patience)
    y_pred = best_model.predict(x_test)

    accuracy_value = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(conf_matrix)

    return accuracy_value, report, df_cm


def RF_prediction(x_train, y_train, x_test, y_test, comparison=4):
    max_features = {1: 1, 2: 2, 4: 3}.get(comparison, 3)
    rfc = RandomForestClassifier(random_state=42, bootstrap=True, criterion='entropy',
                                 max_depth=20, max_features=max_features,
                                 min_samples_leaf=1, min_samples_split=10,
                                 n_estimators=500)
    return fit_and_evaluate(rfc, x_train, y_train, x_test, y_test)


def DT_prediction(x_train, y_train, x_test, y_test):
    dt = DecisionTreeClassifier(random_state=42, max_leaf_nodes=3, min_samples_split=2, max_depth=10,
                                min_samples_leaf=1)
    return fit_and_evaluate(dt, x_train, y_train, x_test, y_test)


def SVC_prediction(x_train, y_train, x_test, y_test):
    svc = SVC(C=1000, gamma=0.01, kernel='rbf')
    return fit_and_evaluate(svc, x_train, y_train, x_test, y_test)


def knn_prediction(x_train, y_train, x_test, y_test):
    knn = KNeighborsClassifier(n_neighbors=11, p=1, weights='distance', metric='cosine')
    return fit_and_evaluate(knn, x_train, y_train, x_test, y_test)
