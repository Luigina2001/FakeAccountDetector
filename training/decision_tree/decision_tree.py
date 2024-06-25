from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from training.utils import *

model_filename = 'dt_model.pkl'
dataset_path = '../../dataset_normalizzato/user_fake_authentic_2class_cleaned.csv'

def train_model(X_train, y_train):
    param_grid = {
        'max_depth': [None, 10, 20, 30], # Profondità massima dell'albero
        'min_samples_split': [2, 5, 10], # Numero minimo di campioni richiesti per suddividere un nodo
        'min_samples_leaf': [1, 2, 4] # Numero minimo di campioni richiesti per essere in un nodo foglia
    }
    dt = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def plot_roc_curve(model, X_test, y_test):
    # Calcolo delle probabilità previste dal modello
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calcolo dei valori di FPR e TPR per diverse soglie
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Plot della ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasso di Falsi Positivi')
    plt.ylabel('Tasso di Veri Positivi')
    plt.title('Curva ROC (Receiver Operating Characteristic) - Decision Tree')
    plt.legend(loc='lower right')

    # Salvataggio della ROC curve come immagine
    plt.savefig('dt_roc_curve.png')
    # plt.show()

def main():
    X_train, X_test, y_train, y_test = load_data(dataset_path)
    model = load_model(model_filename)
    if model is None:
        model = train_model(X_train, y_train)
        save_model(model, model_filename)
    evaluate_model(model, X_test, y_test)

    parameters = model.get_params()

    print("\nMigliori parametri trovati dalla Grid Search:")
    print(f"max_depth: {parameters['max_depth']}")
    print(f"min_samples_split: {parameters['min_samples_split']}")
    print(f"min_samples_leaf: {parameters['min_samples_leaf']}")

    plot_and_save_confusion_matrix(model, X_test, y_test, 'dt')
    plot_roc_curve(model, X_test, y_test)

if __name__ == "__main__":
    main()