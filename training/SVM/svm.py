from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from training.utils import *

model_filename = 'svm_model.pkl'
dataset_path = '../../dataset_normalizzato/user_fake_authentic_2class_cleaned.csv'


def train_model(train, y_train):
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],  # Parametro di regolarizzazione
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],  # Coefficiente del kernel
        'kernel': ['rbf']  # Tipo di kernel
    }
    svm = SVC(probability=True)
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, refit=True, verbose=3)
    grid_search.fit(train, y_train)
    return grid_search.best_estimator_\


def plot_roc_curve(model, test, y_test):
    # Calcolo delle probabilità previste dal modello
    y_prob = model.predict_proba(test)[:, 1]

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
    plt.title('Curva ROC (Receiver Operating Characteristic) - SVM')
    plt.legend(loc='lower right')

    plt.savefig('svm_roc_curve.png')
    plt.show()


def main():
    train, test, y_train, y_test = load_data(dataset_path)

    model = load_model(model_filename)
    if model is None:
        model = train_model(train, y_train)
        save_model(model, model_filename)
    evaluate_model(model, test, y_test)

    parameters = model.get_params()

    print("\nMigliori parametri trovati dalla Grid Search:")
    print(f"C: {parameters['C']}")
    print(f"gamma: {parameters['gamma']}")
    print(f"kernel: {parameters['kernel']}")

    plot_and_save_confusion_matrix(model, test, y_test, 'svm')
    plot_roc_curve(model, test, y_test)


if __name__ == "__main__":
    main()
