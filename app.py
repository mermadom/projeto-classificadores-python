from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix,ConfusionMatrixDisplay
from sklearn.datasets import load_iris
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Carregue seu conjunto de dados aqui
# X, y = ...

# Divida o conjunto de dados em treinamento e teste
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Coleta e Preparação de Dados.
iris = load_iris()
X = iris.data # caracteristica
y = iris.target # rotulos

# Divisão dos Dados em Treinamento e Teste.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Dicionário de classificadores
classifiers = {
    '---': '',
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(),
    'MLP': MLPClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

classifiers_params = {
    'KNN': ['KNN_n_neighbors', 'KNN_algorithm', 'KNN_leaf_size'],
    'SVM': ['SVM_degree', 'SVM_gamma', 'SVM_max_iter'],
    'MLP': ['MLP_random_state', 'MLP_max_iter', 'MLP_activation'],
    'Decision Tree': ['DT_max_leaf_nodes', 'DT_random_state', 'DT_max_depth'],
    'Random Forest': ['RF_n_estimators', 'RF_random_state', 'RF_max_depth']
}


# Rota principal
@app.route('/', methods=['GET', 'POST'])
def index():
    global selected_classifier
    confusion_matrix_img = ""  # Variável para armazenar a imagem da matriz de confusão
    selected_classifier =  'KNN'
    classification_results = {}  # Variável para armazenar outros resultados da classificação
    if request.method == 'POST':
        classifier_name = request.form['classifier']
        params = {}
        if classifier_name == 'KNN':
            for param_name in request.form:
                if param_name == 'KNN_n_neighbors':
                    param_value = int(request.form[param_name])
                    params['n_neighbors'] = param_value
                if param_name == 'KNN_algorithm':
                    param_value = str(request.form[param_name])
                    params['algorithm'] = param_value
                if param_name == 'KNN_leaf_size':
                    param_value = int(request.form[param_name])
                    params['leaf_size'] = param_value

        if classifier_name == 'SVM':
            for param_name in request.form:
                if param_name == 'SVM_degree':
                    param_value = int(request.form[param_name])
                    params['degree'] = param_value
                if param_name == 'SVM_gamma':
                    param_value = str(request.form[param_name])
                    params['gamma'] = param_value
                if param_name == 'SVM_max_iter':
                    param_value = int(request.form[param_name])
                    params['max_iter'] = param_value

        if classifier_name == 'MLP':
            for param_name in request.form:
                if param_name == 'MLP_random_state':
                    param_value = int(request.form[param_name])
                    params['random_state'] = param_value
                if param_name == 'MLP_max_iter':
                    param_value = int(request.form[param_name])
                    params['max_iter'] = param_value
                if param_name == 'MLP_activation':
                    param_value = str(request.form[param_name])
                    params['activation'] = param_value

        if classifier_name == 'Decision Tree':
            for param_name in request.form:
                if param_name == 'DT_max_leaf_nodes':
                    param_value = int(request.form[param_name])
                    params['max_leaf_nodes'] = param_value
                if param_name == 'DT_random_state':
                    param_value = int(request.form[param_name])
                    params['random_state'] = param_value
                if param_name == 'DT_max_depth':
                    param_value = int(request.form[param_name])
                    params['max_depth'] = param_value

        if classifier_name == 'Random Forest':
            for param_name in request.form:
                if param_name == 'RF_n_estimators':
                    param_value = int(request.form[param_name])
                    params['n_estimators'] = param_value
                if param_name == 'RF_random_state':
                    param_value = int(request.form[param_name])
                    params['random_state'] = param_value
                if param_name == 'RF_max_depth':
                    param_value = int(request.form[param_name])
                    params['max_depth'] = param_value
        print(params)
        classifier = classifiers[classifier_name].set_params(**params)
        

        # Treinamento do Modelo.
        classifier.fit(X_train, y_train)

        # Teste / Previsão do Modelo.
        y_pred = classifier.predict(X_test)

        print(f'PRED: {y_pred}')
        print(f'RESPOSTA: {y_test}')
        # Treine o classificador e obtenha resultados
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        classes = iris.target_names.tolist()
        confusion_matrix_img = plot_confusion_matrix(y_test, y_pred,classes)
        
        classification_results = {"accuracy": round(accuracy_score(y_test, y_pred),3), "macro-avg": round(f1_score(y_test, y_pred, average='macro'),3)}

        # Adicione a lógica para exibir os resultados na página

    return render_template('index.html', classifiers=list(classifiers.keys()),classifier_params=classifiers_params, confusion_matrix_img=confusion_matrix_img, classification_results=classification_results)

# Função para gerar a imagem da matriz de confusão
def plot_confusion_matrix(y_true, y_pred,classes):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.read()).decode('utf-8')

if __name__ == '__main__':
    app.run(debug=True)