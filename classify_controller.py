from views.classify.classify_view import ClassifyView
from models.custom_table_model import CustomTableModel
from PyQt5.QtWidgets import QMessageBox
from models.classifier import Classifier
from models.dataframe import DataFrame
import time
from models.evaluation import Evaluation
from models.model import Model
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import os
import psutil


class ClassifyController:
    def __init__(self, view: ClassifyView, model: Model):
        self.view = view
        self.model = model

        self.load_table_models()
        self.load_classifier_options()

        self.view.classifier_combobox.currentIndexChanged.connect(
            self.change_table_header
        )
        self.view.run_button.clicked.connect(self.run_classifier)

        self.change_table_header(0)

    def load_table_models(self):
        general_cols = [
            "Uso de CPU",
            "Tiempo de ejecución",
            "Exactitud",
            "Precisión",
            "Recall",
            "F-Measure",
            "AUC",
        ]
        logistic_regression_cols = ["Algoritmo", *general_cols]
        neural_network_cols = [
            "Algoritmo",
            "Tasa de aprendizaje",
            "Función",
            "Tamaño",
            *general_cols,
        ]
        svm_cols = ["Algoritmo", "Kernel", *general_cols]
        bayesian_cols = ["Algoritmo", *general_cols]
        decision_tree_cols = ["Algoritmo", *general_cols]
        neighbors_tree_cols = ["Algoritmo", *general_cols]
        sgd_tree_cols = ["Algoritmo", *general_cols]

        self.table_models = [
            CustomTableModel(logistic_regression_cols),
            CustomTableModel(neural_network_cols),
            CustomTableModel(svm_cols),
            CustomTableModel(bayesian_cols),
            CustomTableModel(decision_tree_cols),
            CustomTableModel(neighbors_tree_cols),
            CustomTableModel(sgd_tree_cols),
        ]

    def load_classifier_options(self):
        options = [
            "Regresión logística",
            "Red neuronal",
            "SVM",
            "Bayesiano",
            "Árbol de decisión",
            "Neighbors",
            "SGD",
        ]
        classifier_combobox = self.view.classifier_combobox
        classifier_combobox.addItems(options)

    def change_table_header(self, index: int):
        table_model = self.table_models[index]
        self.view.classifier_table.setModel(table_model)

    # Retornar los valores de la evaluación del modelo con la prueba
    def evaluation_results(self, model, train_df: DataFrame, test_df: DataFrame):
        # Calcular el cpu
        load1, load5, load15 = psutil.getloadavg()

        # transcurre el tiempo
        start_time = time.time()
        # Envoltura de modelo
        classifier = Classifier(model)
        # Entrenar modelo
        classifier.build_classifier(train_df)

        # Instanciar evaluación
        evaluation = Evaluation()

        # Evaluar modelo con el dataset de prueba
        evaluation.evaluate_model(classifier.get_model(), test_df)

        end_time = time.time()

        cpu_usage = (load15 / os.cpu_count()) * 100

        result = (
            str(cpu_usage),
            str(end_time - start_time),
            str(evaluation.accuracy_score),
            str(evaluation.precision_score),
            str(evaluation.recall_score),
            str(evaluation.f1_score),
            str(evaluation.auc_score),
        )
        return result

    # Ejecutar clasificadores
    def run_classifier(self):
        classifier_index = self.view.classifier_combobox.currentIndex()

        train_df = self.model.get_train_df()
        test_df = self.model.get_test_df()

        if train_df is None:
            message_box = QMessageBox()
            message_box.setText("Dataset de entramiento no importado")
            message_box.setInformativeText(
                "Debes importar el dataset de entramiento para realizar esta acción."
            )
            message_box.exec()
            return

        if test_df is None:
            message_box = QMessageBox()
            message_box.setText("Dataset de prueba no importado")
            message_box.setInformativeText(
                "Debes importar el dataset de prueba para realizar esta acción."
            )
            message_box.exec()
            return

        table_model = self.table_models[classifier_index]

        # 0) CLASIFICADORES DE REGRESIÓN LOGÍSTICA
        if classifier_index == 0:
            # 1) MODELO DE REGRESIÓN LOGÍSTICA - REGRESIÓN LOGÍSTICA
            # solver: define que algoritmo se utilizar en el problema de optimización, lbfgs manejan la pérdida multinomial
            # max_iter: define el numero de iteraciones que tendra el modelo
            # penalty: especifica la norma utilizada en la penalización

            # Instanciar modelo
            model = LogisticRegression(
                solver="liblinear",
                max_iter=1000,
                penalty="l1",
            )
            # Retornar los valores de la evaluación del modelo con la prueba
            results = self.evaluation_results(model, train_df, test_df)
            # Agregar resultados a fila de la tabla
            table_model.insert_row(["Regresión logística", *results])

        # 1) CLASIFICADORES DE REDES NEURONALES
        if classifier_index == 1:
            # 1) MODELO DE CON 1 CAPA - REGRESIÓN LOGÍSTICA
            # activation: función de salida identidad
            # hidden_layer_sizes:  capas ocultas
            # max_iter: el numero máximo de iteracciones.
            # verbose: cancela el mostrar mensajes de progreso
            # solver: considerado porque para los conjunsos relativamente grandes
            # learning_rate_init: controla la actualización de pesos
            # alpha: penalización, en cuanto a los valores de regularización.
            # beta_1: bajar la estimación del primer vector solo se usa con adam
            # tol: optimiza cuando la pérdida está mejorando durante iteraciones consecutivas
            # learning_rate: tasa de aprendizaje para actualizaciones de peso.

            # Instanciar modelo
            model = MLPClassifier(
                activation="relu",
                hidden_layer_sizes=(300),
                max_iter=1000,
                verbose=False,
                solver="adam",
                learning_rate_init=0.005,
                alpha=0.003,
                beta_1=0.7,
                tol=0.002,
                learning_rate="constant",
            )
            # Retornar los valores de la evaluación del modelo con la prueba
            results = self.evaluation_results(model, train_df, test_df)
            # Agregar resultados a fila de la tabla
            table_model.insert_row(
                ["Red neuronal 1", str(0.005), "relu", "[300]", *results]
            )

            # 2) MODELO DE CON 2 CAPAS - REGRESIÓN LOGÍSTICA
            # activation: función de salida identidad
            # hidden_layer_sizes:  capas ocultas
            # max_iter: el numero máximo de iteracciones.
            # verbose: cancela el mostrar mensajes de progreso
            # solver: considerado porque para los conjunsos relativamente grandes
            # learning_rate_init: controla la actualización de pesos
            # alpha: penalización, en cuanto a los valores de regularización.
            # beta_1: bajar la estimación del primer vector solo se usa con adam
            # tol: optimiza cuando la pérdida está mejorando durante iteraciones consecutivas
            # learning_rate: tasa de aprendizaje para actualizaciones de peso.

            # Instanciar modelo
            model = MLPClassifier(
                activation="relu",
                hidden_layer_sizes=(70, 3),
                max_iter=1000,
                verbose=False,
                solver="adam",
                learning_rate_init=0.005,
                alpha=0.003,
                beta_1=0.7,
                tol=0.002,
                learning_rate="constant",
            )
            # Retornar los valores de la evaluación del modelo con la prueba
            results = self.evaluation_results(model, train_df, test_df)
            # Agregar resultados a fila de la tabla
            table_model.insert_row(
                ["Red neuronal 2", str(0.005), "relu", "[70, 3]", *results]
            )

            # 3) MODELO DE CON 3 CAPAS - REGRESIÓN LOGÍSTICA
            # activation: función de salida identidad
            # hidden_layer_sizes:  capas ocultas
            # max_iter: el numero máximo de iteracciones.
            # verbose: cancela el mostrar mensajes de progreso
            # solver: considerado porque para los conjunsos relativamente grandes
            # learning_rate_init: controla la actualización de pesos
            # alpha: penalización, en cuanto a los valores de regularización.
            # beta_1: bajar la estimación del primer vector solo se usa con adam
            # tol: optimiza cuando la pérdida está mejorando durante iteraciones consecutivas
            # learning_rate: tasa de aprendizaje para actualizaciones de peso.

            # Instanciar modelo
            model = MLPClassifier(
                activation="relu",
                hidden_layer_sizes=(50, 20, 30),
                max_iter=1000,
                verbose=False,
                solver="adam",
                learning_rate_init=0.005,
                alpha=0.003,
                beta_1=0.7,
                tol=0.002,
                learning_rate="constant",
            )
            # Retornar los valores de la evaluación del modelo con la prueba
            results = self.evaluation_results(model, train_df, test_df)
            # Agregar resultados a fila de la tabla
            table_model.insert_row(
                ["Red neuronal 3", str(0.005), "relu", "[50, 20, 30]", *results]
            )

            # 4) MODELO DE CON N CAPAS - REGRESIÓN LOGÍSTICA

        # 2) CLASIFICADORES DE SUPER VECTOR MANCHINE (SVM)
        elif classifier_index == 2:
            # 1) MODELO SVM CON KERNEL LINEAR - SVM
            # probability: Se utiliza para habilitar estimaciones de probabilidad
            # Instanciar modelo
            model = SVC(kernel="linear", probability=True)
            # Retornar los valores de la evaluación del modelo con la prueba
            results = self.evaluation_results(model, train_df, test_df)
            # Agregar resultados a fila de la tabla
            table_model.insert_row(["SVM", "linear", *results])

            # 2) MODELO SVM CON KERNEL POLINOMIAL - SVM
            # Instanciar modelo
            # probability: Se utiliza para habilitar estimaciones de probabilidad
            # degree: Grado de la función del núcleo polinomial
            model = SVC(kernel="poly", probability=True, degree=1.5)
            # Retornar los valores de la evaluación del modelo con la prueba
            results = self.evaluation_results(model, train_df, test_df)
            # Agregar resultados a fila de la tabla
            table_model.insert_row(["SVM", "Polinomial", *results])

            # 3) MODELO SVM CON KERNEL RADIAL - SVM
            # probability: Se utiliza para habilitar estimaciones de probabilidad
            # Instanciar modelo
            model = SVC(kernel="rbf", probability=True)
            # Retornar los valores de la evaluación del modelo con la prueba
            results = self.evaluation_results(model, train_df, test_df)
            # Agregar resultados a fila de la tabla
            table_model.insert_row(["SVM", "Radial", *results])

            # 4) MODELO SVM CON KERNEL RADIAL - SVM
            # probability: Se utiliza para habilitar estimaciones de probabilidad
            # gamma: decide cuánta curvatura queremos en un límite de decisión
            # Instanciar modelo
            model = SVC(kernel="sigmoid", probability=True, gamma="auto")
            # Retornar los valores de la evaluación del modelo con la prueba
            results = self.evaluation_results(model, train_df, test_df)
            # Agregar resultados a fila de la tabla
            table_model.insert_row(["SVM", "Sigmoide", *results])

        # 3) CLASIFICADORES DE BAYESIANO
        elif classifier_index == 3:
            # 1) MODELO DE NAIVE BAYES - BAYESIANO
            # probability: Se utiliza para habilitar estimaciones de probabilidad
            # Instanciar modelo
            model = GaussianNB()
            # Retornar los valores de la evaluación del modelo con la prueba
            results = self.evaluation_results(model, train_df, test_df)
            # Agregar resultados a fila de la tabla
            table_model.insert_row(["Naive Bayes", *results])

            # 2) MODELO DE MULTINOMIAL NAIVE BAYES - BAYESIANO
            # probability: Se utiliza para habilitar estimaciones de probabilidad
            # Instanciar modelo
            model = MultinomialNB()
            # Retornar los valores de la evaluación del modelo con la prueba
            results = self.evaluation_results(model, train_df, test_df)
            # Agregar resultados a fila de la tabla
            table_model.insert_row(["Multinomial Naive Bayes", *results])

            # 3) MODELO DE BAYESNET - BAYESIANO

        # 4) CLASIFICADORES DE ARBOL DE DECISIONES
        elif classifier_index == 4:
            # 1) MODELO DE RANDOM FOREST - ARBOL DE DECISIONES
            # n_estimators: La cantidad de árboles en el bosque.
            # random_state: Controla la aleatoriedad al construir los árboles
            # min_samples_leaf: El número mínimo de muestras necesarias para estar en un nodo hoja.
            # max_depth: para prepodar el árbol y asegurar que no tenga una profundidad mayor a 4
            # max_leaf_nodes: el número máximo de nodos hoja

            # Instanciar modelo
            model = RandomForestClassifier(
                n_estimators=100, random_state=2016, min_samples_leaf=8
            )
            # Retornar los valores de la evaluación del modelo con la prueba
            results = self.evaluation_results(model, train_df, test_df)
            # Agregar resultados a fila de la tabla
            table_model.insert_row(["RandomForest", *results])

            # 2) MODELO DE J48 - ARBOL DE DECISIONES
            # es para llamar al metodo de decision tree (j48)
            # el valor de 10 puede cambiar, significa el numero de arboles que se inicia
            model = DecisionTreeClassifier(
                max_leaf_nodes=10, criterion="entropy", max_depth=4, random_state=0
            )
            # Retornar los valores de la evaluación del modelo con la prueba
            results = self.evaluation_results(model, train_df, test_df)
            # Agregar resultados a fila de la tabla
            table_model.insert_row(["J48 ", *results])

        # 4) CLASIFICADORES DE NEIGHBORS
        elif classifier_index == 5:
            # 1) MODELO DE K-Nearest - NEIGHBORS
            # n_neighbors: vecinos cercanos
            # p: parametro de potencia
            # weights: Todos los puntos de cada vecindario se ponderan por igual.
            # algorithm: Intentará decidir el algoritmo más apropiado (tambien hay de 'ball_tree', 'kd_tree', 'brute')

            # Instanciar modelo
            model = KNeighborsClassifier(
                n_neighbors=60, p=2, weights="uniform", algorithm="auto"
            )
            # Retornar los valores de la evaluación del modelo con la prueba
            results = self.evaluation_results(model, train_df, test_df)
            # Agregar resultados a fila de la tabla
            table_model.insert_row(["KNeighbors ", *results])

        elif classifier_index == 6:
            # Instanciar modelo
            # tol: técnica para la optimización
            # max_iter: el numero máximo de iteracciones.
            model = SGDClassifier(
                max_iter=1000, tol=1e-3, penalty="l1", loss="modified_huber"
            )
            # Retornar los valores de la evaluación del modelo con la prueba
            results = self.evaluation_results(model, train_df, test_df)
            # Agregar resultados a fila de la tabla
            table_model.insert_row(["SGD ", *results])
