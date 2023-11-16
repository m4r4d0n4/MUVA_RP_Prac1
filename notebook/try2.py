
import random
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split, cross_validate
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, accuracy_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.mixture import GaussianMixture
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.manifold import LocallyLinearEmbedding, Isomap, TSNE
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron
from sklearn.svm import LinearSVC, NuSVC, LinearSVR, NuSVR, SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.manifold import LocallyLinearEmbedding, Isomap, TSNE
from datetime import datetime
import csv
#UTILS
def describe_pipeline(pipeline):
    """
    Función para describir las funciones aplicadas en cada etapa de un pipeline de Scikit-Learn.

    Args:
    pipeline (Pipeline): El pipeline de Scikit-Learn a describir.

    Returns:
    str: Una cadena de texto describiendo las funciones aplicadas en cada etapa del pipeline.
    """
    steps = pipeline.steps
    
    descriptions = []
    for step_name, step_object in steps:
        if isinstance(step_object, Pipeline):
            # Si el paso es un pipeline, describe recursivamente el subpipeline
            subpipeline_description = describe_pipeline(step_object)
            descriptions.append(f"{step_name}:\n{subpipeline_description}")
        else:
            if isinstance(step_object, BaseEstimator):
                descriptions.append(f"{step_name}: {type(step_object).__name__}")
            else:
                descriptions.append(f"{step_name}: Custom Function")
    
    return '\n'.join(descriptions)


def guardar_en_csv(texto, params ,accuracy):
    # Obtener la fecha y hora actual
    fecha_actual = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Eliminar saltos de línea del texto
    texto = texto.replace('\n', ' ')

    
    params_str = f'{params}'
    info = texto + " / " + params_str
    data = [[fecha_actual,info, accuracy ]]

    # Nombre del archivo CSV
    nombre_archivo = 'metricas_obtenidas.csv'

    # Escribir los datos en el archivo CSV
    with open(nombre_archivo, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Si es la primera vez que se escribe en el archivo, se escribe el encabezado
        if file.tell() == 0:
            writer.writerow(['Fecha y Hora','Texto', 'Accuracy' ])
        writer.writerows(data)

def lecturaDatos(path_train:str, path_tags:str):   

    #Leemos el CSV de entrenamiento
    X_init = pd.read_csv(path_train, sep = ',', decimal = '.', index_col=0)

    #Leemos el CSV de entrenamiento
    Y_train_init = pd.read_csv(path_tags, sep = ',', decimal = '.', index_col=0)

    #Quitamos la columna de´nombre de la foto que no nos da ninguna info
    X_train_init = X_init.iloc[:, 1:]
    return X_train_init,Y_train_init

def filteringFeatures()->Pipeline:
    #Normalizado y demás
    filters_features = []
    #Normalizado
    minmax = MinMaxScaler().set_output(transform='pandas')
    filters_features.append( ('minmax', minmax) )
    var_th = 0.01
    var_thrs = VarianceThreshold(var_th).set_output(transform='pandas')
    filters_features.append(('var_thrs', var_thrs))
    return Pipeline(filters_features)

def addingFeatures()->Pipeline:
    more_features = []
    #degree es la variable que editaremos
    polyfeat = PolynomialFeatures().set_output(transform="pandas")
    more_features.append( ('addPolyFeat', polyfeat) )
    return Pipeline(more_features)
def model()->Pipeline:

    kernel = 'rbf' #<- 'linear', 'poly', 'rbf', 'sigmoid'
    C = 0.01
    degree = 1

    return SVC(kernel=kernel, degree=degree, C=C)

def adaboost()->Pipeline:
    n_estimators = 50
    learning_rate= .1
    tree_clf = DecisionTreeClassifier(max_depth=1)
    return AdaBoostClassifier(tree_clf, n_estimators=n_estimators, algorithm="SAMME.R", learning_rate=learning_rate)

#Ejecución del código
if __name__ == "__main__":
   
    #Lectura de los datos
    X1,y = lecturaDatos("../traintabs/traintab01.csv","../traintabs/train_label.csv")
    
    X2,y = lecturaDatos("../traintabs/traintab02.csv","../traintabs/train_label.csv")

    X3,y = lecturaDatos("../traintabs/traintab03.csv","../traintabs/train_label.csv")

    X4,y = lecturaDatos("../traintabs/traintab04.csv","../traintabs/train_label.csv")

    X5,y = lecturaDatos("../traintabs/traintab05.csv","../traintabs/train_label.csv")

    X6,y = lecturaDatos("../traintabs/traintab06.csv","../traintabs/train_label.csv")

    X7,y = lecturaDatos("../traintabs/traintab07.csv","../traintabs/train_label.csv")

    X8,y = lecturaDatos("../traintabs/traintab08.csv","../traintabs/train_label.csv")

    X = pd.concat([X1,X2,X3,X4,X5,X6,X7,X8],axis=1)

    # Dividimos los datos en conjuntos de entrenamiento y prueba
    # Porcentaje de la muestra reservada para testear
    test_size = 0.2
    # Semilla
    random_state = random.randint(1,10000)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    #####CREACIÓN DEL PIPELINE
    pipeline = Pipeline([  
                        ("normalize", filteringFeatures())
                        ,("add_features",addingFeatures())
                        ,("model",model())
                        ])

    # Definimos el espacio de búsqueda de hiperparámetros

    #Gradient Boosting
    '''param_grid = {
        'model__max_depth': [1,2],  
        'model__n_estimators': [10],  
        'model__learning_rate': [ 0.01,0.1,1 ]  
    }'''
    #SVM
    param_grid = { 
        'model__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  #5 y 0.1  
        'model__degree': [1,2]  
    }
    # Crear un objeto KFold para especificar la estrategia K-fold
    kf = KFold(n_splits=4, shuffle=True, random_state=random_state)  # Puedes ajustar n_splits según tu preferencia

    # Creamos un objeto GridSearchCV para el pipeline

    grid_search = GridSearchCV(pipeline, param_grid, cv=kf,verbose=2 ,n_jobs=4) #usar cv=kf , con njobs paralelizamos las operaciones
    # Realizamos la búsqueda de hiperparámetros utilizando los datos de entrenamiento
    grid_search.fit(X_train, y_train.values.ravel())
    # Mostramos los mejores hiperparámetros encontrados
    print("Mejores hiperparámetros encontrados:")
    print(grid_search.best_params_)

    # Evaluamos el mejor modelo en los datos de prueba
    best_model = grid_search.best_estimator_
    accuracy = best_model.score(X_test, y_test)
    print(f"Precisión del modelo con los mejores hiperparámetros en datos de prueba: {accuracy:.2f}")
    guardar_en_csv(describe_pipeline(pipeline),grid_search.best_params_,accuracy=accuracy)




