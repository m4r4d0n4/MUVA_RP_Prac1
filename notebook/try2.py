
import random
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, cross_validate
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
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.manifold import LocallyLinearEmbedding, Isomap, TSNE
from datetime import datetime
import csv

from load_model import returnPipeline
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
# Clase para seleccionar automáticamente el número óptimo de componentes con PCA
class OptimalPCA(TransformerMixin, BaseEstimator):
    def __init__(self, target_variance=0.95):
        self.target_variance = target_variance
        self.pca = PCA()

    def fit(self, X, y=None):
        self.pca.fit(X)
        cumulative_variance_ratio = np.cumsum(self.pca.explained_variance_ratio_)
        self.n_components_ = np.argmax(cumulative_variance_ratio >= self.target_variance) + 1
        print(self.n_components_)
        return self

    def transform(self, X):
        return self.pca.transform(X)[:, :self.n_components_]
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
# Función para aumentar caracteristicas
def feature_augmentation(X):
    """
    Realiza aumento de características y aplica el logaritmo a un conjunto de datos.

    Parameters:
    - data: numpy array o pandas DataFrame, el conjunto de datos original.

    Returns:
    - augmented_data: numpy array o pandas DataFrame, el conjunto de datos aumentado con logaritmo aplicado.
    """
    # Aplicar logaritmo a las características
    log_transformed_data = np.log1p(X)

    # Aumento de características (puedes personalizar esta parte según tus necesidades)
    #squared_data = np.square(X) este me viene dado por polyfeatures ya
    #cubed_data = np.power(X, 3)
    #sqrt_data = np.sqrt(np.abs(X))

    #Binneamos
    # Aplicar agrupamiento a cada característica
    #binned_data = np.apply_along_axis(lambda x: np.digitize(x, np.linspace(min(x), max(x), num=5)), axis=0, arr=X)
    #Derivadas
    #diff_data = np.diff(X, axis=1)
    #derivative_data = np.gradient(X, axis=1)

    # Concatenar las características originales con las nuevas características aumentadas
    augmented_data = np.concatenate((X,log_transformed_data), axis=1)

    return augmented_data

def addingFeatures()->Pipeline:
    more_features = []
    #degree es la variable que editaremos
    polyfeat = PolynomialFeatures().set_output(transform="pandas")
    more_features.append( ('addPolyFeat', polyfeat) )
    
    #Aumentamos caracteristicas
    feature_transformer = FunctionTransformer(feature_augmentation)
    more_features.append( ('feature_augmentation', feature_transformer) )
    
    # Definir el imputador KNN
    #imputer = KNNImputer()
    #more_features.append(('nan_imputer',imputer))

    #Operaciones en paralelo
    return FeatureUnion(more_features)

def reduccionCaracteristicas():
    reducir = []
    pca = OptimalPCA()
    reducir.append( ('OptimalPCA', pca) )
    return Pipeline(reducir)

def svc()->Pipeline:
    kernel = 'rbf' #<- 'linear', 'poly', 'rbf', 'sigmoid'
    C = 0.01
    degree = 3
    return SVC(kernel=kernel, probability=True,degree=degree, C=C,cache_size=4000)
    #voting_system = 'soft'
    #return VotingClassifier( estimators=[('svm', svc_clf),  ('ada', adaboost())], voting=voting_system)


def nusvc()->Pipeline:
    kernel = 'rbf' #<- 'linear', 'poly', 'rbf', 'sigmoid'
    nu = 1
    degree = 3
    return NuSVC(kernel=kernel, degree=degree, nu=nu,cache_size=4000)
    #voting_system = 'soft'
    #return VotingClassifier( estimators=[('svm', svc_clf),  ('ada', adaboost())], voting=voting_system)

def adaboost()->Pipeline:
    n_estimators = 50
    learning_rate= .1
    tree_clf = DecisionTreeClassifier(max_depth=1)
    return AdaBoostClassifier(tree_clf, n_estimators=n_estimators, algorithm="SAMME.R", learning_rate=learning_rate)


def grid_testing(pipeline,params,X,y,X_test,y_test):

    # Crear un objeto KFold para especificar la estrategia K-fold
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1337)  # Puedes ajustar n_splits según tu preferencia

    # Creamos un objeto GridSearchCV para el pipeline

    grid_search = GridSearchCV(pipeline, params, cv=kf,verbose=3 ,n_jobs=1) #usar cv=kf , con njobs paralelizamos las operaciones
    # Realizamos la búsqueda de hiperparámetros utilizando los datos de entrenamiento
    grid_search.fit(X, y.values.ravel())
    #grid_search_ada.fit(X_train, y_train.values.ravel())
    
    # Mostramos los mejores hiperparámetros encontrados
    print("Mejores hiperparámetros encontrados:")
    print(grid_search.best_params_)

    accuracy = grid_search.best_estimator_.score(X_test, y_test)
    print(f"Precisión del modelo con los mejores hiperparámetros voting 1 en datos de prueba: {accuracy:.5f}")

    joblib.dump(grid_search.best_estimator_, 'aaa.joblib')
    try:
        guardar_en_csv(describe_pipeline(pipeline),grid_search.best_params_,accuracy=grid_search.best_estimator_.score(X_test, y_test))
    except:
        guardar_en_csv("voting",grid_search.best_params_,accuracy=grid_search.best_estimator_.score(X_test, y_test))
    




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
    random_state = 1337

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    #####CREACIÓN DEL PIPELINE
    pipeline_svc = Pipeline([  
                        ("normalize", filteringFeatures())
                        ,("add_features",addingFeatures())
                        ,("reducir_dimension",reduccionCaracteristicas())
                        ,("model",svc())
                        ])
    pipeline_adaboost = Pipeline([  
                        ("normalize", filteringFeatures())
                        ,("add_features",addingFeatures())
                        ,("reducir_dimension",reduccionCaracteristicas())
                        ,("model",adaboost())
                        ])

    # Definimos el espacio de búsqueda de hiperparámetros

    #Gradient Boosting
    '''param_grid = {
        'model__max_depth': [1,2],  
        'model__n_estimators': [10],  
        'model__learning_rate': [ 0.01,0.1,1 ]  
    }'''
    #SVM
    n_estimators = 50
    learning_rate= .1
    '''params_grid_svc ={
        'model__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  #5 y 0.1  
        'model__degree': [5],
        'model__C' : [1,0.05,0.1]}'''
    params_grid_svc ={
        'model__kernel': ['linear'],  #5 y 0.1  
        'model__degree': [5],
        'model__C' : [0.1]}
    params_grid_adaboost = { 
        'model__n_estimators': [800],
        'model__learning_rate':[0.1,0.01,1],
        
    }
    param_grid_rf = {
    'model__n_estimators': [200],
    'model__max_depth': [None],
    'model__min_samples_split': [5],
    'model__min_samples_leaf': [ 2]
    }
    pipeline_rf = Pipeline([  
                        ("normalize", filteringFeatures())
                        ,("add_features",addingFeatures())
                        ,("reducir_dimension",reduccionCaracteristicas())
                        ,("model",RandomForestClassifier())
                        ])
    
    #grid_testing(pipeline_svc,params_grid_svc,X_train,y_train,X_test,y_test)
    
    # Crear el VotingClassifier
    voting_classifier = VotingClassifier(
        estimators=[
            ('ada', returnPipeline("ada_0.6.joblib")),
            #('svc', returnPipeline("svcc_0.63.joblib")),
            ('rf', returnPipeline("rf_0.58.joblib"))
        ],
        voting='hard'  # 'hard' para voto mayoritario, 'soft' para voto ponderado
    )
    # Parámetros adicionales a ajustar para VotingClassifier
    voting_param_grid = {
        'voting': ['hard']
    }
    grid_testing(voting_classifier,voting_param_grid,X_train,y_train,X_test,y_test)















'''
    # Crear un objeto KFold para especificar la estrategia K-fold
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)  # Puedes ajustar n_splits según tu preferencia

    # Creamos un objeto GridSearchCV para el pipeline

    grid_search_svc = GridSearchCV(pipeline_svc, params_grid_svc, cv=kf,verbose=3 ,n_jobs=2) #usar cv=kf , con njobs paralelizamos las operaciones
    grid_search_ada = GridSearchCV(pipeline_adaboost, params_grid_adaboost, cv=kf,verbose=3 ,n_jobs=2) #usar cv=kf , con njobs paralelizamos las operaciones
    # Realizamos la búsqueda de hiperparámetros utilizando los datos de entrenamiento
    grid_search_svc.fit(X_train, y_train.values.ravel())
    #grid_search_ada.fit(X_train, y_train.values.ravel())
    
    # Mostramos los mejores hiperparámetros encontrados
    print("Mejores hiperparámetros encontrados:")
    print(grid_search_svc.best_params_)
   # print(grid_search_ada.best_params_)

    
    
    accuracy = grid_search_svc.best_estimator_.score(X_test, y_test)
    print(f"Precisión del modelo con los mejores hiperparámetros voting 1 en datos de prueba: {accuracy:.2f}")
    #accuracy = grid_search_ada.best_estimator_.score(X_test, y_test)
    #print(f"Precisión del modelo con los mejores hiperparámetros voting 2 en datos de prueba: {accuracy:.2f}")
    joblib.dump(grid_search_svc.best_estimator_, 'voting1_svc.joblib')
    #joblib.dump(grid_search_ada.best_estimator_, 'voting2_ada.joblib')
    
   # guardar_en_csv(describe_pipeline(pipeline_adaboost),grid_search_ada.best_params_,accuracy=grid_search_ada.best_estimator_.score(X_test, y_test))
    
    guardar_en_csv(describe_pipeline(pipeline_svc),grid_search_svc.best_params_,accuracy=grid_search_svc.best_estimator_.score(X_test, y_test))


    voting_classifier = BaggingClassifier(estimator=
    ('svc', grid_search_svc.best_estimator_), voting='soft')

    pipeline= Pipeline([  
                        ("normalize", filteringFeatures())
                        ,("add_features",addingFeatures())
                        ,("model",voting_classifier)
                        ])
    params_grid = { 
        'model__n_estimators': [10,100]
        
    }
    grid_search = GridSearchCV(pipeline_svc, params_grid, cv=kf,verbose=2 ,n_jobs=2) #usar cv=kf , con njobs paralelizamos las operaciones
    grid_search.fit(X_train, y_train.values.ravel())
    # Evaluamos el mejor modelo en los datos de prueba
    best_model = grid_search.best_estimator_
    accuracy = best_model.score(X_test, y_test)
    print(f"Precisión del modelo con los mejores hiperparámetros en datos de prueba: {accuracy:.2f}")
    guardar_en_csv(describe_pipeline(pipeline),grid_search.best_params_,accuracy=accuracy)
    # Guardar el modelo entrenado en un archivo usando joblib
    joblib.dump(best_model, 'mejor_modelo_entrenado.joblib')


    #Output competicion

    #Lectura de los datos
    Y1,y = lecturaDatos("../testtabs/testtab01.csv","../traintabs/train_label.csv")
    
    Y2,y = lecturaDatos("../testtabs/testtab02.csv","../traintabs/train_label.csv")

    Y3,y = lecturaDatos("../testtabs/testtab03.csv","../traintabs/train_label.csv")

    Y4,y = lecturaDatos("../testtabs/testtab04.csv","../traintabs/train_label.csv")

    Y5,y = lecturaDatos("../testtabs/testtab05.csv","../traintabs/train_label.csv")

    Y6,y = lecturaDatos("../testtabs/testtab06.csv","../traintabs/train_label.csv")

    Y7,y = lecturaDatos("../testtabs/testtab07.csv","../traintabs/train_label.csv")

    Y8,y = lecturaDatos("../testtabs/testtab08.csv","../traintabs/train_label.csv")

    Y = pd.concat([Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8],axis=1)

    y_pred = grid_search.predict(Y)

    np.savetxt('Competicion1.txt', y_pred, fmt='%i', delimiter=',')



'''