

import argparse
import random
import joblib
import numpy as np
import pandas as pd


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, cross_validate
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_classif
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.manifold import LocallyLinearEmbedding, Isomap, TSNE
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron
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



from sklearnex import patch_sklearn
patch_sklearn()#optimizacion intel cpus

from sklearn.svm import LinearSVC, NuSVC, LinearSVR, NuSVR, SVC
from sklearn.decomposition import PCA
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


    #Quitamos la columna de´nombre de la foto que no nos da ninguna info
    X_train_init = X_init.iloc[:, 1:]
    return X_train_init
def lecturaTags(path_tags:str):
    
    #Leemos el CSV de entrenamiento
    return pd.read_csv(path_tags, sep = ',', decimal = '.', index_col=0)

# Clase para seleccionar automáticamente el número óptimo de componentes con PCA
class OptimalPCA(TransformerMixin, BaseEstimator):
    '''
    Esta clase nos calcula el numero de componentes del PCA de acuerdo
    a los datos insertados, queremos considerar los componentes que nos da una cierta varianza explicada.
    Es importante no seleccionar una varianza explicada objetivo no muy alta, porque perderíamos generalización
    en nuestro modelo. Se han considerado valores mayores a 0.95 y aunque en ciertas etapas daba buenos resultados,
    del orden de 0.64 de accuracy, proporcionaba sobreajuste y en el conjunto de test bajaba su accuracy a 0.6.

    Mantiendo el target_variance en 0.9 se han obtenido los mejores resultados.

    '''
    def __init__(self, target_variance=0.9):
        self.target_variance = target_variance
        self.pca = PCA()

    def fit(self, X, y=None):
        self.pca.fit(X)
        #Sumam
        cumulative_variance_ratio = np.cumsum(self.pca.explained_variance_ratio_)
        self.n_components_ = np.argmax(cumulative_variance_ratio >= self.target_variance) + 1
        #self.n_components_ = 100
        #print(self.n_components_)
        return self

    def transform(self, X):
        return self.pca.transform(X)[:, :self.n_components_]
    


def normalizing_and_reducing()->Pipeline:
    '''
    Normali
    '''
    filters_features = []
    # Normalizado
    minmax = MinMaxScaler().set_output(transform='pandas')
    filters_features.append( ('minmax', minmax) )


    var_th = 0.01
    var_thrs = VarianceThreshold(var_th).set_output(transform='pandas')
    filters_features.append(('var_thrs', var_thrs))
    
    return Pipeline(filters_features)

# Función para aumentar caracteristicas
def feature_augmentation(X):
    """
    Realiza aumento de características

    Parametros:
    - data: numpy array o pandas DataFrame, el conjunto de datos original.

    Devuelve:
    - augmented_data: numpy array o pandas DataFrame, el conjunto de datos aumentado.
    """
    # Aplicar logaritmo a las características
    log_transformed_data = np.log1p(X)

    # Aumento de características (puedes personalizar esta parte según tus necesidades)
    #squared_data = np.square(X) este me viene dado por polyfeatures ya
    #cubed_data = np.power(X, 3)
    #sqrt_data = np.sqrt(np.abs(X))

    # Binneamos
    # Aplicar agrupamiento a cada característica
    #binned_data = np.apply_along_axis(lambda x: np.digitize(x, np.linspace(min(x), max(x), num=5)), axis=0, arr=X)

    # Derivadas
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
    
    #Aumentamos caracteristicas (esta comentado porque todos los aumetnados de caracteristicas que hemos probado, salvo el Poly nos reducen la precisión)
    #feature_transformer = FunctionTransformer(feature_augmentation) 
    #more_features.append( ('feature_augmentation', feature_transformer) )
    
    # Definir el imputador KNN (regulariza los datos mucho y no es deseable, solo lo usamos si nos da NaN algun aumentado de caracteristicas)
    #imputer = KNNImputer()
    #more_features.append(('nan_imputer',imputer))

    #Operaciones en paralelo
    return FeatureUnion(more_features)

def reduccionCaracteristicas():
    reducir = []
    pca = OptimalPCA()
    reducir.append( ('OptimalPCA', pca) ) #OptimalPCA nos busca cuantas componentes necesitamos de acuerdo a una varianza_objetivo
    return Pipeline(reducir)

def grid_testing(pipeline,params,X,y,X_test,y_test):

    # Crear un objeto KFold para especificar la estrategia K-fold
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1337)  # Puedes ajustar n_splits según lo que necesitemos

    # Creamos un objeto GridSearchCV para el pipeline
    grid_search = GridSearchCV(pipeline, params, cv=kf,verbose=3 ,n_jobs=1) #usar cv=kf , con njobs paralelizamos las operaciones, usamos 1 aunque se puede aumentar
    
    # Realizamos la búsqueda de hiperparámetros utilizando los datos de entrenamiento, los hiperparámetros se encuentran en params dentro del objeto gridsearch
    grid_search.fit(X, y.values.ravel())
    
    # Mostramos los mejores hiperparámetros encontrados
    print("Mejores hiperparámetros encontrados:")
    print(grid_search.best_params_)

    # Evaluamos nuestro modelo con el conjunto de prueba que hemos apartado al principio 
    accuracy = grid_search.best_estimator_.score(X_test, y_test)

    print(f"Precisión del modelo con los mejores hiperparámetros en datos de prueba: {accuracy:.5f}")

    #Guardamos el modelo para no perder el entrenamiento
    joblib.dump(grid_search.best_estimator_, 'modelo_guardado.joblib')

    try:
        guardar_en_csv(describe_pipeline(pipeline),grid_search.best_params_,accuracy=grid_search.best_estimator_.score(X_test, y_test))
    except:
        guardar_en_csv("error_descripcion_pipeline",grid_search.best_params_,accuracy=grid_search.best_estimator_.score(X_test, y_test))
    
    #Devolvemos el modelo entrenado

    return grid_search.best_estimator_


#Ejecución del código
if __name__ == "__main__":
   
    #Lectura de los datos
    X1 = lecturaDatos("../traintabs/traintab01.csv")
    
    X2 = lecturaDatos("../traintabs/traintab02.csv")

    X3 = lecturaDatos("../traintabs/traintab03.csv")

    X4 = lecturaDatos("../traintabs/traintab04.csv")

    X5 = lecturaDatos("../traintabs/traintab05.csv")

    X6 = lecturaDatos("../traintabs/traintab06.csv")

    X7 = lecturaDatos("../traintabs/traintab07.csv")

    X8 = lecturaDatos("../traintabs/traintab08.csv")

    X = pd.concat([X1,X2,X3,X4,X5,X6,X7,X8],axis=1)

    # Dividimos los datos en conjuntos de entrenamiento y prueba
    # Porcentaje de la muestra reservada para testear
    test_size = 0.2
    # Semilla
    random_state = 1338
    y = lecturaTags("../traintabs/train_label.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)


    ###CLASIFICADOR ELEGIDO
    # Opciones
    option = 'bagging' # 'voting','svm','adaboost','rf'

    if option == 'voting':
        # Definir el modelo de votación
        clf = VotingClassifier(
            estimators=[
                        ('dt', DecisionTreeClassifier(random_state=1337)), 
                        ('svm', SVC(probability=True, random_state=1337)) 
                        #('knn', KNeighborsClassifier()),
                        #('lda', LinearDiscriminantAnalysis()),
                        #('lr', LogisticRegression(random_state=1337))
                        ],
            voting='soft'
        )
        ###HIPERPARAMETROS
        params_grid = { 
            # Hiperparámetros SVC
            'model__svm__kernel': ['rbf'],  #5 y 0.1  
            'model__svm__C' : [0,1,1,10],
            # Hiperparámetros DT
            'model__dt__min_samples_split': [2,4],
            'model__dt__min_samples_leaf': [1,2],
            # Hiperparámetros Voting
            'model__voting': ['soft','hard'],
         }

    elif option == 'bagging':
        # Crear el BaggingClassifier con SVC estimator
        clf = BaggingClassifier(estimator=SVC(cache_size=4000,probability=True, random_state=1337), n_estimators=10, random_state=1337)

        ####HIPERPARÁMETROS
        params_grid = {
            # Hiperparámetros SVC
            'model__estimator__kernel': ['rbf'],  # 'linear', 'poly' 
            'model__estimator__C' : [0.01,0.5,1,10], #Ajustamos cuanto afecta el Slack
            # Hiperparámetros Bagging
            'model__n_estimators': [20,50,100] #Cuantos estimadores creamos

        }
    elif option == 'SVM':
        #Clasificador SVM, seleccionamos el kernel al ajustar hiperparametros
        clf = SVC(cache_size=4000,probability=True, random_state=1337)

        ####HIPERPARÁMETROS
        params_grid = { 
            # Hiperparámetros SVC
            'model__kernel': ['linear'],  #5 y 0.1  
            'model__C' : [0.05],  #5 y 0.1  
        }
    elif option=='rf':
        clf = RandomForestClassifier()
        ###HIPERPARAMETROS
        params_grid = {
            'model__n_estimators': [200],
            'model__max_depth': [None],
            'model__min_samples_split': [5],
            'model__min_samples_leaf': [ 2]
        }
    elif option == 'adaboost':
        clf = AdaBoostClassifier()
        ###HIPERPARAMETROS
        params = { 
            'model __estimator': [DecisionTreeClassifier(max_depth=1),SVC(cache_size=4000,probability=True, random_state=1337)],
            'model__n_estimators': [50,100,500],
            'model__learning_rate':[0.1,0.01,1],
        }


    #####CREACIÓN DEL PIPELINE
    pipeline = Pipeline([  
                         ("normalize", normalizing_and_reducing()) #Normalizamos y filtramos por la varianza
                        ,("add_features",addingFeatures()) #Hacemos aumentado de datos
                        ,("reducir_dimension",reduccionCaracteristicas()) #Usamos PCA para reducir la dimensionalidad
                        ,("model",clf) #Usamos el clasificador seleccionado anteriormente
                        ])
    
    modelo = grid_testing(pipeline,params_grid,X_train,y_train,X_test,y_test)


    #GENERAMOS EL FICHERO COMPETICION DE METEMOS LOS PATHS
    data = [lecturaDatos(i) for i in ["../testtabs/testtab01.csv",
                                                     "../testtabs/testtab02.csv",
                                                     "../testtabs/testtab03.csv",
                                                     "../testtabs/testtab04.csv",
                                                     "../testtabs/testtab05.csv",
                                                     "../testtabs/testtab06.csv",
                                                     "../testtabs/testtab07.csv",
                                                     "../testtabs/testtab08.csv"
                                                     ]
            ]
    TEST = pd.concat(data,axis=1)
    y_pred = modelo.preditc(TEST)
    np.savetxt('Competicion1.txt', y_pred, fmt='%i', delimiter=',')




