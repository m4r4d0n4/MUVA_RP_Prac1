

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

    #Leemos el CSV de entrenamiento
    Y_train_init = pd.read_csv(path_tags, sep = ',', decimal = '.', index_col=0)

    #Quitamos la columna de´nombre de la foto que no nos da ninguna info
    X_train_init = X_init.iloc[:, 1:]
    return X_train_init,Y_train_init
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
    #feature_transformer = FunctionTransformer(feature_augmentation)
    #more_features.append( ('feature_augmentation', feature_transformer) )
    
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
    random_state = 1338

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)


    ###CLASIFICADOR ELEGIDO
    # Definir el modelo de votación
    voting_clf = VotingClassifier(
        estimators=[
                    ('dt', DecisionTreeClassifier(random_state=1337)), 
                    ('svm', SVC(probability=True, random_state=1337)) 
                    #('knn', KNeighborsClassifier()),
                    #('lda', LinearDiscriminantAnalysis()),
                    #('lr', LogisticRegression(random_state=1337))
                    ],
        voting='soft'
    )
    # Crear el BaggingClassifier con SVC como base_estimator
    bagging_classifier = BaggingClassifier(estimator=SVC(cache_size=4000,probability=True, random_state=1337), n_estimators=10, random_state=1337)


    #####CREACIÓN DEL PIPELINE
    pipeline_voting = Pipeline([  
                        ("normalize", filteringFeatures())
                        ,("add_features",addingFeatures())
                        ,("reducir_dimension",reduccionCaracteristicas())
                        ,("model",voting_clf)
                        ])
    #####CREACIÓN DEL PIPELINE
    pipeline_bagging = Pipeline([  
                        ("normalize", filteringFeatures())
                        ,("add_features",addingFeatures())
                        ,("reducir_dimension",reduccionCaracteristicas())
                        ,("model",bagging_classifier)
                        ])
    pipeline_svm = Pipeline([  
                        ("normalize", filteringFeatures())
                        ,("add_features",addingFeatures())
                        ,("reducir_dimension",reduccionCaracteristicas())
                        ,("model",SVC(cache_size=4000,probability=True, random_state=1337))
                        ])
    ####HIPERPARÁMETROS
    voting_params_grid = { 
        # Hiperparámetros SVC
        'model__svm__kernel': ['rbf'],  #5 y 0.1  
        'model__svm__C' : [10],
        # Hiperparámetros DT
        'model__dt__min_samples_split': [2,4],
        'model__dt__min_samples_leaf': [1,2],
        # Hiperparámetros Voting
        'model__voting': ['soft','hard'],

    }

    ####HIPERPARÁMETROS
    bagging_params_grid = { 
        # PCA

        # Hiperparámetros SVC
        'model__estimator__kernel': ['rbf'],  #5 y 0.1  
        'model__estimator__C' : [10],  #5 y 0.1  
        # Hiperparámetros Voting
        'model__n_estimators': [100]

    }
    svm_params_grid = { 
        # Hiperparámetros SVC
        'model__kernel': ['linear'],  #5 y 0.1  
        'model__C' : [0.05],  #5 y 0.1  

    }

    #grid_testing(pipeline_voting,voting_params_grid,X_train,y_train,X_test,y_test)
    grid_testing(pipeline_bagging,bagging_params_grid,X_train,y_train,X_test,y_test)
    #grid_testing(pipeline_svm,svm_params_grid,X_train,y_train,X_test,y_test)
