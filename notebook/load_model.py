import joblib
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

# Clase para seleccionar automáticamente el número óptimo de componentes con PCA
class OptimalPCA(TransformerMixin, BaseEstimator):
    def __init__(self, target_variance=0.95):
        self.target_variance = target_variance
        self.pca = PCA()

    def fit(self, X, y=None):
        self.pca.fit(X)
        cumulative_variance_ratio = np.cumsum(self.pca.explained_variance_ratio_)
        self.n_components_ = np.argmax(cumulative_variance_ratio >= self.target_variance) + 1
        #self.n_components_ = 6
        print(self.n_components_)
        return self

    def transform(self, X):
        return self.pca.transform(X)[:, :self.n_components_]
    
def lecturaDatos(path_train:str, path_tags:str):   

    #Leemos el CSV de entrenamiento
    X_init = pd.read_csv(path_train, sep = ',', decimal = '.', index_col=0)

    #Leemos el CSV de entrenamiento
    Y_train_init = pd.read_csv(path_tags, sep = ',', decimal = '.', index_col=0)

    #Quitamos la columna de´nombre de la foto que no nos da ninguna info
    X_train_init = X_init.iloc[:, 1:]
    return X_train_init,Y_train_init
def returnPipeline(path:str):
    return joblib.load(path)

def test_model():
    #Test
    
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
    test_size = 0.98
    # Semilla
    random_state = 1337

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    pipe = joblib.load("bagging_0.6314.joblib")
    acc = pipe.score(X_test, y_test)
    print(acc)


#Output competicion
if __name__ == "__main__":
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
    pipe = joblib.load("bagging_0.6314.joblib")
    
    #test_model()
    y_pred = pipe.predict(Y)

    np.savetxt('Competicion1.txt', y_pred, fmt='%i', delimiter=',')

