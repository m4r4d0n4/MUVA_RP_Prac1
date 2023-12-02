import argparse
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
    
def lecturaDatos(path_train:str):   

    #Leemos el CSV de entrenamiento
    X_init = pd.read_csv(path_train, sep = ',', decimal = '.', index_col=0)
    #Quitamos la columna de´nombre de la foto que no nos da ninguna info
    X_train_init = X_init.iloc[:, 1:]
    
    return X_train_init
def returnPipeline(path:str):
    return joblib.load(path)


#PARSEAR PARA LA ENTRAD

def parse_arguments():
    parser = argparse.ArgumentParser(description='Config Parser')
    parser.add_argument('-f', '--files', required=True, help='Paths de los archivos con los datos a clasificar. Formato -> testtab01.csv,testtab02.csv,..,testab08.csv')
    parser.add_argument('-p', '--pipeline', required=True, help='Path del pipeline guardado. Formato -> pipeline.joblib')
    return parser.parse_args()

def load_data_and_evaluate(file_paths,pipeline_path):


    data = [lecturaDatos(i) for i in file_paths]
    X = pd.concat(data,axis=1)
    pipe = joblib.load(pipeline_path)
    y_pred = pipe.predict(X)
    np.savetxt('Competicion1.txt', y_pred, fmt='%i', delimiter=',')


#Output competicion
if __name__ == "__main__":

    args = parse_arguments()
    file_paths = args.files.split(',')
    pipeline_path = args.pipeline
    load_data_and_evaluate(file_paths=file_paths,pipeline_path=pipeline_path)

