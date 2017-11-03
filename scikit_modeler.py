import pandas as pd
import numpy as np


from sklearn.base import BaseEstimator, TransformerMixin

#########################
#transformers to select a specific column from the data
class TextSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]
    
class NumberSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]



#################################33
#All variable transformation pipeline wrappers 

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

def tf_pipe(column, stopwords = None, one_char = False):
    if one_char:
        return Pipeline([
                ('selector', TextSelector(key=column)),
                ('tfidf', TfidfVectorizer(stop_words = stopwords, token_pattern=u'(?u)\\b\\w+\\b')),
        ])
    else:
        return Pipeline([
                ('selector', TextSelector(key=column)),
                ('tfidf', TfidfVectorizer(stop_words = stopwords))
        ])

from sklearn.preprocessing import StandardScaler
def scaling_pipe(column):
    return Pipeline([
                ('selector', NumberSelector(key=column)),
                ('standard', StandardScaler()),
            ])

def number_plain_pipe(column):
    return Pipeline([
                ('selector', NumberSelector(key=column))
            ])

from sklearn.preprocessing import MinMaxScaler
def minmax_pipe(column):
    return Pipeline([
                ('selector', NumberSelector(key=column)),
                ('standard', MinMaxScaler()),
            ])



##############################
# All modeling as a variable tranformers

from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder

# some require a dense matrix, must add in this transformer
class DenseTransformer(BaseEstimator, TransformerMixin):

    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self
    

class ModelTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, model):
        self.model = model

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self

    def transform(self, X, **transform_params):
        return DataFrame(self.model.predict(X))

def kmeans_pipe(column, clusters, oneHot = True):
    if oneHot:
        return(Pipeline([
                ('selector', TextSelector(key=column)),
                ('tfidf', TfidfVectorizer( stop_words='english')),
                ('means', ModelTransformer(KMeans(n_clusters=clusters))),
                ('encode', OneHotEncoder())
            ])
          )
    else:
        return(Pipeline([
                ('selector', TextSelector(key=column)),
                ('tfidf', TfidfVectorizer( stop_words='english')),
                ('means', ModelTransformer(KMeans(n_clusters=clusters)))
            ])
          )






##########################
# Library main objects
def variable_pipeline(variable, transformation, vartype = None):
    if transformation == 'standard_scale':
        return scaling_pipe(variable)
    elif transformation == 'minmax_scale':
        return minmax_pipe(variable)
    elif transformation == 'tfidf':
        return tf_pipe(variable)
    elif transformation == 'selector':
        if vartype == 'number':
            return number_plain_pipe(variable)
            #elif vartype = 'text':
            #    self.pipeline = number_plain_pipe(variable)


from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import make_union

def feature_pipeline(features):
    return Pipeline([
            ('all_features',FeatureUnion([(t.steps[0][1].key + t.steps[1][0], t) for t in features]))])


def model_pipeline(features, model, dense = False):
    if dense:
         return Pipeline([
            ('features',features),
            ('dense', DenseTransformer()), 
            ('model', model)])
    else:
        return Pipeline([
            ('features',features),
            ('model', model)])
