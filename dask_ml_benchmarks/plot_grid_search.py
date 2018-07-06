"""
.. _plot_grid_search.py

Grid search comparision
=======================

This example shows the comparison of Dask-ML's grid search and Scikit-Learn's
grid search. Both examples use have as many parallel jobs as the cores on the
host machine.

In this example, the Dask-ML grid search does do less work because it caches
``fit`` results. See the documentation for more detail:
https://dask-ml.readthedocs.io/en/latest/hyper-parameter-search.html
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.datasets
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

import dask_ml.model_selection
import sklearn.model_selection

from distributed import Client
from distributed.metrics import time

client = Client()

data = sklearn.datasets.fetch_20newsgroups_vectorized()
names = data.target_names

keep = ['sci.space', 'sci.med', 'alt.atheism',
        'soc.religion.christian', 'comp.graphics',
        'talk.religion.misc']
labels_keep = [i for i, name in enumerate(names) if name in keep]
idx_save = [i for i, label in enumerate(data.target)
            if label in labels_keep]

X = data.data[idx_save]
y = data.target[idx_save]

params = {'clf__alpha': np.logspace(-6, -3, num=4),
          'clf__loss': ['hinge', 'log', 'modified_huber'],
          'clf__average': [True, False],
          'clf__penalty': ['l1', 'l2'],
          'tfidf__norm': ('l1', 'l2'),
         }
est = Pipeline([
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(tol=1e-3))
])

data = []

start = time()
grid_sklearn = sklearn.model_selection.GridSearchCV(est, params, n_jobs=-1)
grid_sklearn.fit(X, y)
data += [{'library': 'scikit-learn',
          'time': time() - start,
          'best_score': grid_sklearn.best_score_}]

start = time()
grid_dask = dask_ml.model_selection.GridSearchCV(est, params)
grid_dask.fit(X, y)
data += [{'library': 'dask-ml',
          'time': time() - start,
          'best_score': grid_dask.best_score_}]


df = pd.DataFrame(data)
ax = df.plot.bar(x='library', y='time', legend=False)
ax.set_ylabel('Time (s)')
plt.xticks(rotation=0)
