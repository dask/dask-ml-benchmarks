"""
.. _plot_grid_search.py

Grid search comparision
=======================

This example shows the comparison of Dask-ML's grid search and Scikit-Learn's
grid search, and is a modification of a Scikit-Learn example at
`"Sample pipeline for text feature extraction and evaluation"
<http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html>`_.

In this example, the Dask-ML grid search does less computation because it
caches ``fit`` results. See the documentation for a good illustration for this
benchmark:
https://dask-ml.readthedocs.io/en/latest/hyper-parameter-search.html.
"""
from pprint import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

import dask_ml.model_selection
import sklearn.model_selection

from distributed import Client
from distributed.metrics import time

client = Client()

categories = ["alt.atheism", "talk.religion.misc", "sci.space", "sci.med"]
data = sklearn.datasets.fetch_20newsgroups(
    subset="train", categories=categories
)

X = data.data
y = data.target
doc_lengths = [len(doc) for doc in X]
print(
    "Dataset has {} documents. Median length is {} words".format(
        len(data.filenames), np.median(doc_lengths)
    )
)
print("Document categories: {}".format(categories))

params = {
    "tfidf__norm": ["l1", "l2"],
    "tfidf__binary": [True, False],
    "clf__alpha": np.logspace(-6, -3, num=4),
    "clf__loss": ["hinge", "log", "modified_huber"],
    "clf__penalty": ["l1", "l2"],
}
pipeline = Pipeline(
    [
        ("tfidf", TfidfVectorizer()),
        ("clf", SGDClassifier(tol=1e-3)),
    ]
)

data = []

start = time()
grid_sklearn = sklearn.model_selection.GridSearchCV(
    pipeline, params, n_jobs=-1
)
grid_sklearn.fit(X, y)
data += [
    {
        "library": "scikit-learn",
        "time": time() - start,
        "best_score": grid_sklearn.best_score_,
        "best_params": grid_sklearn.best_params_,
    }
]

start = time()
grid_dask = dask_ml.model_selection.GridSearchCV(pipeline, params)
grid_dask.fit(X, y)
data += [
    {
        "library": "dask-ml",
        "time": time() - start,
        "best_score": grid_dask.best_score_,
        "best_params": grid_dask.best_params_,
    }
]
print("Best score: ", grid_dask.best_score_)
print("Best parameter set:")
pprint(grid_dask.best_params_)


df = pd.DataFrame(data)
ax = df.plot.bar(x="library", y="time", legend=False)
ax.set_ylabel("Time (s)")
plt.xticks(rotation=0)
