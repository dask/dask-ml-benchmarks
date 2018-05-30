import dask
import distributed
import numpy as np

import dask_ml
from dask_ml.linear_model import PartialSGDClassifier


def make_classification_problem(n_samples, n_features, chunks,
                                n_informative=0.1, scale=1.0, noise=0.1,
                                seed=0):
    if n_informative < 1:
        n_informative = max(int(n_informative * n_features), 1)

    np_rng = np.random.RandomState(seed)
    dask_rng = dask_ml.utils.check_random_state(seed)

    X = dask_rng.normal(0, 1, size=(n_samples, n_features),
                        chunks=chunks).astype(np.float32)
    noise = dask_rng.normal(0, 1, size=n_samples,
                            chunks=chunks).astype(np.float32)
    informative_idx = np_rng.choice(n_features, n_informative)

    coef = np.zeros(n_features, dtype=np.float32)
    coef[informative_idx] = np_rng.normal(size=n_informative) * scale
    intercept = np_rng.normal() * scale

    y_pred = X[:, informative_idx].dot(coef[informative_idx])
    y = y_pred + noise > intercept
    y = y.astype(int)

    return coef, intercept, X, y


if __name__ == "__main__":
    c = distributed.Client()

    n_samples = int(1e6)
    n_features = int(1e2)
    chunks = int(0.1 * n_samples)

    true_coef, true_intercept, X, y = make_classification_problem(
        n_samples, n_features, chunks)
    print(f"Generating {X.nbytes / 1e9} GB of random data")
    X, y = dask.persist(X, y)
    distributed.wait([X, y])

    split = int(0.9 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print("Fitting model...")
    model = PartialSGDClassifier(loss='log', penalty='l2', alpha=1e-4,
                                 classes=[0, 1])
    model.fit(X_train, y_train, get=c.get)  # get to work around dask-ml-185
    print(f"accuracy: {model.score(X_test, y_test):0.3f}")
