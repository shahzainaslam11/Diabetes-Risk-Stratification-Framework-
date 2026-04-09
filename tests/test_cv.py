import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.training.cv import perform_5fold_cv_with_resampling


def test_cv_execution():
    X = pd.DataFrame(np.random.rand(100, 5))
    y = pd.Series(np.random.randint(0, 2, 100))

    model = LogisticRegression(max_iter=1000)

    results, best_fold = perform_5fold_cv_with_resampling(model, X, y)

    assert results.shape[0] == 5
    assert "accuracy" in results.columns
    assert 1 <= best_fold <= 5
