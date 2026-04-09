import numpy as np
from src.models.model_factory import get_model


def test_model_training():
    model = get_model("logistic_regression")

    X = np.random.rand(50, 5)
    y = np.random.randint(0, 2, 50)

    model.fit(X, y)
    preds = model.predict(X)

    assert preds.shape[0] == 50
