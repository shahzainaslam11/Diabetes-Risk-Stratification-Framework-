from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def get_model(model_name: str):
    """
    returns model based on name.
    """

    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=1000)

    elif model_name == "random_forest":
        return RandomForestClassifier()

    elif model_name == "decision_tree":
        return DecisionTreeClassifier()

    else:
        raise ValueError(f"Model {model_name} not supported.")
