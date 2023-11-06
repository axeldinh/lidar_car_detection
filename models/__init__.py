from models.pointnet import PointNetClassifier, PointNetRegressor
from models.pointnetpp import PointNetPPClassifier, PointNetPPRegressor

__regressor_models__ = {
    "pointnet": PointNetRegressor,
    "pointnetpp": PointNetPPRegressor,
}

__classifier_models__ = {
    "pointnet": PointNetClassifier,
    "pointnetpp": PointNetPPClassifier,
}


def string_to_model(string, regression=False):
    if regression:
        available_models = __regressor_models__
    else:
        available_models = __classifier_models__

    if string not in available_models:
        model_type = "regressor" if regression else "classifier"
        raise Exception(
            f"Model {string} not available. Available {model_type} models: {available_models.keys()}"
        )

    return available_models[string]


def get_model(params):
    model = string_to_model(string=params["model"], regression=params["regression"])
    params.pop("model")
    return model(**params)
