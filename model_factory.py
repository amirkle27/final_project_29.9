from typing import Any, Dict, Tuple
from processing_facade import (
    LogisticRegressionFacade, DecisionTreeClassifierFacade, RandomForestClassifierFacade,
    KNNFacade, ANNFacade, LinearRegressionFacade, PolynomialFacade, SVMClassifierFacade,
    SVMRegressorFacade, XGBClassifierFacade, XGBRegressorFacade
)
from preprocessoring_strategy import PolynomialRegressionPreprocessor

class ModelFactory:
    """
    Factory for building ML Facade instances.

    Given a model_name and optional hyperparameters, constructs the appropriate
    facade and returns a tuple (facade, kind), where kind is either
    "classification" or "regression".

    Supported models:
      - Classification: 'logreg', 'dt', 'rf', 'knn', 'ann'
      - Regression:     'linear', 'poly' (requires 'degree', defaults to 2)

    Notes
    -----
    - Unsupported hyperparameter keys are ignored safely.
    - 'poly' is a special case: it wires a PolynomialRegressionPreprocessor
      and uses the provided 'degree' (or 2 by default).
    """

    CLASSIFIERS = {
        "logreg": (LogisticRegressionFacade, ["solver", "penalty", "C", "max_iter", "test_size", "random_state"]),
        "dt":     (DecisionTreeClassifierFacade, ["test_size", "random_state"]),
        "rf":     (RandomForestClassifierFacade, ["criterion", "test_size", "random_state"]),
        "knn":    (KNNFacade, ["n_neighbors", "test_size", "random_state"]),
        "ann":    (ANNFacade, ["hidden_layers", "epochs", "batch_size", "test_size", "random_state"]),
        "svm":    (SVMClassifierFacade, ["C","kernel","gamma","degree","test_size","random_state"]),
        "xgb": (XGBClassifierFacade, ["n_estimators","max_depth","learning_rate","test_size","random_state"])
    }

    REGRESSORS = {
        "linear": (LinearRegressionFacade, ["test_size", "random_state"]),
        "poly":   ("_poly", ["degree"]),
        "svr":    (SVMRegressorFacade, ["C","kernel","gamma","degree","epsilon","test_size","random_state"]),
        "xgbr": (XGBRegressorFacade, ["n_estimators","max_depth","learning_rate","test_size","random_state"])

    }

    @staticmethod
    def _filter_params(allowed_params: list[str], input_params: Dict[str, Any]) -> Dict[str, Any]:
        return {k: input_params[k] for k in allowed_params if k in input_params}

    @classmethod
    def create(cls, model_name: str, input_params: Dict[str, Any] | None = None) -> Tuple[object, str]:
        params = input_params or {}
        if model_name in cls.CLASSIFIERS:
            Facade, allowed_params = cls.CLASSIFIERS[model_name]
            kwargs = cls._filter_params(allowed_params, params)
            return Facade(**kwargs), "classification"

        # Regression
        if model_name in cls.REGRESSORS:
            spec, allowed_params = cls.REGRESSORS[model_name]
            if spec == "_poly":
                degree = int(params.get("degree", 2))
                return PolynomialFacade(PolynomialRegressionPreprocessor(), degree), "regression"
            Facade = spec
            kwargs = cls._filter_params(allowed_params, params)
            return Facade(**kwargs), "regression"

        raise ValueError(f"Unknown model '{model_name}'")
