"""
Tests de l'interface des modèles de prévision (stub).

Vérifie que la hiérarchie de classes est correcte et que les stubs
lèvent les bonnes exceptions.
"""

import pytest

from forecaster.predictors.base import BaseForecastModel, ModelNotLoadedError
from forecaster.predictors.consumption import ConsumptionModel
from forecaster.predictors.pv_production import PVProductionModel


def test_consumption_model_is_base_forecast_model():
    """ConsumptionModel doit hériter de BaseForecastModel."""
    assert issubclass(ConsumptionModel, BaseForecastModel)


def test_pv_production_model_is_base_forecast_model():
    """PVProductionModel doit hériter de BaseForecastModel."""
    assert issubclass(PVProductionModel, BaseForecastModel)


def test_consumption_model_predict_raises_not_loaded():
    """predict() doit lever ModelNotLoadedError si le modèle n'est pas chargé."""
    import pandas as pd
    model = ConsumptionModel(version="test")
    with pytest.raises(ModelNotLoadedError):
        model.predict(pd.DataFrame())


def test_pv_production_model_predict_raises_not_loaded():
    """predict() doit lever ModelNotLoadedError si le modèle n'est pas chargé."""
    import pandas as pd
    model = PVProductionModel(version="test")
    with pytest.raises(ModelNotLoadedError):
        model.predict(pd.DataFrame())


def test_pv_model_build_features_raises_not_implemented():
    """build_features() doit lever NotImplementedError (stub)."""
    import pandas as pd
    model = PVProductionModel(version="test")
    with pytest.raises(NotImplementedError):
        model.build_features(pd.DataFrame())
