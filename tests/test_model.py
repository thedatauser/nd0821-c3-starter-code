import numpy as np

from starter.starter.ml.model import (
    compute_model_metrics,
    compute_model_metrics_on_slices,
    inference,
    train_model,
)


def test_train_model_returns_fitted_model():
    X = np.array([[25, 0], [30, 1], [40, 0], [45, 1]], dtype=float)
    y = np.array([0, 0, 1, 1])

    model = train_model(X, y)

    assert model is not None
    assert hasattr(model, 'predict')


def test_compute_model_metrics_returns_positive_rates():
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1


def test_inference_returns_predictions_for_input():
    X = np.array([[25, 0], [30, 1], [40, 0], [45, 1]], dtype=float)
    y = np.array([0, 0, 1, 1])
    model = train_model(X, y)

    preds = inference(model, X)

    assert preds.shape == (4,)
    assert set(np.unique(preds)).issubset({0, 1})


def test_compute_model_metrics_on_slices_returns_dict():
    X = np.array([[0, 1], [0, 1], [1, 0], [1, 0]], dtype=float)
    y = np.array([0, 0, 1, 1])
    model = train_model(X, y)
    df = {
        "feature_a": [0, 0, 1, 1],
        "feature_b": [1, 1, 0, 0],
    }

    metrics = compute_model_metrics_on_slices(model, df, y, "feature_a")

    assert isinstance(metrics, dict)
    assert "0" in metrics or 0 in metrics
