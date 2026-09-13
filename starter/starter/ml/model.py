import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validate the trained model using precision, recall, and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def compute_model_metrics_on_slices(model, df, y_true, feature_name):
    """Return precision/recall/fbeta for each unique value in one feature."""
    metrics = {}

    if hasattr(df, "loc"):
        values = df[feature_name].unique()
        for value in values:
            mask = df[feature_name] == value
            subset_y = np.asarray(y_true[mask])
            subset_X = df.loc[mask, :].to_numpy()
            preds = inference(model, subset_X)
            precision, recall, fbeta = compute_model_metrics(subset_y, preds)
            metrics[str(value)] = {
                "precision": precision,
                "recall": recall,
                "fbeta": fbeta,
                "support": len(subset_y),
            }
        return metrics

    feature_values = df[feature_name]
    rows = [
        [df[column][index] for column in df]
        for index in range(len(feature_values))
    ]
    values = sorted(set(feature_values))

    for value in values:
        indices = [
            index for index, item in enumerate(feature_values) if item == value
        ]
        subset_y = np.asarray(y_true[indices])
        subset_X = np.asarray(rows)[indices]
        preds = inference(model, subset_X)
        precision, recall, fbeta = compute_model_metrics(subset_y, preds)
        metrics[str(value)] = {
            "precision": precision,
            "recall": recall,
            "fbeta": fbeta,
            "support": len(subset_y),
        }
    return metrics
