from typing import Any, cast

import numpy as np
import scipy.special
import sklearn.metrics

from .util import PredictionType, TaskType


def _get_labels_and_probs(
    prediction: np.ndarray,
    task_type: TaskType,
    prediction_type: PredictionType,
) -> tuple[np.ndarray, None | np.ndarray]:
    """Obtain labels and probabilities from raw predictions."""
    assert task_type in (TaskType.BINCLASS, TaskType.MULTICLASS)

    if prediction_type == PredictionType.LABELS:
        return prediction, None
    elif prediction_type == PredictionType.PROBS:
        probs = prediction
    elif prediction_type == PredictionType.LOGITS:
        probs = (
            scipy.special.expit(prediction)
            if task_type == TaskType.BINCLASS
            else scipy.special.softmax(prediction, axis=1)
        )
    else:
        raise ValueError(f'Unknown prediction type: {prediction_type}')

    assert probs is not None
    labels = np.round(probs) if task_type == TaskType.BINCLASS else probs.argmax(axis=1)
    return labels.astype(np.int64), probs


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str | TaskType,
    prediction_type: str | PredictionType,
) -> dict[str, Any]:
    task_type = TaskType(task_type)
    prediction_type = PredictionType(prediction_type)

    if task_type == TaskType.REGRESSION:
        assert prediction_type == PredictionType.LABELS
        result = {
            'rmse': float(sklearn.metrics.mean_squared_error(y_true, y_pred) ** 0.5),
            'mae': float(sklearn.metrics.mean_absolute_error(y_true, y_pred)),
            'r2': float(sklearn.metrics.r2_score(y_true, y_pred)),
        }

    else:
        assert prediction_type is not None
        labels, probs = _get_labels_and_probs(y_pred, task_type, prediction_type)
        result = cast(
            dict[str, Any],
            sklearn.metrics.classification_report(y_true, labels, output_dict=True),
        )
        if probs is not None:
            result['cross-entropy'] = sklearn.metrics.log_loss(y_true, probs)
        if task_type == TaskType.BINCLASS and probs is not None:
            result['roc-auc'] = sklearn.metrics.roc_auc_score(y_true, probs)

    return result
