import numpy as np
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score

def calc_quality_metrics(y_true, y_pred, y_scores):
    """
    Calculate quality metrics for anomaly detection.
    Args:
        y_true: numpy.array, shape = (n_samples, )
            True labels with values {0, 1}: 0 is for normal observations, 1 is for anomalies/attacks.
        y_pred: numpy.array, shape = (n_samples, )
            Predicted labels with values {0, 1}: 0 is for normal observations, 1 is for anomalies/attacks.
        y_scores: numpy.array, shape = (n_samples, )
            Predicted anomaly scores.

    Returns:
        metrics = [ROC AUC, Recall, Precision, F1]

    Example:
        calc_quality_metrics(y_true=[0, 0, 1, 1, 1],
                             y_pred=[0, 1, 1, 1, 1],
                             y_scores=[-0.1, 0.2, 1, 10, 12])
    """
    rocauc = roc_auc_score(y_true, y_scores)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return [rocauc, recall, precision, f1]


def point_adjustment(y_true, y_pred):
    """
    Apply point adjustment for predicted labels as described in https://arxiv.org/pdf/1802.03903 (Fig. 7).
    Args:
        y_true: numpy.array, shape = (n_samples, )
            True labels with values {0, 1}: 0 is for normal observations, 1 is for anomalies/attacks.
        y_pred: numpy.array, shape = (n_samples, )
            Predicted labels with values {0, 1}: 0 is for normal observations, 1 is for anomalies/attacks.

    Returns:
        y_pred_pa: numpy.array, shape = (n_samples, )
            Adjusted predicted labels with values {0, 1}: 0 is for normal observations, 1 is for anomalies/attacks.
    """
    if len(y_true) != len(y_pred):
        raise Exception("y_true and y_pred must have the same length.")
    y_pred_pa = np.copy(y_pred)

    # find all segmetns of 1
    seg_start = None
    seg_end = None
    segment_inds = []
    for i in range(len(y_true)):
        if y_true[i] == 1:
            if seg_start is None:
                seg_start = i
        elif y_true[i] == 0:
            if seg_start is not None:
                seg_end = i
                segment_inds.append([seg_start, seg_end])
            seg_start = None
            seg_end = None

    # adjust predictions
    for aseg in segment_inds:
        if np.sum(y_pred[aseg[0]:aseg[1]]) > 0:
            y_pred_pa[aseg[0]:aseg[1]] = 1

    return y_pred_pa


def best_quality_metrics(y_true, y_scores, use_point_adjustment=True,
                         min_anomaly_rate=0.001, max_anomaly_rate=1.0, step=0.01):
    """
    Select the best threshold for the quality metrics.
    Args:
        y_true: numpy.array, shape = (n_samples, )
            True labels with values {0, 1}: 0 is for normal observations, 1 is for anomalies/attacks.
        y_scores: numpy.array, shape = (n_samples, )
            Predicted anomaly scores.
        use_point_adjustment: boolean
            Apply point adjustment to the predicted labels or not. {True, False}
        min_anomaly_rate: float, [0, 1]
            The minimal anomaly rate in the prediction.
        max_anomaly_rate: float, [0, 1]
            The maximal anomaly rate in the prediction.
        step: float, [0, 1]
            Increasing step for the anomaly rate.

    Returns:
        best_metrics: [ROC AUC, Recall, Precision, F1]
            The best values of the qualuity metrics.
        best_thresh: float
            The best threshold for the predicted anomaly scores.

    """
    qs = np.arange(1 - max_anomaly_rate, 1 - min_anomaly_rate, step)
    best_f1 = -1
    best_metrics = None
    best_thresh = None
    for aq in qs:
        thresh = np.quantile(y_scores, aq)
        y_pred = 1 * (y_scores > thresh)
        if use_point_adjustment:
            y_pred = point_adjustment(y_true, y_pred)
        metrics = calc_quality_metrics(y_true, y_pred, y_scores)
        if metrics[3] > best_f1:
            best_f1 = metrics[3]
            best_metrics = metrics
            best_thresh = thresh
    return best_metrics, best_thresh
