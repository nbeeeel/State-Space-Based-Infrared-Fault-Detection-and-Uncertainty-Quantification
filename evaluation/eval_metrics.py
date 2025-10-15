import tensorflow as tf
import numpy as np
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score
)

class BinaryIoU(tf.keras.metrics.Metric):
    """Binary IoU metric"""
    def __init__(self, target_class_id=1, threshold=0.5, name='binary_iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.target_class_id = target_class_id
        self.threshold = threshold
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_class = tf.cast(y_true[..., self.target_class_id], tf.float32)
        y_pred_class = tf.cast(y_pred[..., self.target_class_id] > self.threshold, tf.float32)
        true_positives = tf.reduce_sum(y_true_class * y_pred_class)
        false_positives = tf.reduce_sum((1 - y_true_class) * y_pred_class)
        false_negatives = tf.reduce_sum(y_true_class * (1 - y_pred_class))
        self.true_positives.assign_add(true_positives)
        self.false_positives.assign_add(false_positives)
        self.false_negatives.assign_add(false_negatives)

    def result(self):
        denominator = self.true_positives + self.false_positives + self.false_negatives
        iou = tf.where(denominator > 0, self.true_positives / (denominator + 1e-8), 0.0)
        return iou

    def reset_state(self):
        self.true_positives.assign(0.0)
        self.false_positives.assign(0.0)
        self.false_negatives.assign(0.0)

def calculate_comprehensive_metrics(y_true, y_pred, threshold=0.5):
    """Calculate all evaluation metrics"""
    if hasattr(y_true, 'numpy'):
        y_true = y_true.numpy()
    if hasattr(y_pred, 'numpy'):
        y_pred = y_pred.numpy()
    y_true_binary = y_true[..., 1].flatten()
    y_pred_binary = (y_pred[..., 1].flatten() > threshold).astype(int)
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    mcc = matthews_corrcoef(y_true_binary, y_pred_binary)
    kappa = cohen_kappa_score(y_true_binary, y_pred_binary)
    balanced_acc = balanced_accuracy_score(y_true_binary, y_pred_binary)
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()
    ious = []
    for i in range(len(y_true)):
        true_i = y_true[i, ..., 1]
        pred_i = (y_pred[i, ..., 1] > threshold).astype(float)
        intersection = np.sum(true_i * pred_i)
        union = np.sum(true_i) + np.sum(pred_i) - intersection
        iou = intersection / (union + 1e-8)
        ious.append(iou)
    fpr, tpr, _ = roc_curve(y_true_binary, y_pred[..., 1].flatten())
    roc_auc = auc(fpr, tpr)
    prec_curve, rec_curve, _ = precision_recall_curve(y_true_binary, y_pred[..., 1].flatten())
    pr_auc = auc(rec_curve, prec_curve)
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'mcc': mcc,
        'kappa': kappa,
        'balanced_accuracy': balanced_acc,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'mean_iou': np.mean(ious),
        'std_iou': np.std(ious),
        'confusion_matrix': cm.tolist(),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
        'per_image_ious': ious
    }

def find_optimal_threshold(y_true, y_pred):
    """Find optimal prediction threshold"""
    if hasattr(y_true, 'numpy'):
        y_true = y_true.numpy()
    if hasattr(y_pred, 'numpy'):
        y_pred = y_pred.numpy()
    thresholds = np.arange(0.1, 0.95, 0.01)
    results = {'threshold': [], 'f1': [], 'iou': [], 'mcc': []}
    y_true_binary = y_true[..., 1]
    for thresh in thresholds:
        y_pred_binary = (y_pred[..., 1] > thresh).astype(float)
        f1 = f1_score(y_true_binary.flatten(), y_pred_binary.flatten(), zero_division=0)
        mcc = matthews_corrcoef(y_true_binary.flatten(), y_pred_binary.flatten())
        intersection = np.sum(y_true_binary * y_pred_binary)
        union = np.sum(y_true_binary) + np.sum(y_pred_binary) - intersection
        iou = intersection / (union + 1e-8)
        results['threshold'].append(thresh)
        results['f1'].append(f1)
        results['iou'].append(iou)
        results['mcc'].append(mcc)
    optimal_thresholds = {
        'f1': thresholds[np.argmax(results['f1'])],
        'iou': thresholds[np.argmax(results['iou'])],
        'mcc': thresholds[np.argmax(results['mcc'])]
    }
    return optimal_thresholds, results