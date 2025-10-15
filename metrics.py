import tensorflow as tf

class BinaryIoU(tf.keras.metrics.Metric):
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

class DiceCoefficient(tf.keras.metrics.Metric):
    def __init__(self, target_class_id=1, threshold=0.5, name='dice_coefficient', **kwargs):
        super().__init__(name=name, **kwargs)
        self.target_class_id = target_class_id
        self.threshold = threshold
        self.intersection = self.add_weight(name='intersection', initializer='zeros')
        self.union = self.add_weight(name='union', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_class = tf.cast(y_true[..., self.target_class_id], tf.float32)
        y_pred_class = tf.cast(y_pred[..., self.target_class_id] > self.threshold, tf.float32)
        intersection = tf.reduce_sum(y_true_class * y_pred_class)
        union = tf.reduce_sum(y_true_class) + tf.reduce_sum(y_pred_class)
        self.intersection.assign_add(intersection)
        self.union.assign_add(union)

    def result(self):
        dice = tf.where(self.union > 0, 2.0 * self.intersection / (self.union + 1e-8), 0.0)
        return dice

    def reset_state(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)