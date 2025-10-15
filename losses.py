import tensorflow as tf
import logging

def weighted_binary_crossentropy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    positive_pixels = tf.reduce_sum(y_true[..., 1])
    total_pixels = tf.cast(tf.reduce_prod(tf.shape(y_true)[:-1]), tf.float32)
    negative_pixels = total_pixels - positive_pixels
    pos_weight = tf.where(positive_pixels > 0, tf.sqrt(negative_pixels / positive_pixels), 1.0)
    neg_weight = tf.where(negative_pixels > 0, tf.sqrt(positive_pixels / negative_pixels), 1.0)
    pos_weight = tf.clip_by_value(pos_weight, 1.0, 10.0)
    neg_weight = tf.clip_by_value(neg_weight, 0.1, 1.0)
    weights = tf.where(y_true[..., 1] > 0.5, pos_weight, neg_weight)
    bce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    weighted_bce = bce * weights
    return tf.reduce_mean(weighted_bce)

def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_pos = y_true[..., 1]
    y_pred_pos = y_pred[..., 1]
    intersection = tf.reduce_sum(y_true_pos * y_pred_pos)
    union = tf.reduce_sum(y_true_pos) + tf.reduce_sum(y_pred_pos)
    dice = tf.where(union > 0, 2.0 * intersection / (union + 1e-8), 1.0)
    return 1.0 - dice

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-6, 1 - 1e-6)
    y_true_pos = y_true[..., 1]
    y_pred_pos = y_pred[..., 1]
    pt = tf.where(tf.equal(y_true_pos, 1), y_pred_pos, 1 - y_pred_pos)
    focal_weight = tf.pow(1 - pt, gamma)
    ce = -tf.math.log(tf.clip_by_value(pt, 1e-6, 1.0))
    loss = alpha * focal_weight * ce
    loss = tf.where(tf.math.is_finite(loss), loss, tf.zeros_like(loss))
    return tf.reduce_mean(loss)

def boundary_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
    sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tf.float32)
    sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])
    sobel_y = tf.reshape(sobel_y, [3, 3, 1, 1])
    true_input = tf.pad(y_true[..., 1:2], [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    pred_input = tf.pad(y_pred[..., 1:2], [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    true_boundaries_x = tf.nn.conv2d(true_input, sobel_x, strides=[1,1,1,1], padding='VALID')
    true_boundaries_y = tf.nn.conv2d(true_input, sobel_y, strides=[1,1,1,1], padding='VALID')
    pred_boundaries_x = tf.nn.conv2d(pred_input, sobel_x, strides=[1,1,1,1], padding='VALID')
    pred_boundaries_y = tf.nn.conv2d(pred_input, sobel_y, strides=[1,1,1,1], padding='VALID')
    true_boundaries = tf.sqrt(tf.square(true_boundaries_x) + tf.square(true_boundaries_y) + 1e-8)
    pred_boundaries = tf.sqrt(tf.square(pred_boundaries_x) + tf.square(pred_boundaries_y) + 1e-8)
    true_boundaries = tf.clip_by_value(true_boundaries, 0.0, 1.0)
    pred_boundaries = tf.clip_by_value(pred_boundaries, 0.0, 1.0)
    boundary_diff = tf.abs(true_boundaries - pred_boundaries)
    boundary_weight = tf.where(true_boundaries > 0.1, 2.0, 1.0)
    weighted_diff = boundary_diff * boundary_weight
    return tf.reduce_mean(weighted_diff)

def uncertainty_loss(model, x, num_samples=3):
    predictions = []
    try:
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        for _ in range(num_samples):
            pred = model(x, training=True)
            predictions.append(pred)
        predictions = tf.stack(predictions)
        epistemic_uncertainty = tf.math.reduce_variance(predictions, axis=0)
        unc_loss = tf.reduce_mean(epistemic_uncertainty)
        return unc_loss
    except Exception as e:
        logging.error(f"Error in uncertainty_loss: {str(e)}")
        return tf.constant(0.0, dtype=tf.float32)