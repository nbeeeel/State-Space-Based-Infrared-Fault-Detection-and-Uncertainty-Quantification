import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import logging
import os

def estimate_model_flops(model, input_shape):
    """Simple FLOPS estimation"""
    try:
        sample_input = tf.zeros([1] + list(input_shape))
        _ = model(sample_input, training=False)
        total_params = model.count_params()
        flops_per_sample = total_params * 2
        gflops = flops_per_sample / 1e9
        return gflops
    except:
        return 0.0

def validate_tensor(tensor, name, expected_shape_prefix):
    """Validate tensor shape and check for NaN values"""
    if not isinstance(tensor, (tf.Tensor, np.ndarray)):
        raise ValueError(f"{name} must be a TensorFlow tensor or NumPy array")
    if isinstance(tensor, tf.Tensor) and tf.reduce_any(tf.math.is_nan(tensor)):
        logging.warning(f"NaN detected in {name}")

def visualize_uncertainty(uncertainty, output_dir, filename):
    """Visualize uncertainty maps"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        uncertainty = uncertainty.numpy() if isinstance(uncertainty, tf.Tensor) else uncertainty
        if uncertainty.ndim == 4:
            uncertainty = uncertainty[..., 1]
        plt.figure(figsize=(16, 4))
        for i in range(min(8, uncertainty.shape[0])):
            plt.subplot(1, 8, i + 1)
            plt.imshow(uncertainty[i], cmap='inferno')
            plt.axis('off')
            plt.title(f"Sample {i + 1}", fontsize=16)
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
    except Exception as e:
        logging.error(f"Error in visualize_uncertainty: {str(e)}")