import os
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def custom_tf_function(func):
    return tf.function(func, jit_compile=False, reduce_retracing=True)

def estimate_model_flops(model, input_shape):
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
    logging.debug(f"Validating tensor: {name}, expected shape: {expected_shape_prefix}")
    if not isinstance(tensor, (tf.Tensor, np.ndarray)):
        raise ValueError(f"{name} must be a TensorFlow tensor or NumPy array, got {type(tensor)}")
    if isinstance(tensor, tf.Tensor) and tf.reduce_any(tf.math.is_nan(tensor)):
        logging.warning(f"NaN detected in {name}")
    if expected_shape_prefix:
        actual_shape = tensor.shape
        expected_non_batch = [dim for dim in expected_shape_prefix if dim is not None]
        actual_non_batch = actual_shape[len(actual_shape) - len(expected_non_batch):]
        if actual_non_batch != tuple(expected_non_batch):
            raise ValueError(
                f"{name} shape {actual_shape} does not match expected non-batch shape {expected_non_batch}"
            )

def visualize_uncertainty(uncertainty, output_dir, filename):
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
            plt.title(f"Sample {i + 1}")
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
    except Exception as e:
        logging.error(f"Error in visualize_uncertainty: {str(e)}")

def visualize_comprehensive_uncertainties(images, masks, predictions, uncertainties, output_dir, epoch, fold):
    try:
        os.makedirs(output_dir, exist_ok=True)
        mean_pred, total_unc, epistemic_unc, aleatoric_unc = uncertainties
        images = images.numpy() if hasattr(images, 'numpy') else images
        masks = masks.numpy() if hasattr(masks, 'numpy') else masks
        mean_pred = mean_pred.numpy() if hasattr(mean_pred, 'numpy') else mean_pred
        total_unc = total_unc.numpy() if hasattr(total_unc, 'numpy') else total_unc
        epistemic_unc = epistemic_unc.numpy() if hasattr(epistemic_unc, 'numpy') else epistemic_unc
        aleatoric_unc = aleatoric_unc.numpy() if hasattr(aleatoric_unc, 'numpy') else aleatoric_unc
        num_samples = min(4, images.shape[0])
        fig, axes = plt.subplots(6, num_samples, figsize=(20, 24))
        for i in range(num_samples):
            axes[0, i].imshow(images[i, :, :, 0], cmap='gray')
            axes[0, i].set_title(f'Input {i+1}')
            axes[0, i].axis('off')
            axes[1, i].imshow(masks[i, :, :, 1], cmap='gray')
            axes[1, i].set_title('Ground Truth')
            axes[1, i].axis('off')
            pred_binary = (mean_pred[i, :, :, 1] > 0.5).astype(np.float32)
            axes[2, i].imshow(pred_binary, cmap='gray')
            iou = calculate_iou_single(masks[i, :, :, 1], pred_binary)
            axes[2, i].set_title(f'Prediction (IoU: {iou:.3f})')
            axes[2, i].axis('off')
            im1 = axes[3, i].imshow(total_unc[i, :, :, 1], cmap='hot')
            axes[3, i].set_title('Total Uncertainty')
            axes[3, i].axis('off')
            plt.colorbar(im1, ax=axes[3, i], fraction=0.046)
            im2 = axes[4, i].imshow(epistemic_unc[i, :, :, 1], cmap='viridis')
            axes[4, i].set_title('Epistemic (Model)')
            axes[4, i].axis('off')
            plt.colorbar(im2, ax=axes[4, i], fraction=0.046)
            im3 = axes[5, i].imshow(aleatoric_unc[i, :, :, 1], cmap='plasma')
            axes[5, i].set_title('Aleatoric (Data)')
            axes[5, i].axis('off')
            plt.colorbar(im3, ax=axes[5, i], fraction=0.046)
        plt.suptitle(f'Fold {fold+1}, Epoch {epoch}: Predictions and Uncertainty Analysis', fontsize=16)
        plt.tight_layout()
        filename = f'comprehensive_uncertainty_fold{fold+1}_epoch{epoch}.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved comprehensive uncertainty visualization to {filename}")
    except Exception as e:
        logging.error(f"Error in visualize_comprehensive_uncertainties: {str(e)}")

def calculate_iou_single(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / (union + 1e-8)