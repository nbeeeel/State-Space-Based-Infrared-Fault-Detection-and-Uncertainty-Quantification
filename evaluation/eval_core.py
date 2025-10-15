import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import logging
import json
import time
from datetime import datetime
from eval_model import build_unet, PhysicsAwareModel, EnhancedCustomModel
from eval_metrics import calculate_comprehensive_metrics, find_optimal_threshold
from eval_utils import validate_tensor, visualize_uncertainty
from eval_data_processing import load_and_split_data

def load_test_data(image_directory, mask_directory, num_images, output_dir):
    """Load test data"""
    logging.info("Loading test data...")
    X_train, X_test, y_train, y_test = load_and_split_data(image_directory, mask_directory, num_images, output_dir)
    validate_tensor(X_test, 'X_test', [None, 224, 224, 1])
    validate_tensor(y_test, 'y_test', [None, 224, 224, 2])
    logging.info(f"Test data loaded: {X_test.shape[0]} samples")
    return X_test, y_test, X_train, y_train

def find_model_weights(weights_dir):
    """Find best available model weights"""
    weights_patterns = [
        'final_enhanced_model.weights.h5',
        'best_model.weights.h5',
        'latest_model.weights.h5',
        'fold_*/best_model.weights.h5'
    ]
    all_weights = []
    for pattern in weights_patterns:
        found = glob.glob(os.path.join(weights_dir, pattern))
        all_weights.extend(found)
    all_weights = sorted(list(set(all_weights)))
    if not all_weights:
        raise FileNotFoundError(f"No model weights found in {weights_dir}")
    logging.info(f"Found {len(all_weights)} weight files, using: {all_weights[0]}")
    return all_weights[0]

def load_model_with_weights(weights_path, config):
    """Load model with weights"""
    logging.info(f"Loading model with weights: {weights_path}")
    input_shape = (config['img_height'], config['img_width'], config['img_channels'])
    unet_model = build_unet(
        input_shape,
        d_model=config['d_model'],
        d_model1=config['d_model1'],
        num_classes=config['num_classes'],
        use_physics_loss=True
    )
    physics_model = PhysicsAwareModel(unet_model)
    model = EnhancedCustomModel(
        physics_model,
        config['lambda_uncertainty'],
        config['num_classes'],
        config['physics_weight'],
        config['focal_weight'],
        config['boundary_weight']
    )
    dummy_input = tf.zeros((1,) + input_shape, dtype=tf.float32)
    _ = model(dummy_input, training=False)
    model.load_weights(weights_path)
    logging.info("Model loaded successfully")
    return model

def predict_with_uncertainty(model, x, num_samples=10, mc_dropout=True):
    """Prediction with uncertainty estimation"""
    predictions = []
    @tf.function(reduce_retracing=True)
    def single_prediction(x_input, training):
        return model(x_input, training=training)
    try:
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        mc_predictions = []
        for _ in range(num_samples):
            pred = single_prediction(x, training=mc_dropout)
            mc_predictions.append(pred)
        aug_predictions = []
        augmentations = [
            lambda img: img,
            lambda img: tf.image.flip_left_right(img),
            lambda img: tf.image.flip_up_down(img),
            lambda img: tf.image.rot90(img, k=1),
            lambda img: tf.image.rot90(img, k=2),
            lambda img: tf.image.rot90(img, k=3),
        ]
        for aug_fn in augmentations:
            x_aug = aug_fn(x)
            pred_aug = single_prediction(x_aug, training=False)
            if aug_fn == augmentations[1]:
                pred_aug = tf.image.flip_left_right(pred_aug)
            elif aug_fn == augmentations[2]:
                pred_aug = tf.image.flip_up_down(pred_aug)
            elif aug_fn == augmentations[3]:
                pred_aug = tf.image.rot90(pred_aug, k=3)
            elif aug_fn == augmentations[4]:
                pred_aug = tf.image.rot90(pred_aug, k=2)
            elif aug_fn == augmentations[5]:
                pred_aug = tf.image.rot90(pred_aug, k=1)
            aug_predictions.append(pred_aug)
        mc_predictions = tf.stack(mc_predictions)
        aug_predictions = tf.stack(aug_predictions)
        all_predictions = tf.concat([mc_predictions, aug_predictions], axis=0)
        mean_pred = tf.reduce_mean(all_predictions, axis=0)
        epistemic_uncertainty = tf.math.reduce_variance(mc_predictions, axis=0)
        aleatoric_uncertainty = tf.math.reduce_variance(aug_predictions, axis=0)
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        return mean_pred, total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty
    except Exception as e:
        logging.error(f"Error in predict_with_uncertainty: {str(e)}")
        safe_pred = model(x, training=False)
        zeros = tf.zeros_like(safe_pred)
        return safe_pred, zeros, zeros, zeros

def create_visualizations(X_test, y_test, predictions, uncertainties, optimal_thresholds, metrics, output_dir):
    """Create comprehensive visualizations with larger fonts and 5 rows"""
    viz_dir = os.path.join(output_dir, 'visualizations')
    if hasattr(X_test, 'numpy'):
        X_test = X_test.numpy()
    if hasattr(y_test, 'numpy'):
        y_test = y_test.numpy()
    if hasattr(predictions, 'numpy'):
        predictions = predictions.numpy()
    if hasattr(uncertainties, 'numpy'):
        uncertainties = uncertainties.numpy()
    num_samples = min(5, len(X_test))
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    plt.rcParams.update({'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 16})
    fig, axes = plt.subplots(num_samples, 5, figsize=(30, 6*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    threshold = optimal_thresholds['iou']
    for i, idx in enumerate(indices):
        axes[i, 0].imshow(X_test[idx, ..., 0], cmap='gray')
        axes[i, 0].set_title(f'Input {idx}')
        axes[i, 0].axis('off')
        axes[i, 1].imshow(y_test[idx, ..., 1], cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        axes[i, 2].imshow(predictions[idx, ..., 1], cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title('Prediction (Raw)')
        axes[i, 2].axis('off')
        pred_binary = (predictions[idx, ..., 1] > threshold).astype(float)
        axes[i, 3].imshow(pred_binary, cmap='gray')
        iou = np.sum(y_test[idx, ..., 1] * pred_binary) / (np.sum(y_test[idx, ..., 1]) + np.sum(pred_binary) - np.sum(y_test[idx, ..., 1] * pred_binary) + 1e-8)
        axes[i, 3].set_title(f'Thresholded (IoU: {iou:.3f})')
        axes[i, 3].axis('off')
        im = axes[i, 4].imshow(uncertainties[idx, ..., 1], cmap='hot')
        axes[i, 4].set_title('Uncertainty')
        axes[i, 4].axis('off')
        plt.colorbar(im, ax=axes[i, 4], fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'sample_predictions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    plt.rcParams.update({'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 16})
    plt.figure(figsize=(18, 12))
    plt.subplot(2, 3, 1)
    metric_names = ['IoU', 'F1', 'Precision', 'Recall', 'MCC']
    metric_values = [
        metrics['mean_iou'], metrics['f1_score'],
        metrics['precision'], metrics['recall'], metrics['mcc']
    ]
    bars = plt.bar(metric_names, metric_values, alpha=0.7, color='skyblue')
    plt.ylim(0, 1)
    plt.title('Performance Metrics')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=14)
    plt.subplot(2, 3, 2)
    thresh_results = find_optimal_threshold(y_test, predictions)[1]
    plt.plot(thresh_results['threshold'], thresh_results['f1'], label='F1', linewidth=2)
    plt.plot(thresh_results['threshold'], thresh_results['iou'], label='IoU', linewidth=2)
    plt.axvline(x=optimal_thresholds['iou'], color='red', linestyle='--', alpha=0.7, label=f'Optimal: {optimal_thresholds["iou"]:.3f}')
    plt.xlabel('Threshold', fontsize=16)
    plt.ylabel('Score', fontsize=16)
    plt.title('Threshold Analysis', fontsize=18)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(2, 3, 3)
    plt.hist(metrics['per_image_ious'], bins=20, alpha=0.7, label='IoU', color='orange')
    plt.axvline(x=metrics['mean_iou'], color='orange', linestyle='--', label=f'Mean IoU: {metrics["mean_iou"]:.3f}')
    plt.xlabel('Score', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.title('Per-Image Score Distribution', fontsize=18)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(2, 3, 4)
    cm = np.array(metrics['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                annot_kws={"size": 16})
    plt.title('Confusion Matrix', fontsize=18)
    plt.ylabel('True Label', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=16)
    plt.subplot(2, 3, 5)
    uncertainty_flat = uncertainties[..., 1].flatten()
    plt.hist(uncertainty_flat, bins=50, alpha=0.7, color='red')
    plt.axvline(x=np.mean(uncertainty_flat), color='darkred', linestyle='--',
                label=f'Mean: {np.mean(uncertainty_flat):.4f}')
    plt.xlabel('Uncertainty', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.title('Uncertainty Distribution', fontsize=18)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(2, 3, 6)
    stats_text = f"""Model Performance Summary

Best Metrics:
• IoU: {metrics['mean_iou']:.4f} ± {metrics['std_iou']:.4f}
• F1: {metrics['f1_score']:.4f}
• Precision: {metrics['precision']:.4f}
• Recall: {metrics['recall']:.4f}

Quality Metrics:
• MCC: {metrics['mcc']:.4f}
• Kappa: {metrics['kappa']:.4f}
• ROC-AUC: {metrics['roc_auc']:.4f}
• PR-AUC: {metrics['pr_auc']:.4f}

Optimal Threshold: {optimal_thresholds['iou']:.3f}"""
    plt.text(0.1, 0.5, stats_text, fontsize=16, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'comprehensive_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)
    logging.info("Visualizations created successfully")

def generate_report(results, output_dir):
    """Generate evaluation report"""
    report_path = os.path.join(output_dir, 'reports', 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODEL EVALUATION REPORT\n")
        f.write("Generated: " + str(results['timestamp']) + "\n")
        f.write("="*80 + "\n\n")
        metrics = results['metrics']
        f.write("MODEL PERFORMANCE\n")
        f.write("-"*40 + "\n")
        f.write("IoU Score: {:.4f} ± {:.4f}\n".format(metrics['mean_iou'], metrics['std_iou']))
        f.write("F1 Score: {:.4f}\n".format(metrics['f1_score']))
        f.write("Precision: {:.4f}\n".format(metrics['precision']))
        f.write("Recall: {:.4f}\n".format(metrics['recall']))
        f.write("MCC: {:.4f}\n".format(metrics['mcc']))
        f.write("Cohen's Kappa: {:.4f}\n".format(metrics['kappa']))
        f.write("ROC-AUC: {:.4f}\n".format(metrics['roc_auc']))
        f.write("PR-AUC: {:.4f}\n".format(metrics['pr_auc']))
        f.write("\n")
        f.write("OPTIMAL THRESHOLDS\n")
        f.write("-"*40 + "\n")
        for metric, threshold in results['optimal_thresholds'].items():
            f.write("{}: {:.3f}\n".format(metric, threshold))
        f.write("\n")
        f.write("CONFIGURATION\n")
        f.write("-"*40 + "\n")
        f.write("MC Dropout samples: {}\n".format(results['config']['mc_samples']))
        f.write("TTA augmentations: {}\n".format(', '.join(results['config']['tta_augmentations'])))
        f.write("Model weights: {}\n".format(results['model_weights']))
        f.write("\n")
        f.write("="*80 + "\n")
    logging.info(f"Evaluation report saved to {report_path}")

def run_evaluation(config):
    """Run the complete evaluation"""
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config['base_output_dir'], f'eval_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'reports'), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'evaluation.log')),
            logging.StreamHandler()
        ]
    )
    json.dump(config, open(os.path.join(output_dir, 'config.json'), 'w'), indent=4)
    logging.info(f"Starting evaluation at {timestamp}")
    logging.info(f"Output directory: {output_dir}")
    logging.info("Loading test data...")
    X_test, y_test, X_train, y_train = load_test_data(
        config['image_directory'], config['mask_directory'], config['num_images'], output_dir
    )
    logging.info("Loading trained model...")
    weights_path = find_model_weights(config['weights_dir'])
    model = load_model_with_weights(weights_path, config)
    results = {
        'timestamp': timestamp,
        'config': config,
        'model_weights': weights_path,
        'test_samples': len(X_test)
    }
    logging.info("Generating predictions with uncertainty estimation...")
    predictions = []
    uncertainties = []
    dataset = tf.data.Dataset.from_tensor_slices(X_test).batch(config['batch_size'])
    for i, batch in enumerate(dataset):
        if i % 5 == 0:
            logging.info(f"Processing batch {i+1}/{len(X_test)//config['batch_size']+1}")
        mean_pred, total_unc, epistemic_unc, aleatoric_unc = predict_with_uncertainty(
            model, batch, config['mc_samples'], mc_dropout=True
        )
        predictions.append(mean_pred)
        uncertainties.append(total_unc)
    y_pred = tf.concat(predictions, axis=0)
    uncertainty = tf.concat(uncertainties, axis=0)
    logging.info("Finding optimal prediction threshold...")
    optimal_thresholds, threshold_analysis = find_optimal_threshold(y_test, y_pred)
    results['optimal_thresholds'] = optimal_thresholds
    results['threshold_analysis'] = threshold_analysis
    logging.info("Calculating comprehensive metrics...")
    threshold = optimal_thresholds['iou']
    metrics = calculate_comprehensive_metrics(y_test, y_pred, threshold)
    results['metrics'] = metrics
    logging.info("Creating visualizations...")
    create_visualizations(X_test, y_test, y_pred, uncertainty, optimal_thresholds, metrics, output_dir)
    if config['save_predictions']:
        logging.info("Saving predictions...")
        np.save(os.path.join(output_dir, 'predictions', 'predictions.npy'), y_pred.numpy())
        np.save(os.path.join(output_dir, 'predictions', 'uncertainties.npy'), uncertainty.numpy())
        np.save(os.path.join(output_dir, 'predictions', 'ground_truth.npy'), y_test)
    logging.info("Generating evaluation report...")
    generate_report(results, output_dir)
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        json.dump(convert_numpy(results), f, indent=4)
    elapsed_time = time.time() - start_time
    logging.info("="*80)
    logging.info("EVALUATION COMPLETED SUCCESSFULLY!")
    logging.info("="*80)
    logging.info(f"Total time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    logging.info(f"Results saved to: {output_dir}")
    logging.info("")
    logging.info("FINAL RESULTS:")
    logging.info(f"  IoU Score: {metrics['mean_iou']:.4f} ± {metrics['std_iou']:.4f}")
    logging.info(f"  F1 Score: {metrics['f1_score']:.4f}")
    logging.info(f"  Precision: {metrics['precision']:.4f}")
    logging.info(f"  Recall: {metrics['recall']:.4f}")
    logging.info(f"  Optimal Threshold: {optimal_thresholds['iou']:.3f}")
    logging.info("="*80)
    return results