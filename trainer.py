import os
import json
import time
import shutil
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from utils import estimate_model_flops, validate_tensor, visualize_comprehensive_uncertainties, visualize_uncertainty
from model import build_unet, PhysicsAwareModel
from losses import weighted_binary_crossentropy, dice_loss, focal_loss, boundary_loss, uncertainty_loss
from metrics import BinaryIoU, DiceCoefficient
from data_processing import predict_with_uncertainty

class DifficultyWeightedSampler:
    def __init__(self, X, y, model):
        predictions = model.predict(X, verbose=0)
        difficulties = self.calculate_difficulty(y, predictions)
        self.weights = 1.0 + difficulties

    def calculate_difficulty(self, y_true, y_pred):
        ious = []
        for i in range(len(y_true)):
            iou = self.calculate_iou(y_true[i], y_pred[i])
            ious.append(1.0 - iou)
        return np.array(ious)

    def calculate_iou(self, y_true, y_pred):
        y_true_flat = y_true[..., 1].flatten()
        y_pred_flat = (y_pred[..., 1] > 0.5).flatten()
        intersection = np.sum(y_true_flat * y_pred_flat)
        union = np.sum(y_true_flat) + np.sum(y_pred_flat) - intersection
        return intersection / (union + 1e-8)

class EnhancedCustomModel(tf.keras.Model):
    def __init__(self, base_model, lambda_uncertainty, num_classes, physics_weight=0.5,
                 focal_weight=0.3, boundary_weight=0.2, mixup_alpha=0.2):
        super().__init__()
        self.base_model = base_model
        self.lambda_uncertainty = lambda_uncertainty
        self.num_classes = num_classes
        self.physics_weight = physics_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        self.mixup_alpha = mixup_alpha
        self.difficulty_sampler = None

    def build(self, input_shape):
        super().build(input_shape)
        self.base_model.build(input_shape)

    def call(self, inputs, training=None):
        return self.base_model(inputs, training=training)

    def train_step(self, data):
        x, y = data
        def apply_mixup():
            batch_size = tf.shape(x)[0]
            indices = tf.random.shuffle(tf.range(batch_size))
            x2 = tf.gather(x, indices)
            y2 = tf.gather(y, indices)
            lam = tf.random.uniform([], minval=0.1, maxval=0.3)
            x_mixed = lam * x + (1 - lam) * x2
            y_mixed = lam * y + (1 - lam) * y2
            return x_mixed, y_mixed
        def no_mixup():
            return x, y
        x, y = tf.cond(tf.random.uniform([]) < 0.3, apply_mixup, no_mixup)
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            bce_loss = weighted_binary_crossentropy(y, y_pred)
            dice_loss_val = dice_loss(y, y_pred)
            focal_loss_val = focal_loss(y, y_pred)
            boundary_loss_val = boundary_loss(y, y_pred)
            bce_loss = tf.where(tf.math.is_finite(bce_loss), bce_loss, tf.constant(0.1, dtype=tf.float32))
            dice_loss_val = tf.where(tf.math.is_finite(dice_loss_val), dice_loss_val, tf.constant(0.1, dtype=tf.float32))
            focal_loss_val = tf.where(tf.math.is_finite(focal_loss_val), focal_loss_val, tf.constant(0.0, dtype=tf.float32))
            boundary_loss_val = tf.where(tf.math.is_finite(boundary_loss_val), boundary_loss_val, tf.constant(0.0, dtype=tf.float32))
            segmentation_loss = (
                0.4 * bce_loss +
                0.5 * dice_loss_val +
                self.focal_weight * focal_loss_val +
                self.boundary_weight * boundary_loss_val
            )
            unc_loss = tf.constant(0.0, dtype=tf.float32)
            if self.lambda_uncertainty > 0:
                try:
                    unc_loss = uncertainty_loss(self, x, num_samples=3)
                    unc_loss = tf.where(tf.math.is_finite(unc_loss), unc_loss, tf.constant(0.0, dtype=tf.float32))
                except:
                    unc_loss = tf.constant(0.0, dtype=tf.float32)
            physics_loss = tf.constant(0.0, dtype=tf.float32)
            if hasattr(self.base_model, 'physics_layers') and self.base_model.physics_layers:
                try:
                    for layer in self.base_model.physics_layers:
                        layer_physics_loss = layer.get_physics_loss()
                        if layer_physics_loss is not None:
                            layer_physics_loss = tf.where(
                                tf.math.is_finite(layer_physics_loss),
                                layer_physics_loss,
                                tf.constant(0.0, dtype=tf.float32)
                            )
                            physics_loss += tf.reduce_mean(layer_physics_loss)
                except:
                    physics_loss = tf.constant(0.0, dtype=tf.float32)
            loss = segmentation_loss + self.lambda_uncertainty * unc_loss + self.physics_weight * physics_loss
            loss = tf.where(tf.math.is_finite(loss), loss, tf.constant(1.0, dtype=tf.float32))
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.trainable_variables if 'bias' not in v.name])
            loss += 1e-5 * l2_loss
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        gradients = [
            tf.clip_by_norm(g, 1.0) if g is not None else tf.zeros_like(v)
            for g, v in zip(gradients, trainable_vars)
        ]
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        try:
            self.compiled_metrics.update_state(y, y_pred)
            metrics = {m.name: m.result() for m in self.metrics}
        except:
            metrics = {}
        metrics['loss'] = loss
        metrics['physics_loss'] = physics_loss
        metrics['segmentation_loss'] = segmentation_loss
        metrics['uncertainty_loss'] = unc_loss
        metrics['bce_loss'] = bce_loss
        metrics['dice_loss_component'] = dice_loss_val
        metrics['focal_loss'] = focal_loss_val
        metrics['boundary_loss'] = boundary_loss_val
        return metrics

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        bce_loss = weighted_binary_crossentropy(y, y_pred)
        dice_loss_val = dice_loss(y, y_pred)
        focal_loss_val = focal_loss(y, y_pred)
        boundary_loss_val = boundary_loss(y, y_pred)
        segmentation_loss = (
            0.4 * bce_loss +
            0.5 * dice_loss_val +
            self.focal_weight * focal_loss_val +
            self.boundary_weight * boundary_loss_val
        )
        unc_loss = 0.0
        if self.lambda_uncertainty > 0:
            try:
                unc_loss = uncertainty_loss(self, x, num_samples=3)
            except:
                unc_loss = 0.0
        physics_loss = 0.0
        if hasattr(self.base_model, 'physics_layers') and self.base_model.physics_layers:
            for layer in self.base_model.physics_layers:
                layer_physics_loss = layer.get_physics_loss()
                if layer_physics_loss is not None:
                    physics_loss += tf.reduce_mean(layer_physics_loss)
        loss = segmentation_loss + self.lambda_uncertainty * unc_loss
        self.compiled_metrics.update_state(y, y_pred)
        metrics = {m.name: m.result() for m in self.metrics}
        metrics['loss'] = loss
        metrics['physics_loss'] = physics_loss
        metrics['segmentation_loss'] = segmentation_loss
        metrics['uncertainty_loss'] = unc_loss
        metrics['focal_loss'] = focal_loss_val
        metrics['boundary_loss'] = boundary_loss_val
        return metrics

    def initialize_difficulty_sampler(self, X, y):
        try:
            if self.difficulty_sampler is None:
                self.difficulty_sampler = DifficultyWeightedSampler(X, y, self)
                logging.info("Initialized difficulty-weighted sampler")
        except Exception as e:
            logging.warning(f"Could not initialize difficulty sampler: {str(e)}")

class NaNDetectionCallback(tf.keras.callbacks.Callback):
    def __init__(self, patience=3):
        super().__init__()
        self.patience = patience
        self.nan_count = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs and 'loss' in logs:
            if np.isnan(logs['loss']) or np.isinf(logs['loss']):
                self.nan_count += 1
                logging.warning(f"NaN/Inf detected in loss! Count: {self.nan_count}/{self.patience}")
                if self.nan_count >= self.patience:
                    logging.error(f"Too many NaN/Inf values. Stopping training.")
                    self.model.stop_training = True
            else:
                self.nan_count = 0

class ImprovedContinuousModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, trainer, fold_idx):
        super().__init__()
        self.trainer = trainer
        self.fold_idx = fold_idx
        self.best_val_loss = float('inf')
        self.best_val_dice = 0.0
        self.fold_dir = os.path.join(trainer.output_dir, f'fold_{fold_idx+1}')
        os.makedirs(self.fold_dir, exist_ok=True)
        self.fold_best_weights_path = os.path.join(self.fold_dir, 'best_model.weights.h5')
        self.fold_latest_weights_path = os.path.join(self.fold_dir, 'latest_model.weights.h5')

    def on_epoch_end(self, epoch, logs=None):
        try:
            self.model.save_weights(self.fold_latest_weights_path)
            self.model.save_weights(self.trainer.latest_weights_path)
            val_dice = logs.get('val_dice_manual', logs.get('val_dice_coefficient', 0))
            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                self.model.save_weights(self.fold_best_weights_path)
                logging.info(f"Fold {self.fold_idx+1}: New best dice {val_dice:.4f} saved")
                if val_dice > self.trainer.global_best_dice:
                    self.trainer.global_best_dice = val_dice
                    self.model.save_weights(self.trainer.best_weights_path)
                    logging.info(f"New global best dice: {val_dice:.4f} (Fold {self.fold_idx+1})")
        except Exception as e:
            logging.error(f"Error saving weights: {str(e)}")

class EnhancedVisualizationCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val, output_dir, fold, num_samples=8):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.output_dir = output_dir
        self.fold = fold
        self.num_samples = num_samples

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            try:
                x_samples = self.X_val[:4]
                y_samples = self.y_val[:4]
                mean_pred, total_unc, epistemic_unc, aleatoric_unc = predict_with_uncertainty(
                    self.model, x_samples, num_samples=self.num_samples
                )
                visualize_comprehensive_uncertainties(
                    x_samples, y_samples, mean_pred,
                    (mean_pred, total_unc, epistemic_unc, aleatoric_unc),
                    self.output_dir, epoch + 1, self.fold
                )
                visualize_uncertainty(
                    total_unc, self.output_dir,
                    f'total_uncertainty_fold{self.fold+1}_epoch{epoch+1}.png'
                )
                total_unc_mean = tf.reduce_mean(total_unc).numpy()
                epistemic_unc_mean = tf.reduce_mean(epistemic_unc).numpy()
                aleatoric_unc_mean = tf.reduce_mean(aleatoric_unc).numpy()
                logging.info(f"Uncertainty Statistics - Total: {total_unc_mean:.4f}, "
                           f"Epistemic: {epistemic_unc_mean:.4f}, "
                           f"Aleatoric: {aleatoric_unc_mean:.4f}")
            except Exception as e:
                logging.error(f"Error in visualization callback: {str(e)}")

class EnhancedTrainingMonitor(tf.keras.callbacks.Callback):
    def __init__(self, fold_idx):
        super().__init__()
        self.fold_idx = fold_idx
        self.best_dice = 0.0
        self.loss_history = []

    def on_epoch_end(self, epoch, logs=None):
        dice = logs.get('dice_coefficient', 0)
        val_dice = logs.get('val_dice_manual', logs.get('val_dice_coefficient', 0))
        if dice > self.best_dice:
            self.best_dice = dice
        self.loss_history.append({
            'epoch': epoch + 1,
            'dice': float(dice),
            'val_dice': float(val_dice),
            'loss': float(logs.get('loss', 0)),
            'val_loss': float(logs.get('val_loss', 0))
        })
        if epoch % 5 == 0:
            logging.info(
                f"Fold {self.fold_idx+1}, Epoch {epoch+1}: "
                f"Loss: {logs.get('loss', 0):.4f}, Val Loss: {logs.get('val_loss', 0):.4f}, "
                f"Dice: {dice:.4f}, Val Dice: {val_dice:.4f}"
            )

class EnhancedValidationMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_dataset, output_dir, fold_idx, use_tta=True):
        super().__init__()
        self.val_dataset = val_dataset
        self.output_dir = output_dir
        self.fold_idx = fold_idx
        self.use_tta = use_tta
        self.val_metrics_history = []

    def on_epoch_end(self, epoch, logs=None):
        try:
            all_preds = []
            all_true = []
            for x_batch, y_batch in self.val_dataset:
                preds = self.model.predict(x_batch, verbose=0)
                all_preds.append(preds)
                all_true.append(y_batch)
            all_preds = np.concatenate(all_preds, axis=0)
            all_true = np.concatenate(all_true, axis=0)
            pred_binary = (all_preds[..., 1] > 0.5).astype(np.float32)
            true_binary = all_true[..., 1]
            intersection = np.sum(pred_binary * true_binary)
            union = np.sum(pred_binary) + np.sum(true_binary) - intersection
            val_iou = float(intersection / (union + 1e-8))
            val_dice = float(2 * intersection / (np.sum(pred_binary) + np.sum(true_binary) + 1e-8))
            logs['val_iou_manual'] = val_iou
            logs['val_dice_manual'] = val_dice
            self.val_metrics_history.append({
                'epoch': epoch + 1,
                'val_iou': val_iou,
                'val_dice': val_dice
            })
        except Exception as e:
            logging.error(f"Error in validation metrics: {str(e)}")
            logs['val_iou_manual'] = 0.0
            logs['val_dice_manual'] = 0.0

class EnhancedSegmentationTrainer:
    def __init__(self, config):
        self.input_shape = config['input_shape']
        self.num_classes = config['num_classes']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.output_dir = config['output_dir']
        self.final_dir = os.path.join(self.output_dir, 'final')
        self.val_folds = config['val_folds']
        self.initial_learning_rate = config['initial_learning_rate']
        self.lambda_uncertainty = config['lambda_uncertainty']
        self.physics_weight = config.get('physics_weight', 0.1)
        self.focal_weight = config.get('focal_weight', 0.2)
        self.boundary_weight = config.get('boundary_weight', 0.3)
        self.mixup_alpha = config.get('mixup_alpha', 0.2)
        self.early_stopping_patience = config.get('early_stopping_patience', 20)
        self.nan_detection_patience = config.get('nan_detection_patience', 3)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.final_dir, exist_ok=True)
        self.model_architecture_path = os.path.join(self.output_dir, 'model_architecture.json')
        self.best_weights_path = os.path.join(self.output_dir, 'best_model.weights.h5')
        self.latest_weights_path = os.path.join(self.output_dir, 'latest_model.weights.h5')
        self.global_best_dice = 0.0
        self.model = None
        self.build_or_load_model()

    def build_or_load_model(self):
        if os.path.exists(self.model_architecture_path) and os.path.exists(self.latest_weights_path):
            logging.info("Loading existing model architecture and weights...")
            try:
                unet_model = build_unet(self.input_shape, num_classes=self.num_classes, use_physics_loss=True)
                physics_model = PhysicsAwareModel(unet_model)
                self.model = EnhancedCustomModel(
                    physics_model,
                    self.lambda_uncertainty,
                    self.num_classes,
                    self.physics_weight,
                    self.focal_weight,
                    self.boundary_weight,
                    self.mixup_alpha
                )
                dummy_input = tf.zeros((1,) + self.input_shape, dtype=tf.float32)
                _ = self.model(dummy_input, training=False)
                self.model.load_weights(self.latest_weights_path)
                logging.info("Successfully loaded existing model and weights")
                physics_model.initialize_physics_layers()
                self.print_model_summary()
            except Exception as e:
                logging.warning(f"Failed to load existing model: {str(e)}. Building new model...")
                self.build_new_model()
        else:
            logging.info("No existing model found. Building new model...")
            self.build_new_model()

    def build_new_model(self):
        logging.info("Building Enhanced U-Net model...")
        unet_model = build_unet(self.input_shape, num_classes=self.num_classes, use_physics_loss=True)
        logging.info(f"U-Net built: {len(unet_model.layers)} layers")
        logging.info("Building PhysicsAwareModel...")
        physics_model = PhysicsAwareModel(unet_model)
        logging.info("Creating Enhanced CustomModel...")
        self.model = EnhancedCustomModel(
            physics_model,
            self.lambda_uncertainty,
            self.num_classes,
            self.physics_weight,
            self.focal_weight,
            self.boundary_weight,
            self.mixup_alpha
        )
        logging.info("Initializing physics layers...")
        physics_model.initialize_physics_layers()
        dummy_input = tf.zeros((1,) + self.input_shape, dtype=tf.float32)
        _ = self.model(dummy_input, training=False)
        self.save_model_architecture()
        self.print_model_summary()
        logging.info("Enhanced model initialization complete")

    def print_model_summary(self):
        import psutil
        process = psutil.Process()
        current_memory = process.memory_info().rss / 1024 / 1024
        logging.info("\n" + "="*50)
        logging.info("ENHANCED MODEL ARCHITECTURE SUMMARY")
        logging.info("="*50)
        logging.info(f"Total parameters: {self.model.count_params():,}")
        logging.info(f"Trainable parameters: {sum([tf.reduce_prod(v.shape).numpy() for v in self.model.trainable_variables]):,}")
        logging.info(f"Non-trainable parameters: {sum([tf.reduce_prod(v.shape).numpy() for v in self.model.non_trainable_variables]):,}")
        logging.info(f"Input shape: {self.input_shape}")
        logging.info(f"Output shape: (None, 224, 224, {self.num_classes})")
        logging.info(f"Number of layers: {len(self.model.base_model.layers)}")
        logging.info(f"Physics layers: {len(self.model.base_model.physics_layers)}")
        logging.info(f"Current memory usage: {current_memory:.2f} MB")
        logging.info(f"Enhanced features: Focal Loss, Boundary Loss, Enhanced Uncertainty, Mixup")
        logging.info(f"Early stopping patience: {self.early_stopping_patience}")
        logging.info(f"NaN detection patience: {self.nan_detection_patience}")
        try:
            flops = estimate_model_flops(self.model, self.input_shape)
            logging.info(f"Estimated FLOPs: {flops:.2f} GFLOPs per sample")
        except:
            pass
        logging.info("="*50 + "\n")

    def save_model_architecture(self):
        model_config = {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'lambda_uncertainty': self.lambda_uncertainty,
            'physics_weight': self.physics_weight,
            'focal_weight': self.focal_weight,
            'boundary_weight': self.boundary_weight,
            'mixup_alpha': self.mixup_alpha,
            'early_stopping_patience': self.early_stopping_patience,
            'nan_detection_patience': self.nan_detection_patience,
            'model_type': 'enhanced_physics_aware_unet'
        }
        with open(self.model_architecture_path, 'w') as json_file:
            json.dump(model_config, json_file)
        logging.info(f"Saved enhanced model architecture to {self.model_architecture_path}")

    def save_enhanced_fold_history(self, history, fold_idx):
        fold_dir = os.path.join(self.output_dir, f'fold_{fold_idx + 1}')
        os.makedirs(fold_dir, exist_ok=True)
        history_dict = {key: [float(v) for v in values] for key, values in history.history.items()}
        history_path = os.path.join(fold_dir, f'history_fold_{fold_idx + 1}.json')
        with open(history_path, 'w') as f:
            json.dump(history_dict, f)
        logging.info(f"Saved fold {fold_idx + 1} history to {history_path}")

    def plot_final_training_summary(self, fold_results):
        successful_folds = [f for f in fold_results if not f.get('catastrophic_failure', False)]
        if not successful_folds:
            logging.warning("No successful folds to plot")
            return
        plt.figure(figsize=(20, 10))
        plt.subplot(2, 3, 1)
        fold_nums = [f['fold'] for f in successful_folds]
        dice_scores = [f['best_val_dice'] for f in successful_folds]
        plt.bar(fold_nums, dice_scores, color='skyblue', edgecolor='navy')
        plt.xlabel('Fold Number')
        plt.ylabel('Best Validation Dice')
        plt.title('Best Dice Score by Fold')
        plt.ylim([0, 1])
        for i, v in enumerate(dice_scores):
            plt.text(fold_nums[i], v + 0.01, f'{v:.3f}', ha='center')
        plt.subplot(2, 3, 2)
        iou_scores = [f['best_val_iou'] for f in successful_folds]
        plt.bar(fold_nums, iou_scores, color='lightgreen', edgecolor='darkgreen')
        plt.xlabel('Fold Number')
        plt.ylabel('Best Validation IoU')
        plt.title('Best IoU Score by Fold')
        plt.ylim([0, 1])
        for i, v in enumerate(iou_scores):
            plt.text(fold_nums[i], v + 0.01, f'{v:.3f}', ha='center')
        plt.subplot(2, 3, 3)
        epochs_trained = [f['epochs_trained'] for f in successful_folds]
        plt.bar(fold_nums, epochs_trained, color='lightcoral', edgecolor='darkred')
        plt.xlabel('Fold Number')
        plt.ylabel('Epochs Trained')
        plt.title('Training Epochs by Fold')
        for i, v in enumerate(epochs_trained):
            plt.text(fold_nums[i], v + 0.5, f'{v}', ha='center')
        plt.subplot(2, 3, 4)
        plt.boxplot(dice_scores, labels=['Dice'])
        plt.ylabel('Score')
        plt.title('Distribution of Best Dice Scores')
        plt.ylim([0, 1])
        plt.subplot(2, 3, 5)
        plt.boxplot(iou_scores, labels=['IoU'])
        plt.ylabel('Score')
        plt.title('Distribution of Best IoU Scores')
        plt.ylim([0, 1])
        plt.subplot(2, 3, 6)
        plt.text(0.1, 0.9, 'Training Summary Statistics', fontsize=16, weight='bold')
        plt.text(0.1, 0.8, f'Total Folds: {len(fold_results)}', fontsize=12)
        plt.text(0.1, 0.7, f'Successful Folds: {len(successful_folds)}', fontsize=12)
        plt.text(0.1, 0.6, f'Failed Folds: {len(fold_results) - len(successful_folds)}', fontsize=12)
        plt.text(0.1, 0.5, f'Mean Dice: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}', fontsize=12)
        plt.text(0.1, 0.4, f'Mean IoU: {np.mean(iou_scores):.4f} ± {np.std(iou_scores):.4f}', fontsize=12)
        plt.text(0.1, 0.3, f'Best Dice: {max(dice_scores):.4f} (Fold {fold_nums[dice_scores.index(max(dice_scores))]})', fontsize=12)
        plt.text(0.1, 0.2, f'Best IoU: {max(iou_scores):.4f} (Fold {fold_nums[iou_scores.index(max(iou_scores))]})', fontsize=12)
        plt.axis('off')
        plt.suptitle('Cross-Validation Training Summary', fontsize=18, weight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_summary_all_folds.png'), dpi=150, bbox_inches='tight')
        plt.close()
        logging.info("Saved final training summary plot")

    def compile_model(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.initial_learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7,
                clipnorm=1.0
            ),
            loss=None,
            metrics=[
                BinaryIoU(name='binary_iou'),
                DiceCoefficient(name='dice_coefficient'),
                tf.keras.metrics.Precision(class_id=1, name='precision_fault'),
                tf.keras.metrics.Recall(class_id=1, name='recall_fault'),
                tf.keras.metrics.BinaryAccuracy(name='accuracy')
            ],
            run_eagerly=False,
            jit_compile=False
        )

    def create_enhanced_dataset(self, X, y, is_training=True):
        validate_tensor(X, 'X', [None, self.input_shape[0], self.input_shape[1], 1])
        validate_tensor(y, 'y', [None, self.input_shape[0], self.input_shape[1], self.num_classes])
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        if is_training:
            dataset = dataset.cache().shuffle(buffer_size=min(1000, len(X)))
        else:
            dataset = dataset.cache()
        dataset = dataset.batch(self.batch_size)
        @tf.function
        def enhanced_augment(x, y):
            try:
                seed = tf.random.uniform([2], maxval=10000, dtype=tf.int32)
                x = tf.image.stateless_random_brightness(x, max_delta=0.15, seed=seed)
                x = tf.image.stateless_random_contrast(x, lower=0.7, upper=1.3, seed=seed)
                if tf.random.uniform([]) < 0.3:
                    noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=0.02)
                    x = x + noise
                    x = tf.clip_by_value(x, 0.0, 1.0)
                if tf.random.uniform([]) < 0.5:
                    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
                    x = tf.image.rot90(x, k)
                    y = tf.image.rot90(y, k)
                return x, y
            except Exception as e:
                logging.error(f"Error in augmentation: {str(e)}")
                return x, y
        if is_training:
            dataset = dataset.map(enhanced_augment, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def evaluate_model(self, X_test, y_test):
        test_dataset = self.create_enhanced_dataset(X_test, y_test, is_training=False)
        results = self.model.evaluate(test_dataset, verbose=1)
        metrics = {
            'test_loss': float(results[0]),
            'test_binary_iou': float(results[1]),
            'test_dice_coefficient': float(results[2]),
            'test_precision_fault': float(results[3]),
            'test_recall_fault': float(results[4]),
            'test_accuracy': float(results[5])
        }
        logging.info(f"Test Evaluation Metrics: {metrics}")
        with open(os.path.join(self.final_dir, 'test_metrics.json'), 'w') as f:
            json.dump(metrics, f)
        return metrics

    def train(self, X_train, y_train, X_test, y_test):
        import psutil
        process = psutil.Process()
        start_time = time.time()
        initial_memory = process.memory_info().rss / 1024 / 1024
        logging.info(f"Initial memory usage: {initial_memory:.2f} MB")
        self.compile_model()
        try:
            flops = estimate_model_flops(self.model, self.input_shape)
            logging.info(f"Estimated model FLOPs: {flops:.2f} GFLOPs per sample")
            total_params = self.model.count_params()
            logging.info(f"Total model parameters: {total_params:,}")
        except Exception as e:
            logging.warning(f"Could not estimate FLOPs: {str(e)}")
            flops = 0
        overall_best_val_loss = float('inf')
        overall_best_dice = 0.0
        overall_best_val_dice = 0.0
        overall_best_val_iou = 0.0
        fold_results = []
        kf = KFold(n_splits=self.val_folds, shuffle=True, random_state=42)
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            try:
                fold_start_memory = process.memory_info().rss / 1024 / 1024
                logging.info(f"\n{'='*60}")
                logging.info(f"STARTING FOLD {fold_idx + 1}/{self.val_folds}")
                logging.info(f"{'='*60}")
                logging.info(f"Current memory usage: {fold_start_memory:.2f} MB")
                if fold_idx > 0 and os.path.exists(self.best_weights_path):
                    logging.info("Loading best weights from previous folds...")
                    try:
                        self.model.load_weights(self.best_weights_path)
                        logging.info("Successfully loaded previous best weights")
                    except Exception as e:
                        logging.warning(f"Could not load previous weights: {str(e)}")
                X_train_fold = tf.gather(X_train, train_idx)
                y_train_fold = tf.gather(y_train, train_idx)
                X_val_fold = tf.gather(X_train, val_idx)
                y_val_fold = tf.gather(y_train, val_idx)
                logging.info(f"Training samples: {len(train_idx)}, Validation samples: {len(val_idx)}")
                if fold_idx == 1:
                    self.model.initialize_difficulty_sampler(X_train_fold.numpy(), y_train_fold.numpy())
                train_dataset = self.create_enhanced_dataset(X_train_fold, y_train_fold, is_training=True)
                val_dataset = self.create_enhanced_dataset(X_val_fold, y_val_fold, is_training=False)
                callbacks = [
                    ImprovedContinuousModelCheckpoint(self, fold_idx),
                    NaNDetectionCallback(patience=self.nan_detection_patience),
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_dice_manual',
                        patience=self.early_stopping_patience,
                        restore_best_weights=True,
                        verbose=1,
                        mode='max',
                        min_delta=0.001
                    ),
                    EnhancedTrainingMonitor(fold_idx),
                    EnhancedValidationMetricsCallback(val_dataset, self.output_dir, fold_idx, use_tta=False),
                    EnhancedVisualizationCallback(X_val_fold, y_val_fold, self.output_dir, fold_idx),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='val_dice_manual',
                        factor=0.7,
                        patience=10,
                        min_lr=1e-6,
                        verbose=1,
                        mode='max'
                    )
                ]
                logging.info(f"Starting training for fold {fold_idx + 1}...")
                try:
                    tf.keras.backend.clear_session()
                    self.compile_model()
                    history = self.model.fit(
                        train_dataset,
                        validation_data=val_dataset,
                        epochs=self.epochs,
                        callbacks=callbacks,
                        verbose=1,
                    )
                except tf.errors.ResourceExhaustedError as e:
                    logging.error(f"GPU memory exhausted in fold {fold_idx + 1}: {str(e)}")
                    tf.keras.backend.clear_session()
                    fold_results.append({
                        'fold': fold_idx + 1,
                        'catastrophic_failure': True,
                        'error': 'GPU memory exhausted',
                        'best_val_dice': 0.0,
                        'best_val_iou': 0.0,
                        'epochs_trained': 0
                    })
                    continue
                except Exception as e:
                    logging.error(f"Error training fold {fold_idx + 1}: {str(e)}")
                    fold_results.append({
                        'fold': fold_idx + 1,
                        'catastrophic_failure': True,
                        'error': str(e),
                        'best_val_dice': 0.0,
                        'best_val_iou': 0.0,
                        'epochs_trained': 0
                    })
                    continue
                self.save_enhanced_fold_history(history, fold_idx)
                fold_result = {
                    'fold': fold_idx + 1,
                    'catastrophic_failure': False,
                    'best_val_dice': max(history.history.get('val_dice_manual', [0])),
                    'best_val_iou': max(history.history.get('val_iou_manual', [0])),
                    'final_val_loss': history.history['val_loss'][-1] if history.history['val_loss'] else float('inf'),
                    'epochs_trained': len(history.history['loss']),
                    'best_val_loss': min(history.history.get('val_loss', [float('inf')]))
                }
                if np.isnan(fold_result['final_val_loss']) or np.isinf(fold_result['final_val_loss']):
                    logging.error(f"Fold {fold_idx + 1} ended with NaN/Inf loss!")
                    fold_result['catastrophic_failure'] = True
                fold_results.append(fold_result)
                logging.info(f"\n{'='*60}")
                logging.info(f"FOLD {fold_idx + 1} COMPLETED")
                logging.info(f"  Best val_loss: {fold_result['best_val_loss']:.4f}")
                logging.info(f"  Best val_dice: {fold_result['best_val_dice']:.4f}")
                logging.info(f"  Best val_iou: {fold_result['best_val_iou']:.4f}")
                logging.info(f"  Epochs trained: {fold_result['epochs_trained']}")
                logging.info(f"  Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
                logging.info(f"{'='*60}\n")
                if fold_result['best_val_loss'] < overall_best_val_loss:
                    overall_best_val_loss = fold_result['best_val_loss']
                if fold_result['best_val_dice'] > overall_best_val_dice:
                    overall_best_val_dice = fold_result['best_val_dice']
                    logging.info(f"New overall best validation dice: {overall_best_val_dice:.4f}")
                if fold_result['best_val_iou'] > overall_best_val_iou:
                    overall_best_val_iou = fold_result['best_val_iou']
                    logging.info(f"New overall best validation IoU: {overall_best_val_iou:.4f}")
            except Exception as e:
                logging.error(f"Unexpected error in fold {fold_idx + 1}: {str(e)}")
                fold_results.append({
                    'fold': fold_idx + 1,
                    'catastrophic_failure': True,
                    'error': str(e),
                    'best_val_dice': 0.0,
                    'best_val_iou': 0.0,
                    'epochs_trained': 0
                })
                continue
        self.plot_final_training_summary(fold_results)
        if os.path.exists(self.best_weights_path):
            try:
                self.model.load_weights(self.best_weights_path)
                logging.info(f"Loaded best weights from {self.best_weights_path} for final evaluation")
            except Exception as e:
                logging.error(f"Error loading best weights for final evaluation: {str(e)}")
        test_metrics = self.evaluate_model(X_test, y_test)
        total_time = (time.time() - start_time) / 3600.0
        final_summary = {
            'total_folds': self.val_folds,
            'successful_folds': len([f for f in fold_results if not f.get('catastrophic_failure', False)]),
            'overall_best_val_loss': float(overall_best_val_loss),
            'overall_best_val_dice': float(overall_best_val_dice),
            'overall_best_val_iou': float(overall_best_val_iou),
            'total_training_time_hours': float(total_time),
            'test_metrics': test_metrics,
            'fold_results': fold_results
        }
        with open(os.path.join(self.final_dir, 'training_summary.json'), 'w') as f:
            json.dump(final_summary, f)
        logging.info(f"\n{'='*60}")
        logging.info("TRAINING COMPLETED")
        logging.info(f"Overall best validation loss: {overall_best_val_loss:.4f}")
        logging.info(f"Overall best validation dice: {overall_best_val_dice:.4f}")
        logging.info(f"Overall best validation IoU: {overall_best_val_iou:.4f}")
        logging.info(f"Total training time: {total_time:.2f} hours")
        logging.info(f"Final memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        logging.info(f"Test metrics: {test_metrics}")
        logging.info(f"Saved final summary to {os.path.join(self.final_dir, 'training_summary.json')}")
        logging.info(f"{'='*60}\n")
        return fold_results