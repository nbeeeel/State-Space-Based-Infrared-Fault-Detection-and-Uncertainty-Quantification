import os
import glob
import random
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
from utils import calculate_iou_single

def load_image_and_mask(img_path, mask_path):
    try:
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.resize(img, [224, 224])
        img = tf.cast(img, tf.float32) / 255.0
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.resize(mask, [224, 224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        mask = tf.cast(mask, tf.float32) / 255.0
        mask = tf.where(mask > 0.5, 1.0, 0.0)
        mask = tf.stack([1.0 - mask[..., 0], mask[..., 0]], axis=-1)
        return img, mask
    except Exception as e:
        logging.error(f"Error loading image {img_path} or mask {mask_path}: {str(e)}")
        return None, None

def load_images_and_masks(image_directory, mask_directory, num_images):
    logging.info("Checking directories...")
    if not os.path.exists(image_directory):
        raise FileNotFoundError(f"Image directory {image_directory} not found")
    if not os.path.exists(mask_directory):
        raise FileNotFoundError(f"Mask directory {mask_directory} not found")
    image_names = sorted(glob.glob(os.path.join(image_directory, '*')))
    mask_names = sorted(glob.glob(os.path.join(mask_directory, '*')))
    logging.info(f"Found {len(image_names)} images and {len(mask_names)} masks")
    if len(image_names) < num_images or len(mask_names) < num_images:
        logging.warning(f"Insufficient images ({len(image_names)}) or masks ({len(mask_names)})")
        num_images = min(len(image_names), len(mask_names))
    image_names = image_names[:num_images]
    mask_names = mask_names[:num_images]
    valid_images = []
    valid_masks = []
    dataset = tf.data.Dataset.from_tensor_slices((image_names, mask_names))
    dataset = dataset.map(load_image_and_mask, num_parallel_calls=tf.data.AUTOTUNE)
    for i, (img, mask) in enumerate(dataset):
        try:
            if img is None or mask is None:
                logging.warning(f"Skipping image {image_names[i]} due to loading error")
                continue
            if i % 100 == 0:
                logging.info(f"Processing {i}/{num_images} images and masks")
            img_np = img.numpy()
            mask_np = mask.numpy()
            mask_sum = np.sum(mask_np[..., 1])
            if mask_sum == 0:
                logging.warning(f"Skipping image {image_names[i]}: mask has no fault pixels")
                continue
            valid_images.append(img_np)
            valid_masks.append(mask_np)
        except Exception as e:
            logging.error(f"Error processing image {image_names[i]}: {str(e)}")
            continue
    if not valid_images:
        raise ValueError("No valid images found after filtering")
    images = tf.convert_to_tensor(valid_images, dtype=tf.float32)
    masks = tf.convert_to_tensor(valid_masks, dtype=tf.float32)
    logging.info(f"Skipped {num_images - len(valid_images)} images due to empty masks or errors")
    logging.info(f"Final image data shape: {images.shape}")
    logging.info(f"Final mask shape: {masks.shape}")
    return images, masks

def load_and_split_data(image_directory, mask_directory, num_images):
    logging.info("Starting data loading...")
    images, masks = load_images_and_masks(image_directory, mask_directory, num_images)
    X_train, X_test, y_train, y_test = train_test_split(
        images.numpy(), masks.numpy(), test_size=0.3, random_state=42
    )
    logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logging.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    image_number = random.randint(0, len(X_train) - 1)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(X_train[image_number, :, :, 0], cmap='gray')
    plt.title('Sample Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(y_train[image_number, :, :, 1], cmap='gray')
    plt.title('Sample Mask')
    plt.axis('off')
    plt.show()
    return X_train, X_test, y_train, y_test

def predict_with_uncertainty(model, x, num_samples=10, mc_dropout=True):
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

def predict_with_tta(model, x, augmentations=['none', 'rot90', 'rot180', 'rot270']):
    predictions = []
    for aug in augmentations:
        if aug == 'none':
            x_aug = x
        elif aug == 'rot90':
            x_aug = tf.image.rot90(x, k=1)
        elif aug == 'rot180':
            x_aug = tf.image.rot90(x, k=2)
        elif aug == 'rot270':
            x_aug = tf.image.rot90(x, k=3)
        pred = model.predict(x_aug, verbose=0)
        if aug == 'rot90':
            pred = tf.image.rot90(pred, k=3)
        elif aug == 'rot180':
            pred = tf.image.rot90(pred, k=2)
        elif aug == 'rot270':
            pred = tf.image.rot90(pred, k=1)
        predictions.append(pred)
    return tf.reduce_mean(tf.stack(predictions), axis=0)

def morphological_cleanup(mask, kernel_size=3):
    mask_np = mask.numpy() if hasattr(mask, 'numpy') else mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_clean = cv2.morphologyEx(mask_np.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
    return mask_clean.astype(np.float32)