import tensorflow as tf
import numpy as np
import os
import glob
import logging
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split

def load_image_and_mask(img_path, mask_path):
    """Load and preprocess a single image-mask pair"""
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
    """Load dataset from directories"""
    logging.info("Loading images and masks...")
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
                continue
            if i % 100 == 0:
                logging.info(f"Processing {i}/{num_images} images and masks")
            img_np = img.numpy()
            mask_np = mask.numpy()
            mask_sum = np.sum(mask_np[..., 1])
            if mask_sum == 0:
                continue
            valid_images.append(img_np)
            valid_masks.append(mask_np)
        except Exception as e:
            logging.error(f"Error processing image {i}: {str(e)}")
            continue
    if not valid_images:
        raise ValueError("No valid images found after filtering")
    images = tf.convert_to_tensor(valid_images, dtype=tf.float32)
    masks = tf.convert_to_tensor(valid_masks, dtype=tf.float32)
    logging.info(f"Final dataset: {images.shape[0]} samples")
    return images, masks

def load_and_split_data(image_directory, mask_directory, num_images, output_dir):
    """Load data and split into train/test sets"""
    logging.info("Loading and splitting data...")
    images, masks = load_images_and_masks(image_directory, mask_directory, num_images)
    X_train, X_test, y_train, y_test = train_test_split(
        images.numpy(), masks.numpy(), test_size=0.3, random_state=42
    )
    logging.info(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    image_number = random.randint(0, len(X_train) - 1)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(X_train[image_number, :, :, 0], cmap='gray')
    plt.title('Sample Image', fontsize=16)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(y_train[image_number, :, :, 1], cmap='gray')
    plt.title('Sample Mask', fontsize=16)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'sample_data.png'))
    plt.close()
    return X_train, X_test, y_train, y_test