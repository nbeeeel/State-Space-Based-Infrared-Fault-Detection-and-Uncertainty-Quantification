import os
import logging
from data_processing import load_and_split_data
from trainer import EnhancedSegmentationTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def main():
    # Configuration dictionary
    config = {
        'input_shape': (224, 224, 1),
        'num_classes': 2,
        'batch_size': 8,
        'epochs': 100,
        'output_dir': './output',
        'val_folds': 5,
        'initial_learning_rate': 1e-3,
        'lambda_uncertainty': 0.1,
        'physics_weight': 0.1,
        'focal_weight': 0.2,
        'boundary_weight': 0.3,
        'mixup_alpha': 0.2,
        'early_stopping_patience': 20,
        'nan_detection_patience': 3
    }

    # Specify dataset paths here
    image_directory = 'D:/Fault_Det_Umer/git/Modules/images'  # Replace with your image dataset path
    mask_directory = 'D:/Fault_Det_Umer/git/Modules/masks'    # Replace with your mask dataset path
    num_images = 464  # Adjust based on your dataset size

    # Load and split data
    logging.info("Loading and splitting dataset...")
    X_train, X_test, y_train, y_test = load_and_split_data(
        image_directory=image_directory,
        mask_directory=mask_directory,
        num_images=num_images
    )

    # Initialize trainer
    trainer = EnhancedSegmentationTrainer(config)

    # Train the model
    logging.info("Starting training...")
    fold_results = trainer.train(X_train, y_train, X_test, y_test)

    # Log final results
    logging.info("Training completed. Final fold results:")
    for result in fold_results:
        logging.info(result)

if __name__ == '__main__':
    main()
