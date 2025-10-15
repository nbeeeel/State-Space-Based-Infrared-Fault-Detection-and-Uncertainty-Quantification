Physics-Aware U-Net Segmentation Project

A deep learning framework for image segmentation with physics-informed enhancements. This project implements a U-Net model with Vision Mamba blocks and custom loss functions, designed for segmenting grayscale images (e.g., fault detection in materials). The codebase supports both training and evaluation stages, leveraging TensorFlow for model development and evaluation.
📑 Table of Contents

Overview
Features
Project Structure
Prerequisites
Installation
Dataset Preparation
Execution Steps
Training Stage
Evaluation Stage


Outputs
Troubleshooting
Contributing
License


🌟 Overview
This project provides a robust pipeline for training and evaluating a U-Net-based segmentation model with physics-aware enhancements, including Fourier-based thermal diffusion layers and state-space models. The model is designed for binary segmentation tasks, incorporating custom losses (e.g., weighted binary cross-entropy, focal loss, boundary loss) and uncertainty estimation for improved generalization.

Training Stage: Trains the model using a dataset of images and corresponding masks, with k-fold cross-validation and physics-informed loss functions.
Evaluation Stage: Evaluates the trained model on a test set, computing comprehensive metrics (IoU, F1, precision, recall, etc.), generating visualizations, and producing a detailed report. The weights can be downloaded from [Weights File](https://drive.google.com/file/d/1oeRlIh2s-yFfEyOd1Qn4j4f5BIckMWvt/view?usp=sharing).

✨ Features

Physics-Aware U-Net: Integrates Vision Mamba blocks with Fourier-based thermal diffusion for physics-informed segmentation.
Custom Loss Functions:
Weighted binary cross-entropy for class imbalance.
Focal loss for hard example focus.
Boundary loss for enhanced edge detection.
Uncertainty loss for robust generalization.


Comprehensive Evaluation:
Metrics: IoU, F1, precision, recall, MCC, Cohen's Kappa, ROC-AUC, PR-AUC.
Visualizations: Input images, ground truth, predictions, thresholded predictions, and uncertainty maps.
Optimal threshold analysis for best performance.


Modular Design: Organized into separate modules for model, data processing, losses, metrics, and utilities.
Robust Error Handling: Extensive logging and validation to ensure reliable execution.
Flexible Configuration: Easily customizable via configuration dictionaries in main.py and eval_main.py.


📂 Project Structure
The project is organized into training and evaluation subdirectories for clarity. Below is the directory structure:
segmentation_project/
├── training/
│   ├── main.py                    # Main script for training
│   ├── trainer.py                # Training logic and k-fold cross-validation
│   ├── model.py                  # Model architecture (U-Net with Vision Mamba blocks)
│   ├── data_processing.py        # Data loading and preprocessing
│   ├── losses.py                 # Custom loss functions
├── evaluation/
│   ├── eval_main.py              # Main script for evaluation
│   ├── eval_model.py             # Model architecture for evaluation
│   ├── eval_metrics.py           # Metrics calculation and IoU metric
│   ├── eval_utils.py             # Utility functions (FLOPS, tensor validation, uncertainty visualization)
│   ├── eval_losses.py            # Custom loss functions for evaluation
│   ├── eval_data_processing.py   # Data loading and preprocessing for evaluation
│   ├── eval_core.py              # Core evaluation logic
├── dataset/                      # Dataset directory (user-provided)
│   ├── images/                   # Grayscale image files (*.png)
│   ├── masks/                    # Binary mask files (*.png)
├── output/                       # Output directory for training and evaluation
│   ├── training_YYYYMMDD_HHMMSS/ # Training outputs (weights, logs, plots)
│   ├── evaluation/eval_YYYYMMDD_HHMMSS/ # Evaluation outputs (predictions, visualizations, reports)
├── README.md                     # This file


🛠 Prerequisites
Ensure the following requirements are met before running the project:
Software

Python: Version 3.8 or higher
Operating System: Linux, macOS, or Windows
GPU (Optional): NVIDIA GPU with CUDA support for faster training (TensorFlow GPU version)

Python Libraries
Install the required libraries using pip:
pip install tensorflow==2.10.0 numpy matplotlib seaborn scikit-learn


TensorFlow: 2.10.0 (for model training and evaluation)
NumPy: For numerical operations
Matplotlib: For visualization
Seaborn: For enhanced visualization (e.g., confusion matrix)
Scikit-learn: For metrics calculation

Hardware

CPU: Multi-core processor (e.g., Intel i5 or higher)
RAM: At least 8GB (16GB recommended for larger datasets)
GPU (Optional): For accelerated training and evaluation
Storage: Sufficient space for dataset and output files (e.g., 10GB+)


📦 Installation

Clone the Repository (if applicable):
git clone <repository-url>
cd segmentation_project


Create Directories:Create the dataset and output directories if they don’t exist:
mkdir -p dataset/images dataset/masks output/evaluation


Install Dependencies:Install the required Python libraries:
pip install -r requirements.txt

If you don’t have a requirements.txt, create one with:
tensorflow==2.10.0
numpy
matplotlib
seaborn
scikit-learn


Organize Codebase:

Place training files (main.py, trainer.py, model.py, data_processing.py, losses.py) in the training/ directory.
Place evaluation files (eval_main.py, eval_model.py, eval_metrics.py, eval_utils.py, eval_losses.py, eval_data_processing.py, eval_core.py) in the evaluation/ directory.
Ensure all files are in their respective directories as per the Project Structure.




📊 Dataset Preparation
The dataset should consist of grayscale images and their corresponding binary masks in PNG format.

Directory Structure:
dataset/
├── images/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── masks/
│   ├── mask1.png
│   ├── mask2.png
│   └── ...


Requirements:

Images: Grayscale PNG files (1 channel, 8-bit or higher).
Masks: Binary PNG files (1 channel, 0 for background, 255 for foreground).
Naming: Image and mask files should have corresponding names (e.g., image1.png and mask1.png).
Quantity: At least 100 image-mask pairs recommended; adjust num_images in configuration if fewer.


Placement:

Place images in dataset/images/.
Place masks in dataset/masks/.
Ensure the paths are accessible and correctly specified in main.py and eval_main.py.




🚀 Execution Steps
Training Stage

Configure main.py:

Open training/main.py and update the config dictionary with your dataset paths and desired parameters:config = {
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
    'nan_detection_patience': 3,
    'image_directory': './dataset/images',  # Update with your image path
    'mask_directory': './dataset/masks',    # Update with your mask path
    'num_images': 163                       # Update based on your dataset
}




Run Training:

Navigate to the training/ directory:cd training


Execute the training script:python main.py




Training Process:

Loads images and masks from image_directory and mask_directory.
Performs k-fold cross-validation (default: 5 folds).
Trains the physics-aware U-Net model with custom losses.
Saves model weights, logs, and plots to a timestamped directory in output/ (e.g., output/training_YYYYMMDD_HHMMSS/).



Evaluation Stage

Configure eval_main.py:

Open evaluation/eval_main.py and update the config dictionary with your dataset and weights paths:config = {
    'img_height': 224,
    'img_width': 224,
    'img_channels': 1,
    'num_classes': 2,
    'batch_size': 8,
    'base_output_dir': './output/evaluation',  # Update with your output path
    'weights_dir': './output',                 # Update with your weights path
    'image_directory': './dataset/images',     # Update with your image path
    'mask_directory': './dataset/masks',       # Update with your mask path
    'num_images': 163,
    'd_model': 256,
    'd_model1': 512,
    'lambda_uncertainty': 0.5,
    'physics_weight': 0.5,
    'focal_weight': 0.3,
    'boundary_weight': 0.4,
    'mc_samples': 10,
    'tta_augmentations': ['hflip', 'vflip', 'rot90', 'rot180', 'rot270'],
    'num_visualizations': 5,
    'save_predictions': True
}




Run Evaluation:

Navigate to the evaluation/ directory:cd evaluation


Execute the evaluation script:python eval_main.py




Evaluation Process:

Loads test data from image_directory and mask_directory.
Loads the trained model weights from weights_dir.
Generates predictions with uncertainty estimation (Monte Carlo dropout and test-time augmentation).
Computes comprehensive metrics (IoU, F1, precision, recall, MCC, etc.).
Creates visualizations and a detailed report.
Saves outputs to a timestamped directory in base_output_dir (e.g., output/evaluation/eval_YYYYMMDD_HHMMSS/).




📈 Outputs
Training Outputs
Outputs are saved in output/training_YYYYMMDD_HHMMSS/:

Model Weights: final_enhanced_model.weights.h5, best_model.weights.h5, or fold-specific weights.
Logs: training.log with training progress and errors.
Plots: Loss curves, metric plots, and sample visualizations.
Results: JSON file with training metrics per fold.

Evaluation Outputs
Outputs are saved in output/evaluation/eval_YYYYMMDD_HHMMSS/:

Visualizations:
visualizations/sample_predictions.png: Comparison of input images, ground truth, predictions, thresholded predictions, and uncertainty maps.
visualizations/comprehensive_metrics.png: Plots of metrics, threshold analysis, IoU distribution, confusion matrix, and uncertainty distribution.


Predictions:
predictions/predictions.npy: Model predictions.
predictions/uncertainties.npy: Uncertainty estimates.
predictions/ground_truth.npy: Ground truth masks.


Reports:
reports/evaluation_report.txt: Detailed report with metrics, thresholds, and configuration.
evaluation_results.json: JSON summary of evaluation results.


Logs: evaluation.log with execution details.


🛠 Troubleshooting

FileNotFoundError: Ensure image_directory, mask_directory, and weights_dir paths are correct and accessible. Verify that image and mask files are in PNG format.
TensorFlow Errors: Check TensorFlow version compatibility (2.10.0 recommended). Ensure GPU drivers and CUDA are properly installed if using GPU.
Insufficient Images/Masks: Adjust num_images in main.py or eval_main.py to match your dataset size.
NaN in Tensors: The code includes NaN detection. Check logs for warnings and verify input data integrity.
Memory Issues: Reduce batch_size or num_images if running out of memory, especially on CPU.
Missing Weights: Ensure trained model weights are in weights_dir. Run training first if weights are missing.

For additional support, check the logs (training.log or evaluation.log) for detailed error messages.

🤝 Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a feature branch (git checkout -b feature/your-feature).
Commit changes (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a pull request.

Please include tests and update documentation as needed.

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.
