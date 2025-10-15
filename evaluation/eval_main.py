import os
import logging
import json
from eval_core import run_evaluation

def main():
    # Configuration dictionary
    config = {
        'img_height': 224,
        'img_width': 224,
        'img_channels': 1,
        'num_classes': 2,
        'batch_size': 8,
        'base_output_dir': './output/evaluation',  # Replace with your desired output directory
        'weights_dir': 'D:/Fault_Det_Umer/git/Modules/Evaluation/weight',                 # Replace with your weights directory
        'image_directory': 'D:/Fault_Det_Umer/git/Modules/images',     # Replace with your image dataset path
        'mask_directory': 'D:/Fault_Det_Umer/git/Modules/masks',       # Replace with your mask dataset path
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

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

    # Run evaluation
    logging.info("Starting evaluation process...")
    results = run_evaluation(config)

    # Print summary
    print("\n🎉 EVALUATION COMPLETED SUCCESSFULLY!")
    print(f"📁 Results saved to: {os.path.join(config['base_output_dir'], f'eval_{results['timestamp']}')}")
    print("📊 Check the visualizations and reports folders!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Evaluation failed: {str(e)}")
        raise