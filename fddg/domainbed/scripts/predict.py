#!/usr/bin/env python3

import argparse
import json
import os
import sys
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Add the parent directory to the path to import domainbed modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domainbed import datasets
from domainbed import algorithms
from domainbed.lib.fast_data_loader import FastDataLoader

def load_model(model_path, device):
    """Load a trained model from checkpoint."""
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # Get model configuration from checkpoint
    input_shape = checkpoint['model_input_shape']
    num_classes = checkpoint['model_num_classes']
    num_domains = checkpoint['model_num_domains']
    hparams = checkpoint['model_hparams']

    # Get algorithm class name from args
    algorithm_name = checkpoint['args']['algorithm']
    algorithm_class = algorithms.get_algorithm_class(algorithm_name)

    # Create and load the model
    algorithm = algorithm_class(input_shape, num_classes, num_domains, hparams)
    algorithm.load_state_dict(checkpoint['model_dict'])
    algorithm.to(device)
    algorithm.eval()

    return algorithm, checkpoint['args']

def predict_dataset(algorithm, dataset_path, device, batch_size=64):
    """Generate predictions for a dataset."""
    print(f"Processing dataset: {dataset_path}")

    # Create transform (should match training transform)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Load dataset
    from domainbed.datasets import SensitiveImageFolder
    dataset = SensitiveImageFolder(dataset_path, transform=transform)

    # Create data loader
    loader = FastDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=4
    )

    predictions = []

    with torch.no_grad():
        for batch_idx, (x, y, z) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            z = z.to(device)

            # Get predictions
            logits = algorithm.predict(x)
            probabilities = torch.softmax(logits, dim=1)
            predicted_classes = logits.argmax(1)

            # Process each sample in the batch
            for i in range(x.size(0)):
                sample_idx = batch_idx * batch_size + i

                # Get filename
                filename = "unknown"
                if sample_idx < len(dataset.samples):
                    filepath = dataset.samples[sample_idx][0]
                    filename = os.path.basename(filepath)

                # Create prediction record
                prediction_record = {
                    'filename': filename,
                    'filepath': dataset.samples[sample_idx][0] if sample_idx < len(dataset.samples) else "unknown",
                    'sample_index': sample_idx,
                    'true_label': y[i].item(),
                    'sensitive_attribute': z[i].item(),
                    'predicted_class': predicted_classes[i].item(),
                    'predicted_probabilities': probabilities[i].cpu().numpy().tolist(),
                    'is_person': predicted_classes[i].item() == 1,  # Assuming class 1 is "person"
                    'confidence': probabilities[i].max().item()
                }

                predictions.append(prediction_record)

    return predictions

def main():
    parser = argparse.ArgumentParser(description='Generate predictions from trained model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model (.pkl file)')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the dataset directory')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save predictions JSON file')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto)')

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"Using device: {device}")

    # Load model
    algorithm, model_args = load_model(args.model_path, device)

    # Generate predictions
    predictions = predict_dataset(algorithm, args.data_path, device, args.batch_size)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Save predictions
    output_data = {
        'model_path': args.model_path,
        'data_path': args.data_path,
        'model_args': model_args,
        'total_samples': len(predictions),
        'predictions': predictions
    }

    with open(args.output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Predictions saved to {args.output_path}")

    # Print summary
    total_samples = len(predictions)
    correct_predictions = sum(1 for p in predictions if p['predicted_class'] == p['true_label'])
    person_predictions = sum(1 for p in predictions if p['is_person'])

    print(f"\nSummary:")
    print(f"Total samples: {total_samples}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {correct_predictions / total_samples:.4f}")
    print(f"Person predictions: {person_predictions}")
    print(f"Person rate: {person_predictions / total_samples:.4f}")

    # Save summary separately
    summary_path = args.output_path.replace('.json', '_summary.json')
    summary = {
        'total_samples': total_samples,
        'correct_predictions': correct_predictions,
        'accuracy': correct_predictions / total_samples,
        'person_predictions': person_predictions,
        'person_rate': person_predictions / total_samples,
        'model_path': args.model_path,
        'data_path': args.data_path
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to {summary_path}")

if __name__ == '__main__':
    main()