import argparse
import os
import sys
import torch
import numpy as np
import time
from domainbed import datasets, algorithms, hparams_registry
from domainbed.lib import misc
from domainbed.scripts.train_with_panel import main as train_main
from fairness_panel.backend.train_controller import TrainingController

def debug_training(args):
    """Debug training process without backend integration"""
    print("\nInitializing debug training...")
    print("=" * 50)

    # Add webapp mode to args
    args.webapp_mode = True
    args.save_model_every_checkpoint = False
    args.uda_holdout_fraction = 0
    args.task = "domain_generalization"

    # Use train_main to properly initialize everything
    training_manager = train_main(args)

    # Create training controller with the properly initialized components
    training_controller = TrainingController(
        algorithm=training_manager.algorithm,
        dataset=training_manager.dataset,
        hparams=training_manager.hparams,
        test_envs=args.test_envs,
        args=args
    )

    # Set the training manager in the controller
    training_controller.set_training_manager(training_manager)

    print("\nTraining initialization completed:")
    print(f"Dataset: {args.dataset}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Test environments: {args.test_envs}")
    print(f"Batch size: {training_manager.hparams['batch_size']}")
    print(f"Learning rate: {training_manager.hparams['lr']}")
    print("=" * 50)

    # Start training
    training_controller.start()

    try:
        # Let it run until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Stopping training...")
    finally:
        print("\nShutting down training manager...")
        training_controller.shutdown()

    print("\nTraining simulation completed!")
    print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Debug training without backend')
    parser.add_argument('--data_dir', type=str, default="./domainbed/data/")
    parser.add_argument('--dataset', type=str, default="CCMNIST1")
    parser.add_argument('--algorithm', type=str, default="MBDG")
    parser.add_argument('--hparams', type=str, help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0)
    parser.add_argument('--trial_seed', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--checkpoint_freq', type=int, default=None)
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="debug_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--step', type=int, default=2)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    debug_training(args)