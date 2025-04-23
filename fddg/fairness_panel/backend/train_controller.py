import json
import os
import subprocess
import signal
import time
from threading import Thread
from utils import load_json_atomic, file_lock, update_json_atomic
import threading
import queue
from typing import Dict, Any, Optional
import torch
import sys

# Add the parent directory to Python path to import domainbed
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from domainbed.scripts.train_with_panel import main as train_main


# Global variables
training_process = None

class TrainingController:
    def __init__(self, algorithm, dataset, hparams, test_envs, args=None):
        """Initialize the training controller

        Args:
            algorithm: The algorithm instance
            dataset: The dataset instance
            hparams: Initial hyperparameters
            test_envs: List of test environment indices
            args: Optional argparse.Namespace with training configuration
        """
        if args is None:
            # Create default args if none provided
            import argparse
            args = argparse.Namespace()
            args.data_dir = "./domainbed/data/"
            args.dataset = dataset.__class__.__name__
            args.algorithm = algorithm.__class__.__name__
            args.task = "domain_generalization"
            args.hparams = None
            args.hparams_seed = 0
            args.trial_seed = 0
            args.seed = 0
            args.steps = None
            args.checkpoint_freq = None
            args.test_envs = test_envs
            args.output_dir = os.path.join(os.getcwd(), "train_output")
            args.holdout_fraction = 0.2
            args.skip_model_save = False
            args.save_model_every_checkpoint = False
            args.webapp_mode = True
            args.step = 2

        # Initialize training manager in webapp mode
        args.webapp_mode = True  # Ensure webapp mode is set
        self.training_manager = train_main(args)

        # Add initial log message
        self.training_manager.add_log(f"Training manager initialized with:")
        self.training_manager.add_log(f"- Dataset: {args.dataset}")
        self.training_manager.add_log(f"- Algorithm: {args.algorithm}")
        self.training_manager.add_log(f"- Test environments: {args.test_envs}")
        self.training_manager.add_log(f"- Initial batch size: {hparams.get('batch_size', 32)}")
        self.training_manager.add_log(f"- Initial learning rate: {hparams.get('lr', 0.001)}")
        self.training_manager.add_log("Ready to start training. Click 'Start' to begin.")

        self.current_config = {
            'running': False,
            'batch_size': hparams.get('batch_size', 32),
            'learning_rate': hparams.get('lr', 0.001)
        }

    def start(self):
        """Start the training process"""
        self.training_manager.start_webapp_training()

    def stop(self):
        """Stop the training process"""
        self.training_manager.stop_webapp_training()

    def shutdown(self):
        """Completely shut down the controller"""
        self.training_manager.shutdown_webapp()

    def update_config(self, config: Dict[str, Any]):
        """Update configuration"""
        if 'running' in config:
            if config['running']:
                self.start()
            else:
                self.stop()

        # Update hyperparameters if changed
        hparams_changed = False
        new_hparams = self.training_manager.hparams.copy()

        if 'batch_size' in config and config['batch_size'] != self.current_config['batch_size']:
            new_hparams['batch_size'] = config['batch_size']
            hparams_changed = True

        if 'lr' in config and config['lr'] != self.current_config['learning_rate']:
            new_hparams['lr'] = config['lr']
            hparams_changed = True

        if hparams_changed:
            self.training_manager.command_queue.put({
                'type': 'update_hparams',
                'hparams': new_hparams
            })

        self.current_config.update(config)

    def get_state(self) -> Dict[str, Any]:
        """Get current state"""
        try:
            while True:
                state = self.training_manager.state_queue.get_nowait()
                latest_state = state
        except queue.Empty:
            latest_state = {
                'step': self.training_manager.current_step,
                'running': self.training_manager.running,
                'hparams': self.training_manager.hparams,
                'metrics': {},  # Ensure metrics key exists
                'stdout': '\n'.join(self.training_manager.training_logs)  # Include logs even when queue is empty
            }

        # Add current configuration
        latest_state['batch_size'] = self.current_config['batch_size']
        latest_state['learning_rate'] = self.current_config['learning_rate']

        # Ensure stdout exists in state
        if 'stdout' not in latest_state:
            latest_state['stdout'] = '\n'.join(self.training_manager.training_logs)

        return latest_state

