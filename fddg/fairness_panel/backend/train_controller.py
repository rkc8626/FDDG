import json
import os
import subprocess
import signal
import time
from threading import Thread
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

        # Initialize shared state and lock BEFORE training manager
        self.state_lock = threading.Lock()
        self.shared_state = {
            'step': 0,
            'running': False,
            'hparams': hparams.copy(),
            'stdout': [],
            'metrics': {}
        }

        self.current_config = {
            'running': False,
            'batch_size': hparams.get('batch_size', 32),
            'learning_rate': hparams.get('lr', 0.001)
        }

        # Store the training manager instance that will be set by app.py
        self.training_manager = None

        self.tensorboard_dir = os.path.join(args.output_dir, "logs", "tensorboard")
        self.last_event_file = None
        self.last_step = 0

    def set_training_manager(self, manager):
        """Set the training manager instance and initialize it"""
        self.training_manager = manager
        self.training_manager.set_controller(self)

        # Add initial log message
        self.training_manager.add_output("\nTraining manager initialized with:")
        self.training_manager.add_output("=" * 50)
        self.training_manager.add_output(f"Dataset: {self.training_manager.args.dataset}")
        self.training_manager.add_output(f"Algorithm: {self.training_manager.args.algorithm}")
        self.training_manager.add_output(f"Test environments: {self.training_manager.args.test_envs}")
        self.training_manager.add_output(f"Initial batch size: {self.training_manager.hparams.get('batch_size', 32)}")
        self.training_manager.add_output(f"Initial learning rate: {self.training_manager.hparams.get('lr', 0.001)}")
        self.training_manager.add_output("Ready to start training. Click 'Start' to begin.")
        self.training_manager.add_output("=" * 50)

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

    def update_shared_state(self, updates):
        """Thread-safe update of the shared state and store update for debugging"""
        with self.state_lock:
            # Update the shared state with new values
            for key, value in updates.items():
                if key == 'stdout' and isinstance(value, str):
                    self.shared_state['stdout'].append(value)
                    # Print stdout for debugging evaluation
                    print(value)
                else:
                    self.shared_state[key] = value
                    if key == 'metrics':
                        print("\nMetrics Update:")
                        print("-" * 30)
                        for metric_name, metric_data in value.items():
                            if metric_data:  # Only print if there's data
                                print(f"{metric_name}: {metric_data[-1]}")  # Print latest value
                        print("-" * 30)

    def get_state(self) -> Dict[str, Any]:
        """Get current state including metrics"""
        try:
            # Get a thread-safe copy of the current state
            with self.state_lock:
                state_copy = {
                    'step': self.shared_state['step'],
                    'running': self.shared_state['running'],
                    'hparams': self.shared_state['hparams'].copy(),
                    # 'stdout': '\n'.join(self.shared_state['stdout']),
                    'metrics': self.shared_state.get('metrics', {})
                }

            # Add current configuration
            state_copy['batch_size'] = self.current_config['batch_size']
            state_copy['learning_rate'] = self.current_config['learning_rate']

            return state_copy

        except Exception as e:
            print(f"Error getting state: {str(e)}")
            return {
                'step': 0,
                'running': False,
                'hparams': {},
                'stdout': f"Error: {str(e)}",
                'metrics': {}
            }

