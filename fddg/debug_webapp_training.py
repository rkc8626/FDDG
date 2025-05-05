import argparse
import os
import sys
import torch
import numpy as np
import threading
import queue
import time
from domainbed import datasets, algorithms, hparams_registry
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed.scripts.train_with_panel import TrainingManager, main as train_main
import torch.nn.functional as F

class MockController:
    """Mock controller to simulate the backend controller behavior"""
    def __init__(self):
        self.state_lock = threading.Lock()
        self.shared_state = {
            'step': 0,
            'running': False,
            'hparams': {},
            'stdout': [],
            'metrics': {}
        }
        self.updates = []  # Store all state updates for debugging

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

            # Store update for debugging
            self.updates.append({
                'timestamp': time.time(),
                'updates': updates.copy()
            })

class DebugTrainingManager(TrainingManager):
    def evaluate(self):
        """Run evaluation with enhanced debugging features"""
        torch.cuda.empty_cache()  # Clear GPU memory before evaluation
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        output = f"\nEvaluation at step {self.current_step + 1}:\n"
        output += "=" * 50 + "\n"

        results = {
            'step': self.current_step,
            'epoch': self.current_step / self.steps_per_epoch,
        }

        # Debug checkpoint values
        output += "\nCheckpoint Values:\n"
        output += "-" * 30 + "\n"
        for key, val in self.checkpoint_vals.items():
            try:
                mean_val = np.mean(val)
                if np.isnan(mean_val) or np.isinf(mean_val):
                    output += f"WARNING: {key} has invalid value: {mean_val}\n"
                    mean_val = 0.0  # fallback value
                results[key] = mean_val
                output += f"{key:20s}: {mean_val:.4f} (mean)\n"
            except Exception as e:
                output += f"Error processing checkpoint value {key}: {str(e)}\n"
                results[key] = 0.0

        # Evaluate on all loaders
        for idx, (name, loader, weights) in enumerate(zip(self.eval_loader_names, self.eval_loaders, self.eval_weights)):
            output += f"\n{name} Results:\n"
            output += "-" * 30 + "\n"

            print(f"\nLoader {idx} ({name}):")
            print(f"Loader type: {type(loader)}")

            # Try to get first batch
            print(f"\nAttempting to load first batch from {name}...")

            # Debug data statistics
            total_samples = 0
            nan_samples = 0
            inf_samples = 0

            for batch_idx, (x, y, z) in enumerate(loader):
                total_samples += x.size(0)

                # Input validation
                if torch.isnan(x).any():
                    nan_count = torch.isnan(x).sum().item()
                    nan_samples += nan_count
                    output += f"WARNING: {nan_count} NaN values found in input batch {batch_idx}\n"

                if torch.isinf(x).any():
                    inf_count = torch.isinf(x).sum().item()
                    inf_samples += inf_count
                    output += f"WARNING: {inf_count} Inf values found in input batch {batch_idx}\n"

                # Model prediction debugging
                try:
                    x = x.to(self.device)
                    y = y.to(self.device)

                    with torch.no_grad():
                        logits = self.algorithm.predict(x)

                        # Check prediction statistics
                        if torch.isnan(logits).any():
                            output += f"WARNING: NaN in model output at batch {batch_idx}\n"
                        if torch.isinf(logits).any():
                            output += f"WARNING: Inf in model output at batch {batch_idx}\n"

                        # Check prediction range
                        min_val, max_val = logits.min().item(), logits.max().item()
                        if min_val < -1e6 or max_val > 1e6:
                            output += f"WARNING: Unusual prediction range: [{min_val:.2e}, {max_val:.2e}]\n"

                except Exception as e:
                    output += f"Error during prediction: {str(e)}\n"
                    continue

            output += f"\nData Statistics for {name}:\n"
            output += f"Total samples: {total_samples}\n"
            if nan_samples > 0:
                output += f"WARNING: {nan_samples} total NaN values found\n"
            if inf_samples > 0:
                output += f"WARNING: {inf_samples} total Inf values found\n"

            # Compute metrics with enhanced error handling
            try:
                acc = misc.accuracy(self.algorithm, loader, weights, self.device)
                if np.isnan(acc) or np.isinf(acc):
                    output += f"WARNING: Invalid accuracy value: {acc}\n"
                    acc = 0.0
                results[name+'_acc'] = acc
                output += f"accuracy          : {acc:.4f}\n"
            except Exception as e:
                output += f"Error computing accuracy for {name}: {str(e)}\n"
                results[name+'_acc'] = 0.0

            # Additional metrics with enhanced error handling
            metrics_dict = {}
            for metric_name, metric_fn in [
                ('max_difference', misc.md),
                ('demographic_parity', misc.dp),
                ('equalized_odds', misc.eo),
                ('auc_score', misc.auc)
            ]:
                try:
                    value = metric_fn(self.algorithm, loader, weights, self.device)
                    if np.isnan(value) or np.isinf(value):
                        output += f"WARNING: Invalid {metric_name} value: {value}\n"
                        value = 0.0
                    metrics_dict[metric_name] = value
                except Exception as e:
                    output += f"Error computing {metric_name} for {name}: {str(e)}\n"
                    metrics_dict[metric_name] = 0.0

            for metric_name, value in metrics_dict.items():
                results[f"{name}_{metric_name}"] = value
                self.write_metrics({metric_name: value}, self.current_step, prefix=f"{name}_")
                output += f"{metric_name:20s}: {value:.4f}\n"

        # Memory usage statistics
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            memory_diff = final_memory - initial_memory
            output += f"\nGPU Memory Statistics:\n"
            output += f"Initial allocation : {initial_memory / 1024**2:.2f} MB\n"
            output += f"Final allocation   : {final_memory / 1024**2:.2f} MB\n"
            output += f"Memory difference : {memory_diff / 1024**2:.2f} MB\n"

            if memory_diff > 1024**3:  # If difference is more than 1GB
                output += "WARNING: Large memory increase detected during evaluation\n"

        output += "\nAll Metrics Summary:\n"
        output += "-" * 30 + "\n"
        for key, value in results.items():
            if isinstance(value, (int, float)):
                output += f"{key:20s}: {value:.4f}\n"

        output += "=" * 50
        self.add_output(output)
        print("Results at step " + str(self.current_step) + ": " + str(results))

        # Final cleanup
        torch.cuda.empty_cache()
        return results

def debug_webapp_training(args):
    """Debug the webapp training loop with mock controller"""
    print("\nInitializing webapp training debug...")
    print("=" * 50)

    # Add webapp mode to args
    args.webapp_mode = True
    args.save_model_every_checkpoint = False
    args.uda_holdout_fraction = 0
    args.task = "domain_generalization"

    # Use train_main to properly initialize everything
    training_manager = train_main(args)

    # Convert the regular training manager to debug version
    debug_manager = DebugTrainingManager(
        args=training_manager.args,
        hparams=training_manager.hparams,
        algorithm=training_manager.algorithm,
        dataset=training_manager.dataset,
        train_minibatches_iterator=training_manager.train_minibatches_iterator,
        eval_loaders=training_manager.eval_loaders,
        eval_weights=training_manager.eval_weights,
        eval_loader_names=training_manager.eval_loader_names,
        device=training_manager.device,
        is_webapp_mode=True
    )

    # Set up mock controller
    mock_controller = MockController()
    debug_manager.set_controller(mock_controller)

    print("\nStarting webapp training loop simulation...")
    print(f"Will evaluate every {args.checkpoint_freq} steps")
    print("=" * 50)

    # Start training
    debug_manager.start_webapp_training()

    try:
        # Let it run until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Stopping training...")
    finally:
        print("\nShutting down training manager...")
        debug_manager.shutdown_webapp()

    print("\nTraining simulation completed!")
    print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Debug webapp training loop')
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
    parser.add_argument('--output_dir', type=str, default="debug_webapp_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--step', type=int, default=2)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    debug_webapp_training(args)