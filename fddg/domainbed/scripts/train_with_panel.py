import argparse
import collections
import json
import os
import random
import sys
import time
import uuid
import threading
import queue
from typing import Dict, Any, Optional
import io
import contextlib

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

import tensorboardX

class TrainingManager:
    def __init__(self, args, hparams, algorithm, dataset, train_minibatches_iterator,
                 eval_loaders, eval_weights, eval_loader_names, device,
                 is_webapp_mode=False):
        self.args = args
        self.hparams = hparams
        self.algorithm = algorithm
        self.dataset = dataset
        self.train_minibatches_iterator = train_minibatches_iterator
        self.eval_loaders = eval_loaders
        self.eval_weights = eval_weights
        self.eval_loader_names = eval_loader_names
        self.device = device
        self.is_webapp_mode = is_webapp_mode

        # Training state
        self.current_step = 0
        self.checkpoint_vals = collections.defaultdict(lambda: [])
        self.training_output = []  # Store training output

        # Calculate steps per epoch - fixed to handle dataset structure
        self.steps_per_epoch = min([len(dataset[i])/hparams['batch_size'] for i in range(len(dataset)) if i not in args.test_envs])

        # Web app specific attributes
        if is_webapp_mode:
            self.running = False
            self.command_queue = queue.Queue()
            self.controller = None  # Will be set by the controller
            self._stop_event = threading.Event()
            self.train_thread = None

        # TensorBoard writer
        tensorboard_dir = os.path.join(args.output_dir, "logs", "tensorboard")
        print(f"Initializing TensorBoard writer at: {tensorboard_dir}")
        self.train_writer = tensorboardX.SummaryWriter(tensorboard_dir)

    def set_controller(self, controller):
        """Set the controller reference for state updates"""
        self.controller = controller

    def add_output(self, message):
        """Add a message to training output"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        output = f"[{timestamp}] {message}"
        self.training_output.append(output)
        # Keep only last 1000 lines to prevent memory issues
        if len(self.training_output) > 1000:
            self.training_output = self.training_output[-1000:]
        # print(output)  # Also print to console

        # Update controller's shared state if available
        if self.is_webapp_mode and self.controller:
            self.controller.update_shared_state({'stdout': output})

        return output

    def write_metrics(self, metrics, step, prefix=""):
        """Write evaluation metrics to TensorBoard"""
        for key, value in metrics.items():
            tag = f"{prefix}{key}"
            self.train_writer.add_scalar(tag, value, step + 1)
            # print(f"Writing metric to TensorBoard: {tag} = {value} at step {step + 1}")

    def save_checkpoint(self, filename):
        if self.args.skip_model_save:
            return
        save_dict = {
            "args": vars(self.args),
            "model_input_shape": self.dataset.input_shape,
            "model_num_classes": self.dataset.num_classes,
            "model_num_domains": len(self.dataset) - len(self.args.test_envs),
            "model_hparams": self.hparams,
            "model_dict": self.algorithm.cpu().state_dict()
        }
        torch.save(save_dict, os.path.join(self.args.output_dir, filename))
        self.algorithm.to(self.device)

    def update_hparams(self, new_hparams):
        """Update hyperparameters and reinitialize if needed"""
        old_batch_size = self.hparams.get('batch_size')
        self.algorithm = self.algorithm.reinit_with_new_hparams(new_hparams)
        self.hparams = new_hparams

        if old_batch_size != new_hparams.get('batch_size'):
            self._reinitialize_data_loaders()

    def _reinitialize_data_loaders(self):
        """Reinitialize data loaders with new batch size"""
        in_splits = []
        for env_i, env in enumerate(self.dataset):
            if env_i not in self.args.test_envs:
                in_weights = None
                in_splits.append((env, in_weights))

        self.train_loaders = [InfiniteDataLoader(
            dataset=env,
            weights=env_weights,
            batch_size=self.hparams['batch_size'],
            num_workers=self.dataset.N_WORKERS)
            for i, (env, env_weights) in enumerate(in_splits)]

        self.train_minibatches_iterator = zip(*self.train_loaders)

    def train_step(self):
        """Execute a single training step"""
        step_start_time = time.time()

        minibatches_device = [(x.to(self.device), y.to(self.device), z.to(self.device))
            for x, y, z in next(self.train_minibatches_iterator)]

        step_vals = self.algorithm.update(minibatches_device)
        step_time = time.time() - step_start_time
        self.checkpoint_vals['step_time'].append(step_time)

        # Only format and add output at checkpoint frequency
        checkpoint_freq = self.args.checkpoint_freq or self.dataset.CHECKPOINT_FREQ
        if (self.current_step + 1) % checkpoint_freq == 0:
            # Format step output nicely
            output = f"\nStep {self.current_step + 1}:\n"
            output += "=" * 50 + "\n"
            output += "Training Metrics:\n"
            output += "-" * 30 + "\n"
            for key, val in step_vals.items():
                output += f"{key:20s}: {val:.4f}\n"
            output += f"Step time: {step_time:.2f}s\n"
            output += "=" * 50
            self.add_output(output)

        # Store values without printing
        for key, val in step_vals.items():
            self.checkpoint_vals[key].append(val)
            self.train_writer.add_scalar(key, val, self.current_step + 1)

        return step_vals

    def evaluate(self):
        """Run evaluation and return results"""
        output = f"\nEvaluation at step {self.current_step + 1}:\n"
        output += "=" * 50 + "\n"

        # Initialize results with only step and epoch
        results = {
            'step': self.current_step,
            'epoch': self.current_step / self.steps_per_epoch,
        }

        output += "\nCheckpoint Values:\n"
        output += "-" * 30 + "\n"
        for key, val in self.checkpoint_vals.items():
            mean_val = np.mean(val)
            results[key] = mean_val
            output += f"{key:20s}: {mean_val:.4f} (mean)\n"

        # Initialize aggregated metrics
        aggregated_metrics = {
            'acc': {'values': [], 'weights': []},  # Weighted average
            'md': {'values': []},                  # Maximum
            'dp': {'values': [], 'weights': []},   # Weighted average
            'eo': {'values': [], 'weights': []},   # Weighted average
            'auc': {'values': [], 'weights': []}   # Weighted average
        }

        # Evaluate on all loaders
        for name, loader, weights in zip(self.eval_loader_names, self.eval_loaders, self.eval_weights):
            output += f"\n{name} Results:\n"
            output += "-" * 30 + "\n"

            # Get dataset size for weighting
            dataset_size = len(loader)
            output += f"Dataset size: {dataset_size}\n"

            # Compute metrics with enhanced error handling
            try:
                acc = misc.accuracy(self.algorithm, loader, weights, self.device)
                if np.isnan(acc) or np.isinf(acc):
                    output += f"WARNING: Invalid accuracy value: {acc}\n"
                    acc = 0.0
                output += f"accuracy          : {acc:.4f}\n"

                # Add to aggregated metrics
                aggregated_metrics['acc']['values'].append(acc)
                aggregated_metrics['acc']['weights'].append(dataset_size)
            except Exception as e:
                output += f"Error computing accuracy for {name}: {str(e)}\n"

            # Additional metrics with enhanced error handling
            metrics_dict = {}
            for metric_name, metric_fn in [
                ('md', misc.md),
                ('dp', misc.dp),
                ('eo', misc.eo),
                ('auc', misc.auc)
            ]:
                try:
                    value = metric_fn(self.algorithm, loader, weights, self.device)
                    if np.isnan(value) or np.isinf(value):
                        output += f"WARNING: Invalid {metric_name} value: {value}\n"
                        value = 0.0
                    metrics_dict[metric_name] = value

                    # Add to aggregated metrics
                    if metric_name == 'md':
                        # For max difference, just collect values
                        aggregated_metrics[metric_name]['values'].append(value)
                    else:
                        # For other metrics, use weighted average
                        aggregated_metrics[metric_name]['values'].append(value)
                        aggregated_metrics[metric_name]['weights'].append(dataset_size)
                except Exception as e:
                    output += f"Error computing {metric_name} for {name}: {str(e)}\n"

            for metric_name, value in metrics_dict.items():
                output += f"{metric_name:20s}: {value:.4f}\n"

        # Compute final aggregated metrics
        for metric_name, data in aggregated_metrics.items():
            if not data['values']:
                continue

            if metric_name == 'md':
                # For max difference, take the maximum value
                aggregated_value = max(data['values'])
            else:
                # For other metrics, compute weighted average
                values = np.array(data['values'])
                weights = np.array(data['weights'])
                aggregated_value = np.average(values, weights=weights)

            # Only add aggregated metrics to results
            results[metric_name] = aggregated_value
            self.write_metrics({metric_name: aggregated_value}, self.current_step)
            output += f"\nAggregated {metric_name}: {aggregated_value:.4f}\n"

        output += "\nAll Metrics Summary:\n"
        output += "-" * 30 + "\n"
        for key, value in results.items():
            if isinstance(value, (int, float)):
                output += f"{key:20s}: {value:.4f}\n"

        output += "=" * 50
        self.add_output(output)
        print("Results at step " + str(self.current_step) + ": " + str(results))
        return results

    def webapp_training_loop(self):
        """Training loop for web app mode"""
        self.add_output("\nStarting training loop...")
        self.add_output("=" * 50)
        self.add_output(f"Total steps per epoch: {self.steps_per_epoch}")
        self.add_output(f"Target epochs: {self.args.steps / self.steps_per_epoch if self.args.steps else 'Not specified'}")
        self.add_output("=" * 50)

        n_steps = self.args.steps or self.dataset.N_STEPS
        checkpoint_freq = self.args.checkpoint_freq or self.dataset.CHECKPOINT_FREQ

        # Initialize metrics storage
        all_metrics = {}

        # Update controller with initial state
        if self.controller:
            self.controller.update_shared_state({
                'step': self.current_step,
                'running': self.running,
                'hparams': self.hparams,
                'metrics': all_metrics
            })

        while not self._stop_event.is_set():
            try:
                cmd = self.command_queue.get_nowait()
                if cmd['type'] == 'stop':
                    self.add_output("\nReceived stop command")
                    break
                elif cmd['type'] == 'update_hparams':
                    # Always show hparam updates
                    self.add_output(f"\nUpdating hyperparameters:")
                    self.add_output("=" * 50)
                    for key, value in cmd['hparams'].items():
                        self.add_output(f"{key:20s}: {value}")
                    self.add_output("=" * 50)
                    self.update_hparams(cmd['hparams'])

                    # Update controller with new hparams
                    if self.controller:
                        self.controller.update_shared_state({'hparams': self.hparams})
            except queue.Empty:
                pass

            if not self.running:
                time.sleep(1)
                continue

            step_vals = self.train_step()
            self.current_step += 1

            # Store training metrics but only update controller at checkpoint frequency
            for key, val in step_vals.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append({
                    'step': self.current_step,
                    'value': val
                })

            # Update controller with step (but not metrics or stdout) every step
            if self.controller:
                self.controller.update_shared_state({
                    'step': self.current_step,
                    'running': self.running,
                })

            # Check if we've reached the target number of steps
            if self.current_step >= n_steps:
                self.add_output(f"\nReached target number of steps ({n_steps})")
                self.add_output("Stopping training...")
                self.running = False

                # Update controller with running state and final metrics
                if self.controller:
                    self.controller.update_shared_state({
                        'running': False,
                        'metrics': all_metrics
                    })

                break

            # Run evaluation and update metrics at checkpoint frequency
            if (self.current_step % checkpoint_freq == 0) or (self.current_step == n_steps - 1):
                results = self.evaluate()
                print("Results at step " + str(self.current_step) + ": " + str(results))

                # Store evaluation metrics
                for key, val in results.items():
                    if isinstance(val, (int, float)) or isinstance(val,(int, int)):
                        if key not in all_metrics:
                            all_metrics[key] = []
                        all_metrics[key].append({
                            'step': self.current_step,
                            'value': val
                        })

                # Update controller with new metrics and stdout at checkpoint
                if self.controller:
                    self.controller.update_shared_state({
                        'metrics': all_metrics,
                        'stdout': self.training_output[-checkpoint_freq:]  # Only send recent stdout
                    })

                # Save results
                results.update({
                    'hparams': self.hparams,
                    'args': vars(self.args)
                })

                epochs_path = os.path.join(self.args.output_dir, 'results.jsonl')
                with open(epochs_path, 'a') as f:
                    f.write(json.dumps(results, sort_keys=True) + "\n")

                self.checkpoint_vals = collections.defaultdict(lambda: [])

                if self.args.save_model_every_checkpoint:
                    self.save_checkpoint(f'model_step{self.current_step}.pkl')

        # Save final checkpoint
        self.save_checkpoint('model.pkl')
        with open(os.path.join(self.args.output_dir, 'done'), 'w') as f:
            f.write('done')

    def start_webapp_training(self):
        """Start training in web app mode"""
        if self.train_thread is not None and self.train_thread.is_alive():
            print("Training thread already running")
            return

        self._stop_event.clear()
        self.running = True
        print("Starting training thread...")
        self.train_thread = threading.Thread(target=self.webapp_training_loop)
        self.train_thread.start()

    def stop_webapp_training(self):
        """Stop training in web app mode"""
        print("Stopping training...")
        self.running = False

    def shutdown_webapp(self):
        """Shut down web app training"""
        print("Shutting down training manager...")
        self._stop_event.set()
        if self.train_thread:
            self.train_thread.join()

    def train(self):
        """Main training loop for standalone mode"""
        if self.is_webapp_mode:
            raise ValueError("Cannot use train() in webapp mode. Use webapp_training_loop() instead.")

        n_steps = self.args.steps or self.dataset.N_STEPS
        checkpoint_freq = self.args.checkpoint_freq or self.dataset.CHECKPOINT_FREQ

        for step in range(self.current_step, n_steps):
            self.current_step = step
            step_vals = self.train_step()

            if (step % checkpoint_freq == 0) or (step == n_steps - 1):
                results = self.evaluate()

                # Save results
                results.update({
                    'hparams': self.hparams,
                    'args': vars(self.args)
                })

                epochs_path = os.path.join(self.args.output_dir, 'results.jsonl')
                with open(epochs_path, 'a') as f:
                    f.write(json.dumps(results, sort_keys=True) + "\n")

                self.checkpoint_vals = collections.defaultdict(lambda: [])

                if self.args.save_model_every_checkpoint:
                    self.save_checkpoint(f'model_step{step}.pkl')

        self.save_checkpoint('model.pkl')
        with open(os.path.join(self.args.output_dir, 'done'), 'w') as f:
            f.write('done')

    def get_training_representations(self):
        """Get t-SNE visualization of training data representations"""
        try:
            # Collect representations from all training environments
            all_reps = []
            all_labels = []
            all_sensitive = []
            all_envs = []

            for env_idx, (name, loader, _) in enumerate(zip(self.eval_loader_names, self.eval_loaders, self.eval_weights)):
                if 'in' in name:  # Only use training data
                    reps = []
                    labels = []
                    sensitive = []

                    # Get representations from the model
                    with torch.no_grad():
                        for x, y, z in loader:
                            x = x.to(self.device)
                            # Get representations from the model
                            rep = self.algorithm.featurizer(x)
                            reps.append(rep.cpu().numpy())
                            labels.append(y.cpu().numpy())
                            sensitive.append(z.cpu().numpy())

                    if reps:
                        all_reps.append(np.concatenate(reps))
                        all_labels.append(np.concatenate(labels))
                        all_sensitive.append(np.concatenate(sensitive))
                        all_envs.extend([env_idx] * len(reps[0]))

            if not all_reps:
                return None

            # Concatenate all representations
            all_reps = np.concatenate(all_reps)
            all_labels = np.concatenate(all_labels)
            all_sensitive = np.concatenate(all_sensitive)

            # Apply t-SNE
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=42)
            reps_2d = tsne.fit_transform(all_reps)

            # Prepare data for frontend
            visualization_data = {
                'points': reps_2d.tolist(),
                'labels': all_labels.tolist(),
                'sensitive': all_sensitive.tolist(),
                'environments': all_envs,
                'step': self.current_step
            }

            return visualization_data

        except Exception as e:
            print(f"Error computing training representations: {str(e)}")
            return None

def main(provided_args=None):
    """Main function that can accept args from command line or from another script"""
    if provided_args is None:
        parser = argparse.ArgumentParser(description='Domain generalization')
        parser.add_argument('--data_dir', type=str, default="./domainbed/data/")
        parser.add_argument('--dataset', type=str, default="CCMNIST1")
        parser.add_argument('--algorithm', type=str, default="MBDG")
        parser.add_argument('--task', type=str, default="domain_generalization",
            choices=["domain_generalization", "domain_adaptation"])
        parser.add_argument('--hparams', type=str,
            help='JSON-serialized hparams dict')
        parser.add_argument('--hparams_seed', type=int, default=0,
            help='Seed for random hparams (0 means "default hparams")')
        parser.add_argument('--trial_seed', type=int, default=0,
            help='Trial number (used for seeding split_dataset and random_hparams).')
        parser.add_argument('--seed', type=int, default=0,
            help='Seed for everything else')
        parser.add_argument('--steps', type=int, default=None,
            help='Number of steps. Default is dataset-dependent.')
        parser.add_argument('--checkpoint_freq', type=int, default=None,
            help='Checkpoint every N steps. Default is dataset-dependent.')
        # Support both --test_env and --test_envs for backward compatibility
        group = parser.add_mutually_exclusive_group()
        group.add_argument('--test_env', type=int, help='Legacy single test environment')
        group.add_argument('--test_envs', type=int, nargs='+', help='Test environment(s)')
        parser.add_argument('--output_dir', type=str, default="train_output")
        parser.add_argument('--holdout_fraction', type=float, default=0.2)
        parser.add_argument('--uda_holdout_fraction', type=float, default=0,
            help="For domain adaptation, % of test to use unlabeled for training.")
        parser.add_argument('--skip_model_save', action='store_true')
        parser.add_argument('--save_model_every_checkpoint', action='store_true')
        parser.add_argument('--webapp_mode', action='store_true',
            help='Run in web app mode')
        parser.add_argument('--step', type=int, default=2,
            help='Cotrain:2, Pretrain:3')
        args = parser.parse_args()
    else:
        args = provided_args

    # Handle test environment argument compatibility
    if hasattr(args, 'test_env') and args.test_env is not None:
        args.test_envs = [args.test_env]
    elif not hasattr(args, 'test_envs') or args.test_envs is None:
        args.test_envs = [0]

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    # Print environment info
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset, args.test_envs, args.step)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed), args.test_envs)
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Initialize dataset and algorithm
    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError

    # Split dataset
    print("\nDebugging dataset splitting:")
    print(f"Dataset type: {type(dataset)}")
    print(f"Number of environments: {len(dataset)}")
    print(f"Test environments: {args.test_envs}")
    print(f"Holdout fraction: {args.holdout_fraction}")

    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        print(f"\nProcessing environment {env_i}:")
        print(f"Environment size: {len(env)}")

        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        print(f"Split sizes - in: {len(in_)}, out: {len(out)}")

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))
            print(f"Test env split - in: {len(in_)}, uda: {len(uda)}")

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    print("\nFinal split sizes:")
    print(f"In splits: {[len(x[0]) for x in in_splits]}")
    print(f"Out splits: {[len(x[0]) for x in out_splits]}")
    print(f"UDA splits: {[len(x[0]) for x in uda_splits]}")

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    # Create data loaders
    print("\nCreating data loaders:")
    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]
    print(f"Created {len(train_loaders)} training loaders")

    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)]
    print(f"Created {len(uda_loaders)} UDA loaders")

    # Check class distribution in evaluation data
    print("\nChecking class distribution in evaluation data:")
    for i, (env, _) in enumerate(in_splits + out_splits + uda_splits):
        if i < len(in_splits):
            split_type = "in"
        elif i < len(in_splits) + len(out_splits):
            split_type = "out"
        else:
            split_type = "uda"

        # Count classes
        class_counts = {}
        for _, y, _ in env:
            y = y.item() if isinstance(y, torch.Tensor) else y
            class_counts[y] = class_counts.get(y, 0) + 1

        print(f"\nSplit {i} ({split_type}):")
        print(f"Total samples: {len(env)}")
        print("Class distribution:")
        for cls, count in sorted(class_counts.items()):
            print(f"  Class {cls}: {count} samples ({count/len(env)*100:.1f}%)")

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]

    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]
    print("\nEvaluation loader names:", eval_loader_names)

    train_minibatches_iterator = zip(*train_loaders)

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)

    algorithm.to(device)

    # Initialize training manager
    training_manager = TrainingManager(
        args=args,
        hparams=hparams,
        algorithm=algorithm,
        dataset=dataset,
        train_minibatches_iterator=train_minibatches_iterator,
        eval_loaders=eval_loaders,
        eval_weights=eval_weights,
        eval_loader_names=eval_loader_names,
        device=device,
        is_webapp_mode=args.webapp_mode
    )

    if args.webapp_mode:
        return training_manager
    else:
        training_manager.train()
        training_manager.train_writer.close()

if __name__ == "__main__":
    main()