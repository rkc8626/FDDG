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
        self.training_logs = []  # Store training logs

        # Calculate steps per epoch - fixed to handle dataset structure
        self.steps_per_epoch = min([len(dataset[i])/hparams['batch_size'] for i in range(len(dataset)) if i not in args.test_envs])

        # Web app specific attributes
        if is_webapp_mode:
            self.running = False
            self.command_queue = queue.Queue()
            self.state_queue = queue.Queue()
            self._stop_event = threading.Event()
            self.train_thread = None

        # TensorBoard writer
        self.train_writer = tensorboardX.SummaryWriter(
            os.path.join(args.output_dir + "/logs", "tensorboard"))

    def add_log(self, message):
        """Add a log message with timestamp"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_entry = f"[{timestamp}] {message}"
        self.training_logs.append(log_entry)
        # Keep only last 1000 log entries to prevent memory issues
        if len(self.training_logs) > 1000:
            self.training_logs = self.training_logs[-1000:]
        return log_entry

    def write_metrics(self, metrics, step, prefix=""):
        """Write evaluation metrics to TensorBoard"""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.train_writer.add_scalar(f"{prefix}{key}", value, step + 1)

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

        # Log step information
        log_message = f"Step {self.current_step}: "
        for key, val in step_vals.items():
            self.checkpoint_vals[key].append(val)
            self.train_writer.add_scalar(key, val, self.current_step + 1)
            log_message += f"{key}={val:.4f} "
        log_message += f"(took {step_time:.2f}s)"
        self.add_log(log_message)

        return step_vals

    def evaluate(self):
        """Run evaluation and return results"""
        self.add_log(f"\nEvaluation at step {self.current_step}:")
        results = {
            'step': self.current_step,
            'epoch': self.current_step / self.steps_per_epoch,
        }

        for key, val in self.checkpoint_vals.items():
            results[key] = np.mean(val)

        # Evaluate on all loaders
        for name, loader, weights in zip(self.eval_loader_names, self.eval_loaders, self.eval_weights):
            acc = misc.accuracy(self.algorithm, loader, weights, self.device)
            results[name+'_acc'] = acc

            # Additional metrics
            metrics_dict = {
                'accuracy': acc,
                'max_difference': misc.md(self.algorithm, loader, weights, self.device),
                'demographic_parity': misc.dp(self.algorithm, loader, weights, self.device),
                'equalized_odds': misc.eo(self.algorithm, loader, weights, self.device),
                'auc_score': misc.auc(self.algorithm, loader, weights, self.device)
            }

            # Log evaluation results
            log_message = f"{name} results: "
            for metric_name, value in metrics_dict.items():
                results[f"{name}_{metric_name}"] = value
                self.write_metrics({metric_name: value}, self.current_step, prefix=f"{name}_")
                log_message += f"{metric_name}={value:.4f} "
            self.add_log(log_message)

        return results

    def webapp_training_loop(self):
        """Training loop for web app mode"""
        self.add_log("Starting training loop...")

        while not self._stop_event.is_set():
            try:
                cmd = self.command_queue.get_nowait()
                if cmd['type'] == 'stop':
                    self.add_log("Received stop command")
                    break
                elif cmd['type'] == 'update_hparams':
                    self.add_log(f"Updating hyperparameters: {cmd['hparams']}")
                    self.update_hparams(cmd['hparams'])
            except queue.Empty:
                pass

            if not self.running:
                time.sleep(0.1)
                continue

            step_vals = self.train_step()
            self.current_step += 1

            # Send state update
            state_update = {
                'step': self.current_step,
                'running': self.running,
                'metrics': step_vals,
                'hparams': self.hparams,
                'stdout': '\n'.join(self.training_logs)  # Include logs in state update
            }
            try:
                self.state_queue.put_nowait(state_update)
            except queue.Full:
                pass

    def start_webapp_training(self):
        """Start training in web app mode"""
        if self.train_thread is not None and self.train_thread.is_alive():
            self.add_log("Training thread already running")
            return

        self._stop_event.clear()
        self.running = True
        self.add_log("Starting training thread...")
        self.train_thread = threading.Thread(target=self.webapp_training_loop)
        self.train_thread.start()

    def stop_webapp_training(self):
        """Stop training in web app mode"""
        self.add_log("Stopping training...")
        self.running = False

    def shutdown_webapp(self):
        """Shut down web app training"""
        self.add_log("Shutting down training manager...")
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
    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []

        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))

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

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    # Create data loaders
    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]

    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)]

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