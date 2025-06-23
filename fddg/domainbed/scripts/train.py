import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

# os.environ["CUDA_VISIBLE_DEVICES"] = '6'

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
from torchvision.utils import save_image, make_grid
from domainbed.networks import load_munit_model


from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

import tensorboardX

def save_predictions_to_json(algorithm, dataset, device, output_dir, step=None):
    """Save predictions to JSON files using a simple dataloader for all images"""
    algorithm.eval()
    algorithm.to(device)

    all_predictions = []
    summary = {
        'total_samples': 0,
        'correct_predictions': 0,
        'accuracy': 0.0,
        'class_distribution': {},
        'confidence_stats': {
            'mean': 0.0,
            'std': 0.0,
            'min': 1.0,
            'max': 0.0
        }
    }

    confidences = []

    # Create a simple dataloader for all images across all environments
    all_samples = []
    for env_idx, env in enumerate(dataset):
        for sample_idx, sample in enumerate(env):
            # Add environment and sample information to each sample
            if len(sample) == 3:
                x, y, z = sample
            else:
                x, y = sample
                z = torch.zeros_like(y)  # Default sensitive attribute

            all_samples.append((x, y, z, env_idx, sample_idx))

    # Create a simple dataset class
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            x, y, z, env_idx, sample_idx = self.samples[idx]
            return x, y, z, env_idx, sample_idx

    # Create simple dataloader
    simple_dataset = SimpleDataset(all_samples)
    simple_loader = torch.utils.data.DataLoader(
        dataset=simple_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4
    )

    print(f"Processing {len(all_samples)} total samples across {len(dataset)} environments")

    with torch.no_grad():
        for batch_idx, batch in enumerate(simple_loader):
            x, y, z, env_indices, sample_indices = batch

            x = x.to(device)
            y = y.to(device)
            z = z.to(device)

            # Get predictions
            logits = algorithm.predict(x)
            probabilities = torch.softmax(logits, dim=1)
            predicted_classes = torch.argmax(logits, dim=1)
            confidences_batch = torch.max(probabilities, dim=1)[0]

            # Process each sample in the batch
            for i in range(len(x)):
                env_idx = env_indices[i].item()
                sample_idx = sample_indices[i].item()

                # Get filename and additional labels from the original dataset
                filename = None
                filepath = None
                additional_labels = {}

                # Try to get file information from the original dataset
                try:
                    original_env = dataset[env_idx]
                    if hasattr(original_env, 'samples') and sample_idx < len(original_env.samples):
                        filepath, _ = original_env.samples[sample_idx]
                        filename = os.path.basename(filepath)
                    elif hasattr(original_env, 'imgs') and sample_idx < len(original_env.imgs):
                        filepath, _ = original_env.imgs[sample_idx]
                        filename = os.path.basename(filepath)
                except Exception as e:
                    print(f"Warning: Could not get file info for env {env_idx}, sample {sample_idx}. Error: {e}")

                if filename is None:
                    filename = f"env{env_idx}_sample{sample_idx}.jpg"
                    filepath = ""

                # Try to get additional labels from dataset metadata
                try:
                    if hasattr(dataset, 'dict') and isinstance(dataset.dict, dict):
                        if filename in dataset.dict:
                            additional_labels_raw = dataset.dict[filename]
                            if isinstance(additional_labels_raw, list) and len(additional_labels_raw) >= 3:
                                additional_labels = {
                                    'timeofday': additional_labels_raw[0],
                                    'isperson': additional_labels_raw[1],
                                    'weather': additional_labels_raw[2]
                                }
                            elif isinstance(additional_labels_raw, dict):
                                additional_labels = additional_labels_raw
                except Exception as e:
                    print(f"Warning: Could not get additional labels for {filename}. Error: {e}")

                # Create prediction entry
                prediction = {
                    'filename': filename,
                    'filepath': filepath,
                    'predicted_class': predicted_classes[i].item(),
                    'predicted_probabilities': probabilities[i].cpu().numpy().tolist(),
                    'is_person': predicted_classes[i].item() == 1,
                    'confidence': confidences_batch[i].item(),
                    'true_label': y[i].item(),
                    'sensitive_attribute': z[i].item(),
                    'environment': env_idx,
                    'sample_index': sample_idx,
                    'timeofday': env_idx,
                    'isperson': y[i].item(),
                    'weather': z[i].item()
                }

                # Add additional labels if available
                if additional_labels:
                    prediction.update(additional_labels)

                all_predictions.append(prediction)
                confidences.append(confidences_batch[i].item())

                # Update summary statistics
                summary['total_samples'] += 1
                if predicted_classes[i].item() == y[i].item():
                    summary['correct_predictions'] += 1

                pred_class = predicted_classes[i].item()
                if pred_class not in summary['class_distribution']:
                    summary['class_distribution'][pred_class] = 0
                summary['class_distribution'][pred_class] += 1

    # Calculate final summary statistics
    if summary['total_samples'] > 0:
        summary['accuracy'] = summary['correct_predictions'] / summary['total_samples']

    if confidences:
        summary['confidence_stats']['mean'] = float(np.mean(confidences))
        summary['confidence_stats']['std'] = float(np.std(confidences))
        summary['confidence_stats']['min'] = float(np.min(confidences))
        summary['confidence_stats']['max'] = float(np.max(confidences))

    # Save predictions to JSON
    if step is not None:
        predictions_file = os.path.join(output_dir, f'step_{step}_predictions.json')
        summary_file = os.path.join(output_dir, f'step_{step}_predictions_summary.json')
    else:
        predictions_file = os.path.join(output_dir, 'predictions.json')
        summary_file = os.path.join(output_dir, 'predictions_summary.json')

    with open(predictions_file, 'w') as f:
        json.dump(all_predictions, f, indent=2)

    print(f"Predictions saved to {predictions_file}")
    print(f"Total predictions: {len(all_predictions)}")

    # Save summary
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Prediction summary saved to {summary_file}")
    algorithm.train()

if __name__ == "__main__":
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
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--save_predictions_every_checkpoint', action='store_true', help='Save predictions every checkpoint')
    parser.add_argument('--step', type=int, default=2,
                        help='Cotrain:2, Pretrain:3')
    args = parser.parse_args()


    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

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

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError

    # Debugging dataset splitting
    print("\nDebugging dataset splitting:")
    print(f"Dataset type: {type(dataset)}")
    print(f"Number of environments: {len(dataset)}")
    print(f"Test environments: {args.test_envs}")
    print(f"Holdout fraction: {args.holdout_fraction}")

    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []
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

    # Print sensitive attribute stats for each split
    def print_sensitive_stats(env):
        sensitive_values = []
        for sample in env:
            # sample: (x, y, z)
            if len(sample) == 3:
                z = sample[2]
                if isinstance(z, torch.Tensor):
                    z = z.item() if z.numel() == 1 else z.cpu().numpy().tolist()
                sensitive_values.append(z)
        if len(sensitive_values) > 0:
            unique, counts = np.unique(sensitive_values, return_counts=True)
            print("sensitive attribute stats:")
            print(f"  Unique values: {unique}")
            print(f"  Counts: {counts}")
            print(f"  Mean: {np.mean(sensitive_values):.4f}")
        else:
            print("No sensitive attribute found.")


    print_sensitive_stats(dataset[1])

    print_sensitive_stats(dataset[0])

    print_sensitive_stats(in_splits[0][0])

    print_sensitive_stats(in_splits[1][0])

    print_sensitive_stats(out_splits[0][0])

    print_sensitive_stats(out_splits[1][0])

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")
    print("Hparams: ", hparams)

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
        for i, (env, env_weights) in enumerate(uda_splits)
        if i in args.test_envs]

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

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)


    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    train_writer = tensorboardX.SummaryWriter(os.path.join(args.output_dir + "/logs", "tensorboard"))


    def write_loss(iterations, trainer, train_writer):
        train_writer.add_scalar("loss", trainer.loss, iterations + 1)
        train_writer.add_scalar("l_cls", trainer.l_cls, iterations + 1)
        if hasattr(trainer,'l_inv'):
            train_writer.add_scalar("l_inv", trainer.l_inv, iterations + 1)
        if hasattr(trainer,'l_fair'):
            train_writer.add_scalar("l_fair", trainer.l_fair, iterations + 1)
        train_writer.add_scalar("dual_var1", trainer.dual_var1, iterations + 1)
        train_writer.add_scalar("dual_var2", trainer.dual_var2, iterations + 1)

    def write_metrics(train_writer, metrics, step, prefix=""):
        """Write evaluation metrics to TensorBoard"""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                train_writer.add_scalar(f"{prefix}{key}", value, step + 1)

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.cpu().state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))


    last_results_keys = None
    for step in range(start_step, n_steps):
        step_start_time = time.time()
        # train minibathes train/uda/eval
        minibatches_device = [(x.to(device), y.to(device), z.to(device))
        for x, y, z in next(train_minibatches_iterator) if z is not None and y is not None and x is not None]
        if args.task == "domain_adaptation":
            uda_device = [x.to(device)
                for x,_ in next(uda_minibatches_iterator)]
        else:
            uda_device = None
        step_vals = algorithm.update(minibatches_device, uda_device)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)
        write_loss(step, algorithm, train_writer)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
        # if False:
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }
            mds = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }
            dps = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }
            eos = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }
            aucs = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)
                mds[key] = np.mean(val)
                dps[key] = np.mean(val)
                eos[key] = np.mean(val)
                aucs[key] = np.mean(val)

            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for idx, (name, loader, weights) in enumerate(evals):
                acc = misc.accuracy(algorithm, loader, weights, device)
                # Enable debug print for the first environment only
                debug_flag = (idx == 0)
                md = misc.md(algorithm, loader, weights, device, debug=debug_flag)
                dp = misc.dp(algorithm, loader, weights, device, debug=debug_flag)
                eo = misc.eo(algorithm, loader, weights, device, debug=debug_flag)
                t2 = time.time()
                auc = misc.auc(algorithm, loader, weights, device)

                results[name+'_acc'] = acc
                mds[name + '_md'] = md
                dps[name + '_dp'] = dp
                eos[name + '_eo'] = eo
                aucs[name + '_auc'] = auc

                # Log metrics to TensorBoard
                metrics_dict = {
                    'accuracy': acc,
                    'max_difference': md,
                    'demographic_parity': dp,
                    'equalized_odds': eo,
                    'auc_score': auc
                }
                write_metrics(train_writer, metrics_dict, step, prefix=f"{name}_")

            results_keys = sorted(results.keys())
            mds_keys = sorted(mds.keys())
            dps_keys = sorted(dps.keys())
            eos_keys = sorted(eos.keys())
            aucs_keys = sorted(aucs.keys())

            # print accuracy
            misc.print_row(results_keys, colwidth=12) # print name of each column
            last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                colwidth=12)

            # print Main Difference
            misc.print_row(mds_keys, colwidth=12)
            last_mds_keys = mds_keys
            misc.print_row([mds[key] for key in mds_keys],
                           colwidth=12)

            # print Demographic Parity
            misc.print_row(dps_keys, colwidth=12)
            last_dps_keys = dps_keys
            misc.print_row([dps[key] for key in dps_keys],
                           colwidth=12)

            # print Equalized Odds
            misc.print_row(eos_keys, colwidth=12)
            last_eos_keys = eos_keys
            misc.print_row([eos[key] for key in eos_keys],
                           colwidth=12)

            # print AUC
            misc.print_row(aucs_keys, colwidth=12)
            last_aucs_keys = aucs_keys
            misc.print_row([aucs[key] for key in aucs_keys],
                           colwidth=12)

            print()

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')

            if args.save_predictions_every_checkpoint:
                save_predictions_to_json(algorithm, dataset, device, args.output_dir, step=step)

    save_checkpoint('model.pkl')

    # Save predictions for all images
    if not args.save_predictions_every_checkpoint:
        save_predictions_to_json(algorithm, dataset, device, args.output_dir)

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')

    train_writer.close()
