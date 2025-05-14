import argparse
import itertools
import json
import os
from subprocess import run
import numpy as np
from tqdm import tqdm
import torch

def get_base_param_grid():
    """Base hyperparameters for grid search"""
    return {
        'lr': [5e-5, 1e-4],           # 2 values - typical range for deep learning
        'batch_size': [32, 64],        # 2 values - common batch sizes
        'weight_decay': [0.0, 1e-4]    # 2 values - with/without regularization
    }

def get_algorithm_specific_params():
    """Algorithm-specific hyperparameters (using default values)"""
    return {
        'ERM': {},  # Base ERM has no additional params
        'IRM': {
            'irm_lambda': [1e2],  # default value
            'irm_penalty_anneal_iters': [500]  # default value
        },
        'GroupDRO': {
            'groupdro_eta': [1e-2]  # default value
        },
        'Mixup': {
            'mixup_alpha': [0.2]  # default value
        },
        'CORAL': {
            'mmd_gamma': [1.0]  # default value
        },
        'MMD': {
            'mmd_gamma': [1.0]  # default value
        },
        'VREx': {
            'vrex_lambda': [1e1],  # default value
            'vrex_penalty_anneal_iters': [500]  # default value
        },
        'MLDG': {
            'mldg_beta': [1.0]  # default value
        },
        'MBDG': {
            'mbdg_dual_step_size': [0.05],  # default value
            'mbdg_fair_step_size': [0.05],  # default value
            'mbdg_gamma1': [0.025],  # default value
            'mbdg_gamma2': [0.025]  # default value
        },
        'IGA': {
            'penalty': [1000]  # default value
        },
        'ANDMask': {
            'tau': [1.0]  # default value
        },
        'Fish': {
            'meta_lr': [0.5]  # default value
        },
        'SagNet': {
            'sag_w_adv': [0.1]  # default value
        }
    }

def get_param_combinations(algorithm):
    """Get all parameter combinations for a specific algorithm"""
    if algorithm == 'Fish':
        # Hard code the remaining combinations for Fish
        return [
            {'lr': 1e-4, 'batch_size': 64, 'weight_decay': 0.0, 'meta_lr': 0.5},
            {'lr': 1e-4, 'batch_size': 64, 'weight_decay': 1e-4, 'meta_lr': 0.5}
        ]

    # For other algorithms, use the original grid search
    base_params = get_base_param_grid()
    algorithm_params = get_algorithm_specific_params()[algorithm]

    # Combine base and algorithm-specific parameters
    all_params = {**base_params, **algorithm_params}

    # Generate all combinations
    keys = list(all_params.keys())
    values = list(all_params.values())
    combinations = list(itertools.product(*values))

    return [dict(zip(keys, combo)) for combo in combinations]

def run_experiment(args, algorithm, hparams):
    # Convert hparams to JSON string
    hparams_str = json.dumps(hparams)

    # Create unique output directory for this run
    run_name = f"{algorithm}_lr_{hparams['lr']}_bs_{hparams['batch_size']}_wd_{hparams['weight_decay']}"
    output_dir = os.path.join(args.base_output_dir, run_name)

    # Print detailed parameter configuration
    print("\n" + "="*50)
    print(f"Training Configuration for {algorithm}:")
    print("-"*50)
    print("Base Parameters:")
    print(f"  Learning Rate: {hparams['lr']}")
    print(f"  Batch Size: {hparams['batch_size']}")
    print(f"  Weight Decay: {hparams['weight_decay']}")

    # Print algorithm-specific parameters
    print("\nAlgorithm-Specific Parameters:")
    for key, value in hparams.items():
        if key not in ['lr', 'batch_size', 'weight_decay']:
            print(f"  {key}: {value}")
    print("="*50 + "\n")

    # Construct command
    cmd = [
        "python", "-m", "domainbed.scripts.train",
        f"--data_dir={args.data_dir}",
        f"--dataset={args.dataset}",
        f"--algorithm={algorithm}",
        f"--test_env={args.test_env}",
        f"--output_dir={output_dir}",
        f"--hparams={hparams_str}"
    ]

    # Run command
    process = run(cmd, capture_output=True, text=True)

    # Parse results from the output file
    results_file = os.path.join(output_dir, 'results.jsonl')
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            # Get the last line which contains final results
            for line in f:
                results = json.loads(line)
    else:
        print(f"Error: No results file found for {run_name}")
        print("Command output:", process.stdout)
        print("Command error:", process.stderr)
        return None

    # Extract relevant metrics
    metrics = {
        'accuracy': results.get(f'env{args.test_env}_out_acc', 0),
        'demographic_parity': results.get(f'env{args.test_env}_out_dp', 1),
        'equalized_odds': results.get(f'env{args.test_env}_out_eo', 1),
        'auc': results.get(f'env{args.test_env}_out_auc', 0)
    }

    # Calculate combined score
    combined_score = (
        metrics['accuracy'] * 0.4 +
        (1 - metrics['demographic_parity']) * 0.3 +
        (1 - metrics['equalized_odds']) * 0.3
    )

    metrics['combined_score'] = combined_score
    return metrics

def check_existing_result(output_dir):
    """Check if results already exist for a given configuration"""
    results_file = os.path.join(output_dir, 'results.jsonl')
    exists = os.path.exists(results_file)
    print(f"Checking for results in: {results_file}")
    print(f"File exists: {exists}")
    return exists

def main():
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU is NOT available. Running on CPU.")

    parser = argparse.ArgumentParser(description='Grid search for ERM variants')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to data directory')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name')
    parser.add_argument('--test_env', type=int, default=0,
                        help='Test environment index')
    parser.add_argument('--base_output_dir', type=str, default='grid_search_results',
                        help='Base directory for output')
    parser.add_argument('--algorithms', type=str, nargs='+',
        default=['ERM', 'IRM', 'GroupDRO', 'Mixup', 'CORAL', 'MMD', 'VREx', 'MLDG', 'MBDG', 'IGA', 'ANDMask', 'Fish', 'SagNet'],
        help='Algorithms to evaluate')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip configurations that have already been run')
    args = parser.parse_args()

    # Create base output directory
    os.makedirs(args.base_output_dir, exist_ok=True)

    # Store results
    all_results = []

    # Run experiments for each algorithm
    for algorithm in tqdm(args.algorithms, desc="Algorithms"):
        print(f"\n=== Running experiments for {algorithm} ===")

        # Get parameter combinations for this algorithm
        param_combinations = get_param_combinations(algorithm)

        # Run experiments for each parameter combination
        for hparams in tqdm(param_combinations, desc=f"{algorithm} parameters", leave=False):
            # Create unique output directory for this run
            run_name = f"{algorithm}_lr_{hparams['lr']}_bs_{hparams['batch_size']}_wd_{hparams['weight_decay']}"
            output_dir = os.path.join(args.base_output_dir, run_name)

            # Skip if results exist and skip_existing is True
            if args.skip_existing and check_existing_result(output_dir):
                print(f"\n######Skipping existing configuration######:")
                print(f"  Algorithm: {algorithm}")
                print(f"  Learning Rate: {hparams['lr']}")
                print(f"  Batch Size: {hparams['batch_size']}")
                print(f"  Weight Decay: {hparams['weight_decay']}")
                print(f"  Output Directory: {output_dir}")
                continue

            print(f"\nRunning {algorithm} with parameters:")
            print(f"  lr: {hparams['lr']}")
            print(f"  batch_size: {hparams['batch_size']}")
            print(f"  weight_decay: {hparams['weight_decay']}")

            metrics = run_experiment(args, algorithm, hparams)
            if metrics is not None:
                result = {
                    'algorithm': algorithm,
                    'hparams': hparams,
                    'metrics': metrics
                }
                all_results.append(result)

                print(f"Results:")
                print(f"  Algorithm: {algorithm}")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  Demographic Parity: {metrics['demographic_parity']:.4f}")
                print(f"  Equalized Odds: {metrics['equalized_odds']:.4f}")
                print(f"  AUC: {metrics['auc']:.4f}")
                print(f"  Combined Score: {metrics['combined_score']:.4f}")

    # Sort results by combined score
    all_results.sort(key=lambda x: x['metrics']['combined_score'], reverse=True)

    # Save all results
    results_file = os.path.join(args.base_output_dir, 'all_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print best results
    print("\nTop 5 configurations across all algorithms:")
    for i, result in enumerate(all_results[:5]):
        print(f"\n{i+1}. Algorithm: {result['algorithm']}")
        print("Parameters:")
        print(f"  lr: {result['hparams']['lr']}")
        print(f"  batch_size: {result['hparams']['batch_size']}")
        print(f"  weight_decay: {result['hparams']['weight_decay']}")
        print("Metrics:")
        for k, v in result['metrics'].items():
            print(f"  {k}: {v:.4f}")

    # Print best result for each algorithm
    print("\nBest configuration for each algorithm:")
    by_algorithm = {}
    for result in all_results:
        alg = result['algorithm']
        if alg not in by_algorithm or result['metrics']['combined_score'] > by_algorithm[alg]['metrics']['combined_score']:
            by_algorithm[alg] = result

    for alg, result in by_algorithm.items():
        print(f"\n=== {alg} ===")
        print("Parameters:")
        print(f"  lr: {result['hparams']['lr']}")
        print(f"  batch_size: {result['hparams']['batch_size']}")
        print(f"  weight_decay: {result['hparams']['weight_decay']}")
        print("Metrics:")
        for k, v in result['metrics'].items():
            print(f"  {k}: {v:.4f}")

if __name__ == '__main__':
    main()