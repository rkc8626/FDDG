from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import json
import time
import argparse
from threading import Thread
from train_controller import TrainingController
from domainbed import algorithms, datasets, hparams_registry
import torch

app = Flask(__name__)
# Configure CORS properly for both REST and WebSocket
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "allow_headers": "*",
        "expose_headers": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "supports_credentials": True
    }
})

# Configure SocketIO with proper CORS and WebSocket settings
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    ping_timeout=60,
    ping_interval=25,
    async_mode='threading',
    logger=True,
    engineio_logger=True,
    always_connect=True,
    websocket=True
)

# Global training controller instance
training_controller = None

def parse_args():
    """Parse command line arguments similar to vanilla train script"""
    parser = argparse.ArgumentParser(description='Domain generalization with web monitoring')
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
    parser.add_argument('--step', type=int, default=2,
        help='Cotrain:2, Pretrain:3')
    parser.add_argument('--port', type=int, default=5100,
        help='Port for the web server')

    args = parser.parse_args()

    # Handle test environment argument compatibility
    if args.test_env is not None:
        args.test_envs = [args.test_env]
    elif args.test_envs is None:
        args.test_envs = [0]

    return args

def initialize_training(args):
    """Initialize the training controller with provided settings"""
    global training_controller

    # Get hyperparameters
    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset, args.test_envs, args.step)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed), args.test_envs)
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    # Initialize dataset
    dataset = vars(datasets)[args.dataset](args.data_dir, args.test_envs, hparams)

    # Initialize algorithm
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)

    if torch.cuda.is_available():
        algorithm = algorithm.cuda()

    # Create controller with output directory for TensorBoard logs
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Add webapp_mode to args
    args.webapp_mode = True

    training_controller = TrainingController(algorithm, dataset, hparams, args.test_envs, args=args)

def monitor_training_state():
    """Monitor training state and emit updates via socketio"""
    global training_controller

    while True:
        if training_controller:
            try:
                state = training_controller.get_state()

                # Format metrics for frontend
                metrics = {}
                if 'metrics' in state:
                    raw_metrics = state['metrics']

                    # Group metrics by type
                    for key, value in raw_metrics.items():
                        if isinstance(value, (int, float)):
                            # Determine metric group based on prefix
                            if key.startswith('env'):
                                if '_acc' in key:
                                    group = 'performance'
                                elif '_md' in key:
                                    group = 'fairness'
                                elif '_dp' in key:
                                    group = 'bias'
                                elif '_eo' in key:
                                    group = 'fairness'
                                elif '_auc' in key:
                                    group = 'performance'
                                else:
                                    group = 'others'
                            else:
                                group = 'raw'

                            if group not in metrics:
                                metrics[group] = {}
                            metrics[group][key] = value

                socketio.emit('state_update', {
                    'state': state,
                    'config': {
                        'running': state['running'],
                        'batch_size': state['hparams'].get('batch_size'),
                        'learning_rate': state['hparams'].get('lr')
                    }
                })

                if metrics:
                    socketio.emit('scalar_update', metrics)

            except Exception as e:
                print(f"Error in state monitoring: {e}")

        time.sleep(1)

@socketio.on('update_config')
def handle_config_update(new_config):
    """Handle configuration updates from the frontend"""
    global training_controller

    try:
        if training_controller:
            # Convert frontend config to hparams format
            if 'batch_size' in new_config:
                new_config['batch_size'] = int(new_config['batch_size'])
            if 'learning_rate' in new_config:
                new_config['lr'] = float(new_config['learning_rate'])
                del new_config['learning_rate']  # Remove frontend key

            training_controller.update_config(new_config)
            emit('config_updated', {'status': 'success'})
    except Exception as e:
        print(f"Error in config update: {e}")
        emit('config_updated', {'status': 'error', 'message': str(e)})

@socketio.on('request_state')
def handle_state_request():
    """Handle state requests from the frontend"""
    global training_controller

    try:
        if training_controller:
            state = training_controller.get_state()
            emit('state_update', {
                'state': state,
                'config': {
                    'running': state['running'],
                    'batch_size': state['hparams'].get('batch_size'),
                    'learning_rate': state['hparams'].get('lr')
                }
            })
    except Exception as e:
        emit('state_error', {'message': str(e)})

@app.route("/scalars", methods=["GET"])
def get_scalars():
    """Get current scalar metrics"""
    global training_controller

    if training_controller:
        state = training_controller.get_state()
        metrics = state.get('metrics', {})

        # Group metrics by type
        grouped_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                # Determine metric group based on prefix
                if key.startswith('env'):
                    if '_acc' in key:
                        group = 'performance'
                    elif '_md' in key:
                        group = 'fairness'
                    elif '_dp' in key:
                        group = 'bias'
                    elif '_eo' in key:
                        group = 'fairness'
                    elif '_auc' in key:
                        group = 'performance'
                    else:
                        group = 'others'
                else:
                    group = 'raw'

                if group not in grouped_metrics:
                    grouped_metrics[group] = {}
                grouped_metrics[group][key] = value

        return jsonify(grouped_metrics)
    return jsonify({})

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Initialize training controller with parsed args
    initialize_training(args)

    # Start the monitoring thread
    monitor_thread = Thread(target=monitor_training_state, daemon=True)
    monitor_thread.start()

    # Start Flask with SocketIO, binding to all interfaces with proper WebSocket support
    socketio.run(
        app,
        debug=True,
        host='0.0.0.0',
        port=args.port,
        use_reloader=False,
        allow_unsafe_werkzeug=True  # Required for proper WebSocket handling
    )