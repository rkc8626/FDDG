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
from domainbed.scripts.train_with_panel import main as train_main
import torch
from tensorboard.backend.event_processing import event_file_loader
import glob
import tensorflow as tf

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
output_dir = "train_output"

def parse_args():
    global output_dir
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

    # Convert output_dir to absolute path
    args.output_dir = os.path.abspath(args.output_dir)
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    # Create logs directory for TensorBoard
    os.makedirs(os.path.join(args.output_dir, "logs", "tensorboard"), exist_ok=True)
    output_dir = args.output_dir
    return args

def initialize_training(args):
    """Initialize the training controller with provided settings"""
    global training_controller

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

def monitor_training_state():
    """Monitor training state and emit updates via socketio"""
    global training_controller
    previous_state = None

    while True:
        if training_controller:
            try:
                current_state = training_controller.get_state()

                # Only emit if state has changed
                if previous_state != current_state:
                    # Send state update with just training state
                    socketio.emit('state_update', {
                        'state': current_state,
                        'config': {
                            'running': current_state['running'],
                            'batch_size': current_state['hparams'].get('batch_size'),
                            'learning_rate': current_state['hparams'].get('lr')
                        }
                    })
                    previous_state = current_state

            except Exception as e:
                print(f"Error in state monitoring: {e}")

        time.sleep(4)

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
            del state['metrics']
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

def get_tensorboard_metrics(output_dir):
    """Read metrics from the TensorBoard event files in the correct directory"""
    # Use the same directory structure as in TrainingManager
    tensorboard_dir = os.path.join(output_dir, "logs", "tensorboard")
    print(f"Looking for TensorBoard files in: {tensorboard_dir}")

    # Check if directory exists
    if not os.path.exists(tensorboard_dir):
        print(f"TensorBoard directory does not exist: {tensorboard_dir}")
        return {}

    # Find all event files in the directory (not recursively, as the writer writes directly to this dir)
    event_files = glob.glob(os.path.join(tensorboard_dir, "events.out.tfevents.*"))
    print(f"Found {len(event_files)} event files: {event_files}")

    if not event_files:
        print("No event files found")
        return {}

    # Use only the latest event file based on creation time
    latest_event_file = max(event_files, key=os.path.getctime)
    print(f"Using latest event file: {latest_event_file}")

    metrics = {}
    event_count = 0

    # Process the latest event file
    try:
        # Use TensorFlow's summary iterator to read the events
        for summary in tf.compat.v1.train.summary_iterator(latest_event_file):
            for value in summary.summary.value:
                # Handle different value types
                if hasattr(value, 'simple_value'):
                    tag = value.tag
                    if tag not in metrics:
                        metrics[tag] = []

                    metrics[tag].append({
                        'step': summary.step,
                        'value': value.simple_value
                    })
                    event_count += 1
                    if event_count % 100 == 0:
                        print(f"Processed {event_count} events so far")
    except Exception as e:
        print(f"Error reading event file {latest_event_file}: {str(e)}")

    # Sort metrics by step
    for tag in metrics:
        metrics[tag].sort(key=lambda x: x['step'])

    print(f"Successfully loaded {len(metrics)} metrics with {event_count} total events")
    if metrics:
        print(f"Sample metrics: {list(metrics.keys())[:5]}")
    return metrics

@app.route("/scalars", methods=["GET"])
def get_scalars():
    """Get current scalar metrics directly from the training controller state"""
    global training_controller
    print("\nFetching scalars directly from training controller")

    if not training_controller:
        print("Training controller not initialized")
        return jsonify({})

    try:
        # Get the current state from the training controller
        state = training_controller.get_state()

        # Extract metrics from the state
        metrics = state.get('metrics', {})

        if not metrics:
            print("No metrics found in training controller state")
            return jsonify({})

        print(f"Found {len(metrics)} metrics in training controller state")

        # Group metrics by type
        grouped_metrics = {
            'performance': {},  # acc, auc
            'fairness': {},     # md, eo
            'bias': {},         # dp
            'training': {},     # loss, l_cls, step_time
            'progress': {}      # step, epoch
        }

        # Initialize radar data
        radar_data = {
            'is_radar': True,
            'values': {}
        }

        for key, values in metrics.items():
            if isinstance(values, list) and values:
                # Determine which group this metric belongs to
                if key in ['acc', 'auc']:
                    group = 'performance'
                elif key in ['md', 'eo']:
                    group = 'fairness'
                elif key == 'dp':
                    group = 'bias'
                elif key in ['loss', 'l_cls', 'step_time']:
                    group = 'training'
                elif key in ['step', 'epoch']:
                    group = 'progress'
                else:
                    continue  # Skip unknown metrics

                # Add to appropriate group
                grouped_metrics[group][key] = values

                # Add to radar data if it's a metric we want to show
                if key in ['acc', 'auc', 'md', 'dp', 'eo']:
                    # Get the latest value
                    latest_value = values[-1]['value'] if values else 0

                    # Normalize the value based on metric type
                    if key in ['acc', 'auc']:
                        # These are already in 0-1 range
                        normalized_value = latest_value
                    elif key == 'md':
                        # Max difference: 0 is best, 1 is worst
                        # We want to invert it so 1 is best
                        normalized_value = 1 - min(latest_value, 1)
                    elif key == 'dp':
                        # Demographic parity: 1 is best, 0 is worst
                        normalized_value = latest_value
                    elif key == 'eo':
                        # Equalized odds: 1 is best, 0 is worst
                        normalized_value = latest_value

                    radar_data['values'][key] = [{
                        'step': values[-1]['step'],
                        'value': normalized_value
                    }]

        # Add radar data to the response
        # Add an 'ideal' reference for radar chart (all metrics = 1.0)
        if radar_data['values']:
            latest_step = max(v[0]['step'] for v in radar_data['values'].values() if v)
            radar_data['ideal'] = [
                {'metric': k, 'step': latest_step, 'value': 1.0}
                for k in radar_data['values'].keys()
            ]
        grouped_metrics['radar'] = radar_data

        return jsonify(grouped_metrics)

    except Exception as e:
        print(f"Error getting metrics from training controller: {str(e)}")
        return jsonify({"error": str(e)})

@app.route("/representations", methods=["GET"])
def get_representations():
    """Get t-SNE visualization of training data representations"""
    global training_controller
    print("\nFetching training representations")

    if not training_controller or not training_controller.training_manager:
        print("Training controller not initialized")
        return jsonify({})

    try:
        # Get raw computation results from training manager
        raw_data = training_controller.training_manager.get_training_representations()

        if raw_data is None:
            return jsonify({"error": "No training data available"})

        # Validate raw data structure
        if 'points' not in raw_data or 'metadata' not in raw_data:
            print("Missing required keys in raw data")
            return jsonify({"error": "Invalid data format"})

        metadata = raw_data['metadata']
        points = raw_data['points']

        # Format data for visualization
        try:
            visualization_data = {
                'points': [{'x': float(x), 'y': float(y)} for x, y in points],
                'labels': metadata['labels'].tolist(),
                'sensitive': metadata['sensitive'].tolist(),
                'environments': metadata['environments'],
                'step': training_controller.training_manager.current_step,
                'env_sizes': metadata['env_sizes'],
                # Add unique sets for legend creation
                'labels_set': list(sorted(set(metadata['labels'].tolist()))),
                'sensitive_set': list(sorted(set(metadata['sensitive'].tolist()))),
                # Add a fairness guide: count of points per sensitive group
                'tsne_fairness_guide': {
                    str(group): int((metadata['sensitive'] == group).sum())
                    for group in set(metadata['sensitive'].tolist())
                },
                # Add predicted labels for class+senstive visualization
                'predicted_labels': metadata.get('predicted_labels', []).tolist() if 'predicted_labels' in metadata else [],
                'predicted_labels_set': list(sorted(set(metadata['predicted_labels'].tolist()))) if 'predicted_labels' in metadata and len(metadata['predicted_labels']) > 0 else []
            }

            # Log environment distribution
            print("\nEnvironment distribution:")
            for env_idx, size in metadata['env_sizes'].items():
                print(f"Environment {env_idx}: {size} samples")

            print(f"\nSuccessfully formatted {len(points)} data points for visualization")
            return jsonify(visualization_data)

        except Exception as format_error:
            error_msg = f"Error formatting data: {str(format_error)}"
            print(error_msg)
            return jsonify({"error": error_msg})

    except Exception as e:
        error_msg = f"Error getting training representations: {str(e)}"
        print(error_msg)
        return jsonify({"error": error_msg})

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