import os
import argparse
import glob

def rename_checkpoints(input_dir, output_dir, env, step):
    """
    Rename checkpoint files from gen_XXXXXXXX.pt format to {env}_step{step}.pt format

    Args:
        input_dir: Directory containing the original checkpoint files
        output_dir: Directory where renamed files will be saved
        env: Environment number(s) as a string (e.g., "0" or "01")
        step: Training step (1, 2, or "cotrain_step1")
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find the latest checkpoint file
    gen_files = glob.glob(os.path.join(input_dir, 'gen_*.pt'))
    if not gen_files:
        print(f"No checkpoint files found in {input_dir}")
        return

    # Sort files by iteration number (extracted from filename)
    gen_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    latest_gen = gen_files[-1]

    # Create new filename
    new_filename = f"{env}_step{step}.pt"
    new_path = os.path.join(output_dir, new_filename)

    # Copy the file with new name
    print(f"Copying {latest_gen} to {new_path}")
    with open(latest_gen, 'rb') as src, open(new_path, 'wb') as dst:
        dst.write(src.read())

    print(f"Successfully renamed checkpoint to {new_filename}")

def main():
    parser = argparse.ArgumentParser(description='Rename MUNIT checkpoint files')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Directory containing the original checkpoint files')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory where renamed files will be saved')
    parser.add_argument('--env', type=str, required=True,
                      help='Environment number(s) (e.g., "0" or "01")')
    parser.add_argument('--step', type=str, required=True,
                      help='Training step (1, 2, or "cotrain_step1")')

    args = parser.parse_args()
    rename_checkpoints(args.input_dir, args.output_dir, args.env, args.step)

if __name__ == '__main__':
    main()