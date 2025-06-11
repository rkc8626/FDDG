#  for d in */; do if [ -f "${d}out.txt" ]; then cp "${d}out.txt" "${d%/}.txt"; fi; done
# use above to extract the out.txt and rename it

import os
import glob

def get_last_n_nonblank_lines(file_path, n=9):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Filter out blank lines and strip whitespace
    nonblank_lines = [line.strip() for line in lines if line.strip()]

    # Get the last n lines
    last_n_lines = nonblank_lines[-n:] if len(nonblank_lines) >= n else nonblank_lines

    return last_n_lines

def main():
    # Get all txt files in the current directory
    txt_files = glob.glob('*.txt')

    # Open output file
    with open('extracted_results.txt', 'w') as out_file:
        # Process each file
        for txt_file in txt_files:
            out_file.write(f"\n=== {txt_file} ===\n")
            out_file.write("env0_in_acc   env0_out_acc  env1_in_acc   env1_out_acc  epoch         l_cls         loss          step          step_time\n")

            try:
                last_lines = get_last_n_nonblank_lines(txt_file)
                for line in last_lines:
                    out_file.write(line + "\n")
            except Exception as e:
                out_file.write(f"Error processing {txt_file}: {str(e)}\n")

            # Also print to console
            print(f"\nProcessing {txt_file}:")
            print("-" * 50)
            try:
                last_lines = get_last_n_nonblank_lines(txt_file)
                for line in last_lines:
                    print(line)
            except Exception as e:
                print(f"Error processing {txt_file}: {str(e)}")

if __name__ == "__main__":
    main()