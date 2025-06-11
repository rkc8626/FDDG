import os
import glob
import csv
import re

def extract_metrics_from_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Extract metrics using regex patterns
    metrics = {}

    # Extract accuracy metrics
    acc_pattern = r'(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)\s+(\d+\.\d+)'
    acc_match = re.search(acc_pattern, content)
    if acc_match:
        metrics.update({
            'env0_in_acc': acc_match.group(1),
            'env0_out_acc': acc_match.group(2),
            'env1_in_acc': acc_match.group(3),
            'env1_out_acc': acc_match.group(4),
            'epoch': acc_match.group(5),
            'loss': acc_match.group(6),
            'step': acc_match.group(7),
            'step_time': acc_match.group(8)
        })

    # Extract MD metrics
    md_pattern = r'env0_in_md\s+env0_out_md\s+env1_in_md\s+env1_out_md.*?(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)'
    md_match = re.search(md_pattern, content, re.DOTALL)
    if md_match:
        metrics.update({
            'env0_in_md': md_match.group(1),
            'env0_out_md': md_match.group(2),
            'env1_in_md': md_match.group(3),
            'env1_out_md': md_match.group(4)
        })

    # Extract DP metrics
    dp_pattern = r'env0_in_dp\s+env0_out_dp\s+env1_in_dp\s+env1_out_dp.*?(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)'
    dp_match = re.search(dp_pattern, content, re.DOTALL)
    if dp_match:
        metrics.update({
            'env0_in_dp': dp_match.group(1),
            'env0_out_dp': dp_match.group(2),
            'env1_in_dp': dp_match.group(3),
            'env1_out_dp': dp_match.group(4)
        })

    # Extract EO metrics
    eo_pattern = r'env0_in_eo\s+env0_out_eo\s+env1_in_eo\s+env1_out_eo.*?(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)'
    eo_match = re.search(eo_pattern, content, re.DOTALL)
    if eo_match:
        metrics.update({
            'env0_in_eo': eo_match.group(1),
            'env0_out_eo': eo_match.group(2),
            'env1_in_eo': eo_match.group(3),
            'env1_out_eo': eo_match.group(4)
        })

    # Extract AUC metrics
    auc_pattern = r'env0_in_auc\s+env0_out_auc\s+env1_in_auc\s+env1_out_auc.*?(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)'
    auc_match = re.search(auc_pattern, content, re.DOTALL)
    if auc_match:
        metrics.update({
            'env0_in_auc': auc_match.group(1),
            'env0_out_auc': auc_match.group(2),
            'env1_in_auc': auc_match.group(3),
            'env1_out_auc': auc_match.group(4)
        })

    return metrics

def main():
    # Get all txt files
    txt_files = glob.glob('*.txt')

    # Read existing summary.csv if it exists
    existing_data = []
    if os.path.exists('summary.csv'):
        with open('summary.csv', 'r') as f:
            reader = csv.DictReader(f)
            existing_data = list(reader)

    # Process each file
    for txt_file in txt_files:
        try:
            # Extract model name and hyperparameters from filename
            # Example: VREx_lr_5e-05_bs_64_wd_0.0001.txt
            parts = txt_file.replace('.txt', '').split('_')

            # Skip files that don't match the expected format
            if len(parts) < 7:
                print(f"Skipping {txt_file}: Filename doesn't match expected format")
                continue

            model = parts[0]
            lr = parts[2]
            bs = parts[4]
            wd = parts[6]

            # Extract metrics from file
            metrics = extract_metrics_from_file(txt_file)

            # Create row data
            row = {
                'model': model,
                'lr': lr,
                'bs': bs,
                'wd': wd,
                **metrics
            }

            # Check if this configuration already exists
            exists = False
            for existing_row in existing_data:
                if (existing_row['model'] == row['model'] and
                    existing_row['lr'] == row['lr'] and
                    existing_row['bs'] == row['bs'] and
                    existing_row['wd'] == row['wd']):
                    exists = True
                    break

            if not exists:
                existing_data.append(row)
                print(f"Added data from {txt_file}")

        except Exception as e:
            print(f"Error processing {txt_file}: {str(e)}")

    # Write updated data back to summary.csv
    if existing_data:
        fieldnames = [
            'model', 'lr', 'bs', 'wd',
            'env0_in_acc', 'env0_out_acc', 'env1_in_acc', 'env1_out_acc',
            'env0_in_md', 'env0_out_md', 'env1_in_md', 'env1_out_md',
            'env0_in_dp', 'env0_out_dp', 'env1_in_dp', 'env1_out_dp',
            'env0_in_eo', 'env0_out_eo', 'env1_in_eo', 'env1_out_eo',
            'env0_in_auc', 'env0_out_auc', 'env1_in_auc', 'env1_out_auc',
            'epoch', 'loss', 'step', 'step_time'
        ]

        with open('summary.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_data)
            print("Updated summary.csv successfully")

if __name__ == "__main__":
    main()