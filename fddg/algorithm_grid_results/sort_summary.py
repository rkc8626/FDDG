import csv
import pandas as pd

# Read the CSV file
df = pd.read_csv('summary.csv')

# Sort by model, lr, bs, wd
df_sorted = df.sort_values(['model', 'lr', 'bs', 'wd'])

# Write back to CSV
df_sorted.to_csv('summary.csv', index=False)
print("Sorted summary.csv successfully")