import os
import csv

path_to_raw_file = 'data/raw_data.txt'
path_to_csv_file = 'data/demo2.csv'

with open(path_to_csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["text"])
    with open(path_to_raw_file, 'r') as raw_file:
        raw_file = raw_file.read().split('<s>')
        for item in raw_file:
            writer.writerow([f'<s>{item.strip()}'])
