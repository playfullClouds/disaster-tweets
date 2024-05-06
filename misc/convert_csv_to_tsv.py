import csv
import os
from pathlib import Path

def convert_csv_to_tsv(csv_file, tsv_file):
    with open(csv_file, 'r', newline='', encoding='utf-8') as csv_in, \
         open(tsv_file, 'w', newline='', encoding='utf-8') as tsv_out:
        csv_reader = csv.reader(csv_in)
        tsv_writer = csv.writer(tsv_out, delimiter='\t')

        for row in csv_reader:
            tsv_writer.writerow(row)


def convert_all_csv_to_tsv(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    for csv_file in input_dir.glob('*.csv'):
        tsv_file = output_dir / (csv_file.stem + '.tsv')
        convert_csv_to_tsv(csv_file, tsv_file)


if __name__ == "__main__":
    input_directory = 'C:\\disaster-tweets\\artifacts\\data_ingestion\\data'  # Change this to your input directory path
    output_directory = 'C:\\disaster-tweets\\artifacts\\data_ingestion\\data'  # Change this to your output directory path
    convert_all_csv_to_tsv(input_directory, output_directory)
