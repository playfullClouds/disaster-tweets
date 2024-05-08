import os
import sys
import pandas as pd


from src.logger import log
from pathlib import Path
from dataclasses import dataclass
from src.exception import CustomException
from src.utils import unzip_data, download_file


@dataclass
class DataIngestionConfig:
    base_dir: str = os.path.join('artifacts', 'data_ingestion')
    source_url: str = "https://github.com/xplict33/mlproject/raw/main/data.zip"
    zip_dir: str = os.path.join(base_dir, 'data.zip')
    extracted_dir: str = os.path.join(base_dir, 'data')


class DataIngestion:
    def __init__(self) -> None:
        log.info("Initializing DataIngestion")
        self.config = DataIngestionConfig()

        try:
            # Create only the base directory since the zip file path is part of this directory.
            os.makedirs(self.config.base_dir, exist_ok=True)
            log.info(f"Created directory {self.config.base_dir}")
            
            log.info("Data Ingestion directory setup completed")
        except Exception as e:
            log.error("Failed to set up the data ingestion directories")
            raise CustomException(e, sys)
        
        
    def convert_tsv_to_csv_and_combine(self, directory: str, combined_csv_path: str):
        """Convert all TSV files to CSV format in the given directory."""
        log.info("Converting files from tsv to csv starting")
        
        
        try:
            # Check if any .tsv files are present
            tsv_files = [file for file in os.listdir(directory) if file.endswith('.tsv')]
            csv_files = [file_name for file_name in os.listdir(directory) if file_name.endswith('.csv')]
            
            # If only CSV files are present, skip conversion
            if len(tsv_files) == 0 and len(csv_files) > 0:
                log.info("All files are already in CSV format. Skipping TSV to CSV conversion.")
            else:
                # Convert each TSV file to CSV
                for file_name in tsv_files:
                    tsv_path = os.path.join(directory, file_name)
                    try:
                        # Read the TSV file into a DataFrame
                        df = pd.read_csv(tsv_path, sep='\t')
                        csv_file_name = file_name.replace('.tsv', '.csv')
                        csv_file_path = os.path.join(directory, csv_file_name)
                        # Write the DataFrame to a CSV file
                        df.to_csv(csv_file_path, index=False)
                        log.info(f"Converted {file_name} to {csv_file_name}")
                        
                        # Delete the original TSV file after successful conversion
                        log.info(F"Deleting the original TSV files: {file_name}")
                        os.remove(tsv_path)
                        log.info(f"Deleted the original TSV file: {file_name}")
                        
                    except Exception as e:
                        log.error(f"Error converting {file_name} to CSV: {e}")
                        
                        
            # Combine all CSV files into one DataFrame
            all_dataframes = [] # List to hold individual DataFrames for combining
            for file_name in os.listdir(directory):
                if file_name.endswith('.csv'):
                    file_path = os.path.join(directory, file_name)
                    try:
                        df = pd.read_csv(file_path)
                        all_dataframes.append(df)
                        log.info(f"Successfully loaded {file_path}")
                    except Exception as e:
                        log.error(f"Error reading {file_path}: {e}")

            if not all_dataframes:
                raise ValueError("No valid CSV files found to concatenate.")

            combined_df = pd.concat(all_dataframes, ignore_index=True)
            combined_df.to_csv(combined_csv_path, index=False)
            log.info(f"Combined CSV written to {combined_csv_path}")
        except Exception as e:
            log.error(f"Error converting TSV to CSV and combining data: {e}")
            raise CustomException(e, sys)
        
        
        
    def initiate_ingestion(self):  # Method to start the data ingestion process
        log.info("Data ingestion starting")  # Log entry
        try:  # Try block to catch exceptions
            
            # Check if the ZIP file already exists
            if os.path.exists(self.config.zip_dir):
                log.info(f"{self.config.zip_dir} already exists, skipping download")
            else:
                # Download the dataset using the download_file function from utils
                download_file(self.config.source_url, self.config.zip_dir)
                log.info(f"Downloaded dataset to {self.config.zip_dir}")


            # Check if the extracted directory already exists
            if os.path.exists(self.config.extracted_dir) and os.listdir(self.config.extracted_dir):
                log.info(f"{self.config.extracted_dir} already exists, skipping extraction")
            else:
                # Unzip the downloaded dataset into the base directory using unzip_data
                unzip_data(self.config.zip_dir, self.config.base_dir)
                log.info(f"Unzipped dataset to {self.config.base_dir}")
                
            
            log.info("Converting all TSV file to CSV and combine all starting")
            # Convert all TSV files to CSV format and combine them into one CSV file
            combined_csv_path = os.path.join(self.config.base_dir, 'tweets.csv')
            self.convert_tsv_to_csv_and_combine(self.config.extracted_dir, combined_csv_path)    


            log.info("Ingestion of the data is completed") # Log entry
            
            return self.config.base_dir # Return paths to the training, test, and validation data files
        
        except Exception as e: # Catch exceptions
            raise CustomException(e, sys) # Raise CustomException with the caught exception and sys module
        
