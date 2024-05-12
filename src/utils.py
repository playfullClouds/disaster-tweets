import os
import sys
import dill
import joblib
import zipfile
import traceback
import urllib.request as request



from pathlib import Path
from src.logger import log
from ensure import ensure_annotations
from src.exception import CustomException
# from src.components.data_ingestion import DataIngestionConfig



def unzip_data(zip_dir: str, base_dir: str):
    """
    Unzips a ZIP file to a specified directory.
    Args:
        zip_file_path (str): The path to the ZIP file to extract.
        destination_path (str): The directory where the extracted data will be saved.
    """
    try:
        os.makedirs(base_dir, exist_ok=True)
        with zipfile.ZipFile(zip_dir, "r") as zip_ref:
            zip_ref.extractall(base_dir)
        log.info(f"Data extracted successfully to {base_dir}")
    except Exception as e:
        log.error(f"Failed to extract ZIP file: {e}")
        log.debug(traceback.format_exc())
        raise CustomException(e, sys)
            

   
@ensure_annotations         
def get_size(path: Path) -> str:
    """_summary_

    Args:
        path (Path): path of the file

    Returns:
        str: size in kb
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"


def download_file(source_url: str, zip_dir: str):
    """
    Downloads a file from a specified URL to a given directory.
    Args:
        source_url (str): The URL of the file to download.
        download_path (str): The path where the file will be saved.
    """
    if not os.path.exists(zip_dir):
        try: 
            filename, headers = request.urlretrieve(
                url = source_url,
                filename = zip_dir
            )
            log.info(f"{filename} downloaded with following info: \n{headers}")
        except Exception as e:
            log.error(f"Failed to download file from {source_url}: {e}")
            log.debug(traceback.format_exc())
            raise CustomException(e, sys)
    else:
        log.info(f"file already exists of sie: {get_size(Path(zip_dir))}")
        
        
        
# def load_object(file_path):
    
#     try:
#         with open(file_path, "rb") as file_obj:
#             # return dill.load(file_obj)
#             return joblib.load(file_obj)
        
#     except Exception as e:
#         raise CustomException(e, sys)


def load_object(file_path):
    try:
        log.info(f"Loading object from file: {file_path}")
        with open(file_path, "rb") as file_obj:
            return joblib.load(file_obj)
    except Exception as e:
        log.error(f"Error loading object from file: {e}")
        log.error(traceback.format_exc())  # Print the full traceback for debugging
        raise CustomException(e, sys)
