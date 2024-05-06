import sys


from src.logger import log
from src.exception import CustomException


from src.components.data_ingestion import DataIngestion
from src.components.data_cleaner import DataCleaner



  
STAGE_NAME = "Data Ingestion"
try:
   log.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   ingestion = DataIngestion()
   extracted_dir = ingestion.initiate_ingestion()
#    ingestion.execute_data_ingestion()
   log.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        log.exception(f"Exception occurred during {STAGE_NAME}")
        raise CustomException(e, sys)
    
    
    
STAGE_NAME = "Data Cleaning"
try:
    log.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
    cleaner = DataCleaner()
    cleaner.clean_file()
    log.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        log.exception(f"Exception occurred during {STAGE_NAME}")
        raise CustomException(e, sys)
    