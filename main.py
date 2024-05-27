import sys


from src.logger import log
from src.exception import CustomException


# Plain machine Learning
# from src.components.data_ingestion import DataIngestion
# from src.components.data_cleaner import DataCleaner
# from src.components.data_transformation import DataTransformer
# from src.components.model_trainer import ModelTrainer
# from src.components.model_optimizer import ModelOptimizer






# Plain SMOTE Machine Learning
from src.components.data_ingestion import DataIngestion
from src.components.data_cleaner import DataCleaner
from src.components.data_transformation_SMOTE import DataTransformerSMOTE
from src.components.model_trainer import ModelTrainer
from src.components.model_optimizer_SMOTE import ModelOptimizerSMOTE

def run_stage(stage_name, stage_function):
    log.info(f">>>>>> stage {stage_name} started <<<<<<")
    try:
        stage_function()
        log.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")
    except Exception as e:
        log.exception(f"Exception occurred during {stage_name}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    stages = [
        ("Data Ingestion", lambda: DataIngestion().initiate_ingestion()),
        ("Data Cleaning", lambda: DataCleaner().clean_file()),
        ("Data Transformation SMOTE", lambda: DataTransformerSMOTE().transform_data()),
        ("Model Trainer SMOTE", lambda: ModelTrainer().cross_validation_and_test()),
        ("Model Optimization SMOTE", lambda: ModelOptimizerSMOTE().run())
    ]

    for stage_name, stage_function in stages:
        run_stage(stage_name, stage_function)












# # Plain BERT Tensorflow
# from src.components.data_ingestion import DataIngestion
# from src.components.data_cleaning_BERT import DataCleaner
# from src.components.data_transformation_BERT_tf import DataTransformerPlainBERT
# from src.components.model_trainer_BERT_tf import ModelTrainerPlainBERT


# STAGE_NAME = "Data Ingestion"
# try:
#    log.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#    ingestion = DataIngestion()
#    extracted_dir = ingestion.initiate_ingestion()
# #    ingestion.execute_data_ingestion()
#    log.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         log.exception(f"Exception occurred during {STAGE_NAME}")
#         raise CustomException(e, sys)
    
    
# STAGE_NAME = "Data Cleaning BERT"
# try:
#     log.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#     cleaner = DataCleaner()
#     cleaner.clean_file()
#     log.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         log.exception(f"Exception occurred during {STAGE_NAME}")
#         raise CustomException(e, sys)
    
       
# STAGE_NAME = "Data Transformation BERT TF"
# try:
#     log.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#     transformer = DataTransformerPlainBERT()
#     transformer.transform_data()
#     log.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         log.exception(f"Exception occurred during {STAGE_NAME}")
#         raise CustomException(e, sys)


# STAGE_NAME = "Model Trainer BERT TF"
# try:
#     log.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#     trainer = ModelTrainerPlainBERT()
#     trainer.train()
#     # trainer.evaluate_model()
#     log.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         log.exception(f"Exception occurred during {STAGE_NAME}")
#         raise CustomException(e, sys)










# # SMOTE BERT Tensorflow
# from src.components.data_ingestion import DataIngestion
# from src.components.data_cleaning_BERT import DataCleaner
# from src.components.data_transformation_BERT_tf_SMOTE import DataTransformerTfSMOTE
# from src.components.model_trainer_BERT_tf import ModelTrainerPlainBERT
# from src.components.model_optimizer_BERT_optuna import ModelOptimizerOptuna



# STAGE_NAME = "Data Ingestion"
# try:
#    log.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#    ingestion = DataIngestion()
#    extracted_dir = ingestion.initiate_ingestion()
# #    ingestion.execute_data_ingestion()
#    log.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         log.exception(f"Exception occurred during {STAGE_NAME}")
#         raise CustomException(e, sys)
    
    
# STAGE_NAME = "Data Cleaning BERT"
# try:
#     log.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#     cleaner = DataCleaner()
#     cleaner.clean_file()
#     log.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         log.exception(f"Exception occurred during {STAGE_NAME}")
#         raise CustomException(e, sys)


# STAGE_NAME = "Data Transformation BERT TF"
# try:
#     log.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#     transformer = DataTransformerTfSMOTE()
#     transformer.run_transformation()
#     log.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         log.exception(f"Exception occurred during {STAGE_NAME}")
#         raise CustomException(e, sys)


# STAGE_NAME = "Model Trainer BERT TF"
# try:
#     log.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#     trainer = ModelTrainerPlainBERT()
#     trainer.train()
#     # trainer.evaluate_model()
#     log.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         log.exception(f"Exception occurred during {STAGE_NAME}")
#         raise CustomException(e, sys)


# STAGE_NAME = "Model Optimization Optuna"
# try:
#     log.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#     optimizer = ModelOptimizerOptuna()
#     optimizer.train()
#     log.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         log.exception(f"Exception occurred during {STAGE_NAME}")
#         raise CustomException(e, sys)











