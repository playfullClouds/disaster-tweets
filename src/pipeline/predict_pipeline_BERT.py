import os
import sys
import pandas as pd


from src.logger import log
from src.utils import unzip_data
from src.exception import CustomException
from transformers import BertModel, BertTokenizer


class PredictPipeline:
    def __init__(self):
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # Paths to the saved model and vectorizer
        # self.model_path = os.path.join(BASE_DIR, 'artifacts', 'model_trainer', 'model.zip')
        self.model_path = os.path.join(BASE_DIR, 'artifacts', 'model_optimizer', 'optimized_model.pkl')
        self.preprocessor_path = os.path.join(BASE_DIR, 'artifacts', 'data_transformation', 'tfidf_vectorizer.pkl')
     