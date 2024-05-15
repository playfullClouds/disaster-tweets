import os
import sys
import pandas as pd
import boto3
from io import BytesIO
from src.logger import log
from transformers import BertForSequenceClassification, BertTokenizer
from src.exception import CustomException

class PredictPipeline:
    def __init__(self):
        self.bucket_name = 'disaster-tweet'
        self.model_key = 'plain-BERT/modelsafetensors'
        self.s3_client = boto3.client('s3')
        
    def predict(self, text_data):
        log.info("Loading model from S3.")
        
        try:
            # Download the model from S3
            model_obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=self.model_key)
            
            # Load the trained model
            model = BertForSequenceClassification.from_pretrained(BytesIO(model_obj['Body'].read()))
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            
            # Ensure text_data is in the expected format (e.g., a string)
            if isinstance(text_data, str):
                text_data = [text_data]
                
            log.info("Tokenizing text data.")
            
            # Tokenize the input text data
            inputs = tokenizer(text_data, padding=True, truncation=True, max_length=512, return_tensors="pt")
            
            log.info("Making predictions.")
            
            # Forward pass, get logits and take argmax
            outputs = model(**inputs)
            logits = outputs.logits
            preds = logits.argmax(dim=1)
            
            log.info("Predictions successfully made.")
            return preds.tolist()
            
        except Exception as e:
            log.error("Error in loading model from S3")
            raise CustomException(e, sys)
