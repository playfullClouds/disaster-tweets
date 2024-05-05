import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,format='[%(asctime)s]: $(message)s:')


list_of_files = [
    "src/__init__.py",
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_cleaner.py",
    "src/components/data_validation.py",
    "src/components/data_transformation.py",
    "src/components/model_trainer.py",
    "src/components/model_optimizer.py",
    "src/utils.py",
    "src/logging.py",
    "src/exception.py",
    "src/pipeline/__init__.py",
    "src/pipeline/predict_pipeline.py",
    "templates/index.html",
    "templates/home.html",
    "static/css/style.css",
    "assets/n.jpg",
    "logger.py",
    "app.py",
    "main.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    "research/shorts.ipynb"
    
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Created directory: {filedir} for the file: {filename}")
        
    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Created empty file: {filepath}")
        
        
    else:
        logging.info(f"File: {filepath} already exists")