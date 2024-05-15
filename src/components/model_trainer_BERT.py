import os
import sys
import torch
import pandas as pd



from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup


from src.logger import log
from dataclasses import dataclass
from src.exception import CustomException


@dataclass
class ModelTrainerConfig:
    base_dir: str = os.path.join('artifacts', 'data_transformation')
    destination_dir: str = os.path.join('artifacts', 'model_trainer')
    train_data: str = os.path.join(base_dir, 'train_BERT.csv')  # Use BERT transformed train data
    test_data: str = os.path.join(base_dir, 'test_BERT.csv')  # Use BERT transformed test data
    val_data: str = os.path.join(base_dir, 'val_BERT.csv') 
    best_model_path: str = os.path.join(destination_dir, 'best_model.pkl')
    
    
    
class DisasterDataset(Dataset):
    """Custom Dataset class for loading BERT transformed data."""

    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_ids = eval(self.data['input_ids'].iloc[index])
        attention_mask = eval(self.data['attention_mask'].iloc[index])
        target = self.data['target'].iloc[index]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'targets': torch.tensor(target, dtype=torch.long)
        }
        
        
        
class ModelTrainer:
    def __init__(self) -> None:
        """Initialize ModelTrainer and setup configurations."""
        log.info("Initializing ModelTrainer")
        self.config = ModelTrainerConfig()

        try:
            os.makedirs(self.config.destination_dir, exist_ok=True)
        except Exception as e:
            log.error(f"Error setting up the model trainer")
            raise CustomException(e, sys)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
        self.model.to(self.device)

    def load_data(self, path: str):
        """Load the dataset from the specified path"""
        log.info(f"Loading data from {path}")

        try:
            df = pd.read_csv(path)
            log.info(f"Data loaded successfully from {path}")
            return df
        except Exception as e:
            log.error(f"Failed to load data from {path}")
            raise CustomException(e, sys)

    def train_model(self):
        """Train the model using BERT-transformed data."""
        log.info("Model training starting...")

        try:
            # Load training, validation, and test data
            train_df = self.load_data(self.config.train_data)
            test_df = self.load_data(self.config.test_data)
            val_df = self.load_data(self.config.val_data)

            # Create DataLoader for each dataset
            train_data = DisasterDataset(train_df)
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=32)

            val_data = DisasterDataset(val_df)
            val_sampler = SequentialSampler(val_data)
            val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=32)

            test_data = DisasterDataset(test_df)
            test_sampler = SequentialSampler(test_data)
            test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=32)

            # Define optimizer and scheduler
            optimizer = AdamW(self.model.parameters(), lr=2e-5, correct_bias=False)
            total_steps = len(train_dataloader) * 2
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

            # Train the model
            for epoch in range(2):
                self.model.train()
                for batch in tqdm(train_dataloader, desc="Training"):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    targets = batch['targets'].to(self.device)

                    self.model.zero_grad()
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    logits = outputs.last_hidden_state[:, 0, :]
                    loss = torch.nn.functional.cross_entropy(logits, targets)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                # Evaluate the model on the validation set
                val_metrics = self.evaluate(val_dataloader)

                log.info(f'Epoch: {epoch + 1}')
                log.info(f'Validation Metrics: {val_metrics}')

            # After training, evaluate the model on the test set
            test_metrics = self.evaluate(test_dataloader)
            log.info(f'Test Metrics: {test_metrics}')

            # Save the trained model
            self.model.save_pretrained(self.config.destination_dir)
            log.info(f"Model saved to {self.config.destination_dir}")

            log.info("Model training completed successfully")
        except Exception as e:
            log.error(f"Error during model training")
            raise CustomException(e, sys)

    def evaluate(self, dataloader):
        """Evaluate the model on the given dataloader."""
        total_loss = 0
        total_accuracy = 0
        total_len = 0
        total_preds = []
        total_targets = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                targets = batch['targets'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.last_hidden_state[:, 0, :]
                loss = torch.nn.functional.cross_entropy(logits, targets)

                total_loss += loss.item() * len(targets)
                total_accuracy += (logits.argmax(1) == targets).sum().item()
                total_len += len(targets)
                total_preds.extend(logits.argmax(1).cpu().detach().numpy())
                total_targets.extend(targets.cpu().detach().numpy())

        accuracy = total_accuracy / total_len
        loss = total_loss / total_len
        precision = precision_score(total_targets, total_preds, average='weighted')
        recall = recall_score(total_targets, total_preds, average='weighted')
        f1 = f1_score(total_targets, total_preds, average='weighted')

        return {'accuracy': accuracy, 'loss': loss, 'precision': precision, 'recall': recall, 'f1': f1}
