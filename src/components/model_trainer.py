import os
import sys
import joblib
import pandas as pd


from sklearn.svm import SVC


from src.logger import log
from dataclasses import dataclass, field
from src.exception import CustomException
