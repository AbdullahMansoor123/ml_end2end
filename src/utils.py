import os, sys
import pandas as pd
import numpy as np
import dill
import pickle

from src.exception import CustomException
from src.logger import logging


    
def save_object(file_path, data):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(data, file_obj)

    except Exception as e:
        raise CustomException(e, sys)