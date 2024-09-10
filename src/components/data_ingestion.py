import os, sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngesionConfig:
    raw_data_path:str = os.path.join('artifacts',"raw_data.csv")
    train_data_path:str = os.path.join('artifacts',"train_data.csv")
    test_data_path:str = os.path.join('artifacts',"test_data.csv")

class Data_Ingestion:
    def __init__(self):
        self.data_ingestion_config = DataIngesionConfig()

    def initalize_data_ingestion(self):
        logging.info('Starting Data Ingestion Process')
        try:
            logging.info('Reading data from source')
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info('data converted to dataframe')

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path,header=True)
            logging.info('train test split started')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.data_ingestion_config.train_data_path,index = False,header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path,index =False, header=True)

            logging.info('Data Ingestion Process Completed')
            
            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
                )

        except Exception as e:
            raise CustomException(e,sys)


if __name__ == '__main__':
    obj = Data_Ingestion()
    train_data_path, test_data_path = obj.initalize_data_ingestion()

    data_transformer = DataTransformation()
    train_arr, test_arr, _ = data_transformer.initiate_data_transformation(train_data_path, test_data_path)
    
    modeltrainer = ModelTrainer()
    scores = modeltrainer.initialize_model_trainer(train_array= train_arr,
                                          test_array= test_arr)
    print(scores)