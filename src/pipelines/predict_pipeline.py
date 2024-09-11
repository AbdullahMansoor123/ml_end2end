import os
import sys
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from src.components.data_transformation import DataTransformation

class PredictionPipeline():
    def __init__(self) -> None:
        pass
    def predict(self,features):
        try:
            root_path  = os.getcwd().split('src')[0] # get out of src.pipeline folder
            model_path = os.path.join(root_path,"artifacts/model.pkl")
            preprocessing_path = os.path.join(root_path,"artifacts/preprocessor.pkl")
            model = load_object(model_path)
            preprocessor = load_object(preprocessing_path)           
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return np.rint(preds).astype(int)
        
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(  self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int
        ):
        
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    #saving the app data in the form of dataframe which will send model as in input for prediction
    def get_data_as_dataframe(self):
        try:
            app_data_dict = {
                'gender':[self.gender],
                'race_ethnicity':[self.race_ethnicity],
                'parental_level_of_education':[self.parental_level_of_education],
                'lunch':self.lunch,
                'test_preparation_course':[self.test_preparation_course],
                'reading_score':[self.reading_score],
                'writing_score':[self.writing_score],
            }
            return pd.DataFrame(app_data_dict)
        
        except Exception as e:
            raise CustomException(e,sys)