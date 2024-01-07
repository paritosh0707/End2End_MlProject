import os
from MlProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from MlProject.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config


    
    ## Note: Can different data transformation techniques such as Scaler, PCA and all
    ## Can perform all kinds of EDA in ML cycle here before passing this data to the model

    # I am only adding train_test_spliting bcz this data is already cleaned up


    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_path)

        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logger.info('Splitted data into train and test sets')
        logger.info(f'train data shape -> {train.shape}')
        logger.info(f'test data shape -> {test.shape}')