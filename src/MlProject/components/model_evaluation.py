import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import numpy as np
import joblib
from MlProject.utils.common import save_json
from MlProject.entity.config_entity import ModelEvaluationConfig
from pathlib import Path

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual_val, predicted_val):
        rmse =np.sqrt(mean_squared_error(actual_val, predicted_val))
        mae = mean_absolute_error(actual_val, predicted_val)
        r2 = r2_score(actual_val, predicted_val)

        return {"rmse":rmse, "mae":mae, "r2":r2}

    def save_results(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        x_test =test_data.drop([self.config.target_column],axis=1)
        y_test = test_data[[self.config.target_column]]

        predicted_val = model.predict(x_test)
        scores = self.eval_metrics(y_test, predicted_val)

        save_json(path=Path(self.config.metric_file_name), data=scores)