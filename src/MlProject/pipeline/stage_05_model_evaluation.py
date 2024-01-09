from MlProject.config.configuration import ConfigurationManager
from MlProject.components.model_evaluation import ModelEvaluation
from MlProject import logger

STAGE_NAME = "Model Evaluation Stage"

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.save_results()

if __name__ == '__main__':
    try:
        logger.info(f'>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<')
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
        logger.info(f'>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<\n\nx====================x')
    except Exception as e:
        logger.exception(e)
        raise e