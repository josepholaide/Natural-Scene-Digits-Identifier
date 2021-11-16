import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image
from loguru import logger
from models.model import get_newcnn_model
from models.model import get_new_model

@logger.catch
def get_predictions(image):
    """
    A function that reshapes the incoming JSON data, loads the saved model objects
    and returns predicted class and probability.
    :param data: Dict with keys representing features and values representing the associated value
    :return: Dict with keys 'predicted_class' (class predicted) and 'predicted_prob' (probability of prediction)
    """
    # resize new data and reshape the data to suit the model
    image = np.asarray(image.resize((32, 32, 3)))
    image = np.expand_dims(res, axis=0)
    
    # Load saved keras model
    checkpoint_best_path = 'app/models/checkpoints_best_only/checkpoint'
    model_best_path = get_newcnn_model()
    model_best_path.load_weights(checkpoint_best_path)
    logger.debug('Saved CNN model loaded successfully')

    # Make new predictions from the newly scaled data and return this prediction
    predictions = model_best_path.predict(random_test_images)

    return predictions
