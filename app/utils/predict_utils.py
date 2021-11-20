import tensorflow as tf
import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image
from loguru import logger
from models.model import get_newcnn_model
from models.model import get_new_model

dirname = os.curdir
print(dirname)
@logger.catch
def get_predictions(image):
    """
    A function that reshapes the incoming JSON data, loads the saved model objects
    and returns predicted class and probability.
    :param data: Dict with keys representing features and values representing the associated value
    :return: Dict with keys 'predicted_class' (class predicted) and 'predicted_prob' (probability of prediction)
    """
    # resize new data and reshape the data to suit the model
    image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_CUBIC)
    try:
        random_test_images = image.reshape(32, 32, 3)
    except:
        random_test_images = image
    random_test_images = np.expand_dims(random_test_images, axis=0)
    
    # Load saved keras model
    checkpoint_best_path = '/c/Users/CRUISE/Reading-Digits-in-Natural-Scene-Images/app/utils/'
    # model_best_path = get_newcnn_model()
    model_best_path = tf.keras.models.load_model(checkpoint_best_path + 'saved_model.h5')
    logger.debug('Saved CNN model loaded successfully')

    # Make new predictions from the newly scaled data and return this prediction
    predictions = model_best_path.predict(random_test_images)

    return predictions
