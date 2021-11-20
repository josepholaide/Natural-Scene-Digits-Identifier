# Import Needed Libraries
import uvicorn
from PIL import Image
import io
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from utils.predict_utils import get_predictions
from loguru import logger
from models.model import get_newcnn_model
from models.model import get_new_model


# Initiate app instance
app = FastAPI(title='Natural scene images number', version='1.0',
              description='A Convolution Neural Network that classifies numbers that appear in the context of natural scene images.')



# Define the Response
class Prediction(BaseModel):
  likely_class: int    


# Initiate logging
log_format = "{time} | {level} | {message} | {file} | {line} | {function} | {exception}"
logger.add(sink='app/data/log_files/logs.log', format=log_format, level='DEBUG', compression='zip')


# Api root or home endpoint
@app.get('/home')
def read_home():
    """
    Home endpoint which can be used to test the availability of the application.
    :return: Dict with key 'message' and value 'Fake or Not API live!'
    """
    logger.debug('User checked the root page')
    return {'message': 'Fake or Not API live!'}


# Prediction endpoint
@app.post('/predict', response_model=Prediction)
@logger.catch()  # catch any unexpected breaks
async def get_prediction(file: UploadFile = File(...)):
    """
    This endpoint serves the predictions based on the values received from a user and the saved model.
    :param incoming_data: JSON with keys representing features and values representing the associated values.
    :return: Dict with keys 'predicted_class' (class predicted) and 'predicted_prob' (probability of prediction)
    """
    # Read image contents
    contents = await file.read()
    #pil_image = Image.open(io.BytesIO(contents))
    decoded = cv2.imdecode(np.frombuffer(contents, np.uint8), -1)

    # convert to array
    img_array = tf.keras.preprocessing.image.img_to_array(decoded)

    # Make predictions based on the incoming data and saved neural net
    prediction = get_predictions(img_array)
    logger.debug(f'Predictions successfully generated for the user, and image shape {decoded.shape}')

    likely_class = np.argmax(prediction)
    # Return the predicted class and the predicted probability
    return {
      'likely_class': likely_class
    }


if __name__ == "__main__":
    # Run app with uvicorn with port and host specified. Host needed for docker port mapping
    uvicorn.run(app, port=8000, host="0.0.0.0")
