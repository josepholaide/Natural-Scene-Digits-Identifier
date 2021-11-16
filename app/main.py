# Import Needed Libraries
import uvicorn
from PIL import Image
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from data.predictions_handler import get_predictions
from loguru import logger
from models.model import get_newcnn_model
from models.model import get_new_model


# Initiate app instance
app = FastAPI(title='Forged Or Not Forged', version='1.0',
              description='A Convolution Neural Network that classifies numbers that appear in the context of natural scene images.')



# Define the Response
class Prediction(BaseModel):
  filename: str
  contenttype: str
  prediction: List[int] = []
  likely_class: int    


# Initiate logging
log_format = "{time} | {level} | {message} | {file} | {line} | {function} | {exception}"
logger.add(sink='app/data/log_files/logs.log', format=log_format, level='DEBUG', compression='zip')


# Api root or home endpoint
@app.get('/')
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
def get_prediction(incoming_data: Features):
    """
    This endpoint serves the predictions based on the values received from a user and the saved model.
    :param incoming_data: JSON with keys representing features and values representing the associated values.
    :return: Dict with keys 'predicted_class' (class predicted) and 'predicted_prob' (probability of prediction)
    """
    # Read image contents
    contents = await file.read()
    pil_image = Image.open(io.BytesIO(contents))
    filename = pil_image.filename
    contenttype = pil_image.format

    # convert to array
    img_array = tf.keras.preprocessing.image.img_to_array(pil_image)

    # Make predictions based on the incoming data and saved neural net
    prediction = get_predictions(img_array)
    logger.debug('Predictions successfully generated for the user')

    likely_class = np.argmax(predictions)
    # Return the predicted class and the predicted probability
    return {
      'filename': filename,
      'contenttype': contenttype,
      'prediction': prediction,
      'likely_class': likely_class
    }


if __name__ == "__main__":
    # Run app with uvicorn with port and host specified. Host needed for docker port mapping
    uvicorn.run(app, port=8000, host="0.0.0.0")