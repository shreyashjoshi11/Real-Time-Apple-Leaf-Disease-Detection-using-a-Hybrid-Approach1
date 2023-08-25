from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import cv2

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Model = tf.keras.models.load_model(r"C:\Users\shreyash\Desktop\Msc Project Data\Models1")
class_names = {
    0: "Apple Scab",
    1: "Apple Black Rot",
    2: "Apple Cedar Rust",
    3: "Apple Healthy",
}

@app.get("/ping")
async def ping():
    return "Hello"

def read_file(data) -> np.ndarray:
  image = np.array(Image.open(BytesIO(data)))
  return image

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    # Read the image from the uploaded file
    contents = await file.read()
    image = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Resize the image to (128, 128)
    image = cv2.resize(image, (128, 128))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)

    # Make predictions using the model
    prediction = Model.predict(image)
    predicted_label = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction, axis=1)[0]

    # Get the class name corresponding to the predicted label
    class_name = class_names.get(predicted_label, "Unknown")

    result = {
        "predicted_class": class_name,
        "confidence": float(confidence)
    }

    # Return the result as JSON response
    return result

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
