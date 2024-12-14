from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from io import BytesIO

# Initialize the FastAPI app
app = FastAPI()

# Load the AI model and labels
model = load_model("Model/keras_model.h5")
with open("Model/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

@app.get("/")
def root():
    return {"message": "AI Model API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded image file
        contents = await file.read()

        # Preprocess the image directly from memory
        img = image.load_img(BytesIO(contents), target_size=(224, 224))  # Adjust target size to match model input
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize to [0, 1]

        # Make prediction
        predictions = model.predict(img_array)
        predicted_label = labels[np.argmax(predictions)]
        confidence = float(np.max(predictions))

        return JSONResponse(content={
            "label": predicted_label,
            "confidence": confidence
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Uncomment the following lines to run locally
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
