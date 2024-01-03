from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
from torchvision import transforms

app = FastAPI()

# Load model and processor
processor = AutoImageProcessor.from_pretrained("dima806/face_emotions_image_detection")
model = AutoModelForImageClassification.from_pretrained("dima806/face_emotions_image_detection")

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_emotion(image_path):
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predicted emotion
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    predicted_emotion = emotion_labels[predicted_class]

    return predicted_emotion

@app.post("/predict-emotion")
async def predict_emotion_api(file: UploadFile = File(...)):
    try:
        # Save the uploaded file locally
        file_path = f"temp/{file.filename}"
        with open(file_path, "wb") as image_file:
            image_file.write(file.file.read())

        # Perform emotion prediction
        predicted_emotion = predict_emotion(image_path=file_path)

        # Return the prediction as JSON
        return JSONResponse(content={"predicted_emotion": predicted_emotion})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI application with uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
