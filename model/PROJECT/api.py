from fastai.vision.all import *
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Load the trained FastAI model
model_path = "/Users/arjunbhndary/PROJECT/mixup_best_model.pkl"  # Adjust the path if needed
learner = load_learner(model_path)

# Define inference function
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and preprocess image
        image = await file.read()
        image = Image.open(io.BytesIO(image)).convert("RGB")
        
        # Convert image into FastAI format
        pred, _, probs = learner.predict(image)

        return {
            "prediction": str(pred),
            "confidence": float(probs.max())
        }

    except Exception as e:
        return {"error": str(e)}

# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
