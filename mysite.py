from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from torchvision.transforms import functional as F
from PIL import Image
import torch
import torchvision
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load the trained Faster R-CNN model

def load_model():
    # Define the model architecture
    model = torchvision.models.resnet50(pretrained=False)


    # Load the trained model weights
    model.load_state_dict(torch.load("C:\\Users\\kavya\\Downloads\\model2.pth", map_location=torch.device('cpu')))
    # Set the model to evaluation mode
    model.eval()
    return model
model = load_model() 
def preprocess_image(image):
    # Preprocess the input image
    try:
        image = Image.open(image.file).convert("RGB")
        image_tensor = F.to_tensor(image)
        return image_tensor
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

def detect_road_damage(image_tensor):
    # Perform road damage detection using the trained model
    with torch.no_grad():
        outputs = model([image_tensor])
    return outputs

@app.post("/detect-road-damage/")
async def detect_road_damage_endpoint(file: UploadFile = File(...)):
    try:
        # Preprocess the image
        image_tensor = preprocess_image(file)
        if image_tensor is None:
            return JSONResponse(content={"message": "Error preprocessing image"}, status_code=500)

        # Perform road damage detection
        outputs = detect_road_damage(image_tensor)

        # Extract bounding boxes and class labels from the outputs
        # You may need to customize this part based on the output format of your model
        detected_boxes = outputs[0]['boxes'].cpu().numpy().tolist()
        detected_labels = outputs[0]['labels'].cpu().numpy().tolist()

        # Format the detection results
        detection_results = [{"label": label, "box": box} for label, box in zip(detected_labels, detected_boxes)]

        return JSONResponse(content={"detection_results": detection_results})
    except Exception as e:
        print(f"Error detecting road damage: {str(e)}")
        return JSONResponse(content={"message": "Error detecting road damage"}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
