"""Model prediction server.

$ fastapi run serve.py           # starts on port 5000
"""
from fastapi import FastAPI, UploadFile, File
import torch
from torchvision import io

from dataset import val_transform, classes

model = torch.load('model.pth')
app = FastAPI()

@app.get('/health')
def health():
    return 'working'

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image_uint8_tensor = torch.frombuffer(image_bytes, dtype=torch.uint8)
    image_tensor = torchvision.io.decode_image(image_uint8_tensor)
    transformed_tensor = val_transform(image_tensor)
    return {
        
    }
