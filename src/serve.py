"""Model prediction server.

$ fastapi run serve.py           # starts on default port 5000
"""

from fastapi import FastAPI, UploadFile, File
import torch

from model import cnn_model, predict_label_for_single_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

model = cnn_model().to(device)
model.load_state_dict(torch.load('model.pth', weights_only=True))
app = FastAPI()

@app.get('/health')
def health():
    return 'working'

@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> str:
    image_bytes = await file.read()
    return predict_label_for_single_image(model, image_bytes, device)
