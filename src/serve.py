"""Model prediction server.

$ python serve.py        # starts server on port 8080
"""

from fastapi import FastAPI, UploadFile, File
import torch

from model import cnn_model, predict_label_for_single_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

model = cnn_model().to(device)
model.load_state_dict(torch.load('checkpoints/model.pth', weights_only=True))
app = FastAPI()

@app.get('/health')
def health():
    return 'working'

@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> str:
    image_bytes = await file.read()
    return predict_label_for_single_image(model, image_bytes, device)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
