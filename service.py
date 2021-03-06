from fastapi import FastAPI, UploadFile, File
from fastapi.responses import RedirectResponse
from torchvision import transforms
from PIL import Image
import numpy as np
import onnxruntime
import io

session = onnxruntime.InferenceSession('best_model.onnx')
app = FastAPI()

inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

@app.get('/')
def docs_redirect():
    return RedirectResponse(url='/docs')

@app.post('/image')
async def classify_image(file: UploadFile = File(...)):
    image = await file.read()
    image = Image.open(io.BytesIO(image)).convert('RGB')
    image = inference_transforms(image)
    image = image.unsqueeze(0)
    inputs = {session.get_inputs()[0].name: to_numpy(image)}
    prediction = session.run(None, inputs)
    return {'image': file.filename, 'class': str(np.argmax(prediction))}
