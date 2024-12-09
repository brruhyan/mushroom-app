from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from yolov11 import YOLO  

app = FastAPI()

model = YOLO("path_to_yolov11_model.pt")

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    image = await file.read()
    prediction = model.predict(image)  # Process prediction here
    return JSONResponse({"prediction": prediction})
