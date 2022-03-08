# 1. Library imports
import uvicorn
from fastapi import FastAPI, File, UploadFile
from image_predictor import *
from fastapi import FastAPI, Request
import matplotlib.pyplot

 
import numpy as np
import pickle
import pandas as pd

app = FastAPI() 

@app.get('/')
def index():
    return {"It is a low level API which can classify 5 classes of food: Samosa, Donuts, Fries, Burger, Noodles"}

@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    matplotlib.pyplot.imshow(image)
    prediction = predict(image)
    return prediction

 

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
