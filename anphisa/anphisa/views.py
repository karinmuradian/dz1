from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import torch 
import torch.nn as nn
import onnxruntime
from torch.onnx import export
import numpy as np
from PIL import Image
import base64
from io import BytesIO



imageClassList = {'0': 'Cloud', '1': 'Lion', '2': 'Ray'}  #Сюда указать классы

def scoreImagePage(request):
    return render(request, 'anphisa/scorepage.html')

def predictImage(request):
    fileObj = request.FILES['filePath']
    fs = FileSystemStorage()
    filePathName = fs.save('images/'+fileObj.name,fileObj)
    filePathName = fs.url(filePathName)
    modelName = request.POST.get('modelName')
    scorePrediction = predictImageData(modelName, '.'+filePathName)
    with open('.'+filePathName, "rb") as image_file:
        base64img= base64.b64encode(image_file.read())
        base64img= "data:image/png;base64, %s"%base64img.decode()
    context = {'scorePrediction': scorePrediction, 'base64img': base64img}

    return render(request, 'anphisa/scorepage.html', context)

def predictImageData(modelName, filePath):
    img = Image.open(filePath).convert("RGB")
    img = np.asarray(img.resize((32, 32), Image.ANTIALIAS))
    sess = onnxruntime.InferenceSession(r'C:\Users\katko\Desktop\Django project 1\anphisa\media\models\cifar100_CNN_RESNET20.onnx') #<-Здесь требуется указать свой путь к модели
    outputOFModel = np.argmax(sess.run(None, {'input': np.asarray([img]).astype(np.float32)}))
    score = imageClassList[str(outputOFModel)]
    return score

