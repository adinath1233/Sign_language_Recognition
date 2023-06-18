from tkinter import *
from tkinter import filedialog
import numpy as np
from keras.models import load_model
from tkinter import messagebox
import os
import torch
from torchvision.transforms import ToTensor
from PIL import Image
from Convent_model import Convnet  

window=Tk()
window.geometry("400x150")
window.title("Sign language")

def file_open():
    global filepath
    file= filedialog.askopenfile(mode='r', filetypes=[('PhotoImage','*png')])
    if file:
        filepath=os.path.abspath(file.name)
        



def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')  
    image = image.resize((28, 28))  
    tensor = ToTensor()(image).unsqueeze(0)  
    return tensor

def predict(image_path):
    model = Convnet()
    model.load_state_dict(torch.load('sign_model.pt'))
    model.eval()   
    tensor = preprocess_image(image_path)
    outputs = model(tensor)
    _, predicted = torch.max(outputs.data, 1)
    return predicted.item()


def result():
    image_path = filepath  
    prediction = predict(image_path)
    if prediction==0:
        predictions="A"
    if prediction==1:
        predictions="B"
    if prediction==2:
        predictions="C"
    if prediction==3:
        predictions="D"
    if prediction==4:
        predictions="E"
    if prediction==5:
        predictions="F"
    if prediction==6:
        predictions="G"
    if prediction==7:
        predictions="H"
    if prediction==8:
        predictions="I"
    if prediction==9:
        predictions="J"
    if prediction==10:
        predictions="K"
    if prediction==11:
        predictions="L"
    if prediction==12:
        predictions="M"
    if prediction==13:
        predictions="N"
    if prediction==14:
        predictions="O"
    if prediction==15:
        predictions="P"
    if prediction==16:
        predictions="Q"
    if prediction==17:
        predictions="R"
    if prediction==18:
        predictions="S"
    if prediction==19:
        predictions="T"
    if prediction==20:
        predictions="U"
    if prediction==21:
        predictions="V"
    if prediction==22:
        predictions="W"
    if prediction==23:
        predictions="X"
    if prediction==24:
        predictions="Y"
    if prediction==25:
        predictions='Z'
    lab2=Label(window, text='Prediciton is: ',width=10)
    lab2.place(x=10,y=50)
    res_label= Label(window, text=predictions, width=3)
    res_label.place(x=80,y=50)


lab=Label(window, text='Select image for prediciton: ',width=20)
lab.place(x=5,y=10)

select_button=Button(window, text='select', command=file_open,width=10)
select_button.place(x=180,y=5)

predict_button= Button(window, text='Predict',command=result,width=10)
predict_button.place(x=280,y=5)

window.mainloop()