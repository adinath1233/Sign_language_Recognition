import torch
from torchvision.transforms import ToTensor
from PIL import Image
from Convent_model import Convnet  


model = Convnet()
model.load_state_dict(torch.load('sign_model.pt'))
model.eval()


def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')  
    image = image.resize((28, 28))  
    tensor = ToTensor()(image).unsqueeze(0)  
    return tensor

def predict(image_path):
   
    tensor = preprocess_image(image_path)
    outputs = model(tensor)
    _, predicted = torch.max(outputs.data, 1)
    return predicted.item()



image_path = 'img/D.png'  
prediction = predict(image_path)

hand_signs = {
    '0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G', '7': 'H', '8': 'I',
    '10': 'K', '11': 'L', '12': 'M', '13': 'N', '14': 'O', '15': 'P', '16': 'Q', '17': 'R',
    '18': 'S', '19': 'T', '20': 'U', '21': 'V', '22': 'W', '23': 'X', '24': 'Y', '25': 'Z'
}

alphabet=hand_signs[str(prediction)]
print(f'Prediction is {alphabet}')