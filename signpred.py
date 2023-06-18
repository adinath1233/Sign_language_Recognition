import cv2
import mediapipe as mp
import torch
from Convent_model import Convnet  
from torchvision.transforms import ToTensor

# Load the pre-trained model and set it to evaluation mode
model = Convnet()
model.load_state_dict(torch.load('sign_model.pt'))
model.eval()

hand_signs = {
    '0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G', '7': 'H', '8': 'I',
    '10': 'K', '11': 'L', '12': 'M', '13': 'N', '14': 'O', '15': 'P', '16': 'Q', '17': 'R',
    '18': 'S', '19': 'T', '20': 'U', '21': 'V', '22': 'W', '23': 'X', '24': 'Y', '25': 'Z'
}


def predict_alphabet(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28, 28))
    image = ToTensor()(image).unsqueeze(0)
    prediction = model(image)
    predicted_label = torch.argmax(prediction).item()
    alphabet = hand_signs[str(predicted_label)]
    return alphabet

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('Frame', frame)

    # Call the function to predict the alphabet
    alphabet = predict_alphabet(frame)

    # Display the predicted alphabet on the frame
    cv2.putText(frame, alphabet, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
