# Sign Language Prediction System

## Project Overview
This project is designed to predict alphabets of sign language based on input images or single alphabet hand signs captured from a camera. It utilizes machine learning techniques to recognize and map the hand gestures to their respective alphabets. 

The primary objective of this project is to aid in communication for individuals who use sign language, serving as a prototype for academic purposes.

## System Requirements
- **Anaconda Environment** with Python 3.10 version
- **Required Libraries**: All dependencies are specified in the `requirements.txt` file.

## Installation Instructions

1. **Create a Conda environment** with Python 3.10:
    ```bash
    conda create --name sign-language-prediction python=3.10
    ```

2. **Activate the Conda environment**:
    ```bash
    conda activate sign-language-prediction
    ```

3. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the prediction script**:
    ```bash
    python pred.py
    ```

    The script will start the system, and you can use a camera to input hand sign images for prediction.

## Dataset Information
The dataset used for this project is available on **Kaggle**. It contains labeled images of hand signs corresponding to various alphabets, enabling the system to recognize and predict the correct sign.

- **Dataset source**: The dataset is available on Kaggle (no direct link provided).

## Credits

- **Dataset**: The dataset for training the model is sourced from Kaggle.
- This project was developed for academic purposes to demonstrate sign language recognition using machine learning techniques.

## License

This project is intended for academic purposes only. It is not permitted to use this project for commercial purposes without proper authorization.
