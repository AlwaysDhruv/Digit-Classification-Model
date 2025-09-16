# Simple CNN for Digit Recognition

This project is a simple implementation of a Convolutional Neural Network (CNN) for digit recognition, built from scratch using NumPy.

## Features

- Train the CNN on your own dataset of images.
- Predict digits from images using a trained model.
- Model weights are saved after training and can be reused for prediction.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd <project-directory>
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the application, execute the `main.py` script:

```bash
python main.py
```

You will be presented with a menu with the following options:

1.  **Train Model**: This option allows you to train the CNN on a set of images.
    - You will be prompted to select one or more image files.
    - For each selected image, you will need to enter the corresponding digit (0-9) as its label.
    - After providing the labels, you will be asked to specify the number of training epochs.
    - The trained model will be saved as `model.npz`.

2.  **Predict Image**: This option allows you to predict the digit from an image using the trained model.
    - The application will first load the `model.npz` file. If the model is not found, you will be prompted to train a new one.
    - You will be prompted to select a single image file for prediction.
    - The predicted digit will be displayed in the console.

3.  **Exit**: This option closes the application.

## How to Prepare Your Data

- The model is designed to work with images of handwritten digits (0-9).
- The images can be of any size or format supported by the Pillow library (e.g., PNG, JPEG). They will be automatically converted to grayscale and resized to 28x28 pixels.
- For best results, use images with a single digit and a clear background.
