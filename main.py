import os
import tkinter as tk
from tkinter import filedialog
from src.cnn import CNN

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def train_model(model):
    root = tk.Tk()
    root.withdraw()

    image_paths = filedialog.askopenfilenames(title="Select image files for training")
    if not image_paths:
        print("No files selected. Training cancelled.")
        return

    labels = []
    for image_path in image_paths:
        while True:
            try:
                label = int(input(f"Enter label for {os.path.basename(image_path)}: "))
                labels.append(label)
                break
            except ValueError:
                print("Invalid input. Please enter an integer.")

    epochs = int(input("Enter number of training epochs: "))

    model.train(image_paths, labels, epochs)
    model.save_model()

def predict_image(model):
    root = tk.Tk()
    root.withdraw()

    image_path = filedialog.askopenfilename(title="Select an image file for prediction")
    if not image_path:
        print("No file selected. Prediction cancelled.")
        return

    model.load_model()
    prediction = model.predict(image_path)
    if prediction is not None:
        print(f"The predicted label for the image is: {prediction}")

def main():
    cnn_model = CNN()

    while True:
        clear_screen()
        print("1. Train Model")
        print("2. Predict Image")
        print("3. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            train_model(cnn_model)
        elif choice == '2':
            predict_image(cnn_model)
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please try again.")

        input("Press Enter to continue...")

if __name__ == "__main__":
    main()
