import os
import tkinter as tk
from tkinter import filedialog, messagebox
from src.cnn import CNN
from PIL import UnidentifiedImageError


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def train_model(model):
    root = tk.Tk()
    root.withdraw()

    image_paths = filedialog.askopenfilenames(
        title="Select image files for training",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
    )
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

    try:
        epochs = int(input("Enter number of training epochs: "))
    except ValueError:
        print("Invalid number of epochs. Training cancelled.")
        return

    print("\nTraining model...")
    model.train(image_paths, labels, epochs)
    model.save_model()


def predict_image(model):
    root = tk.Tk()
    root.withdraw()

    image_path = filedialog.askopenfilename(
        title="Select an image file for prediction",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
    )
    if not image_path:
        print("No file selected. Prediction cancelled.")
        return

    try:
        model.load_model()
        print(f"\nModel successfully loaded. Predicting on image: {os.path.basename(image_path)}")

        prediction = model.predict(image_path)
        if prediction is not None:
            print(f"The predicted label for the image is: {prediction}")
        else:
            print("Prediction failed. Model might not be trained or loaded properly.")

    except UnidentifiedImageError:
        print(f"Error: The selected file '{os.path.basename(image_path)}' is not a valid image.")
    except FileNotFoundError:
        print("Error: Model file not found. Train a model first.")
    except Exception as e:
        print(f"Unexpected error during prediction: {e}")


def main():
    cnn_model = CNN()

    while True:
        clear_screen()
        print("==== Digit Classification ====")
        print("1. Train Model")
        print("2. Predict Image")
        print("3. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            train_model(cnn_model)
        elif choice == '2':
            predict_image(cnn_model)
        elif choice == '3':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please try again.")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
