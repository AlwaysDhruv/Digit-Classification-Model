import os
import numpy as np
import tkinter as tk
from PIL import Image
from tkinter import filedialog
n = 1   
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def process_image(input_image_path):
    image = Image.open(input_image_path)
    image = image.convert('L')
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    pixel_array = np.array(image)
    pixel_array = np.clip(pixel_array, 0, 255)
    processed_image = Image.fromarray(pixel_array.astype('uint8'))
    return processed_image

def convolution(pixels, kernel):
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = pixels.shape
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1
    result = np.zeros((output_height, output_width))
    for i in range(output_height):
        for j in range(output_width):
            region = pixels[i:i + kernel_height, j:j + kernel_width]
            result[i, j] = np.sum(region * kernel)
    return result

def max_pooling(input_matrix, kernel_size=2, stride=2):
    input_height, input_width = input_matrix.shape
    output_height = (input_height - kernel_size) // stride + 1
    output_width = (input_width - kernel_size) // stride + 1
    output_matrix = np.zeros((output_height, output_width))
    for i in range(0, output_height):
        for j in range(0, output_width):
            region = input_matrix[i * stride:i * stride + kernel_size, j * stride:j * stride + kernel_size]
            output_matrix[i, j] = np.max(region)
    return output_matrix.flatten()

def fully_connected(input_vector, weights, bias):
    return np.dot(weights, input_vector) + bias

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def cross_entropy_loss(predictions, label):
    epsilon = 1e-12
    predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
    return -np.log(predictions[label])

def backpropagate(pooled_output, fc_weights, fc_bias, predictions, label, learning_rate=0.01):
    d_loss_d_pred = predictions
    d_loss_d_pred[label] -= 1
    d_loss_d_fc_weights = np.outer(d_loss_d_pred, pooled_output)
    d_loss_d_fc_bias = d_loss_d_pred
    fc_weights -= learning_rate * d_loss_d_fc_weights
    fc_bias -= learning_rate * d_loss_d_fc_bias
    return fc_weights, fc_bias
def main(path,label,train):
    kernel = np.array([[1, 0, -1],
                       [1, 0, -1],
                       [1, 0, -1]])
    num_classes = 10
    image = process_image(path)
    pixels = np.array(image)
    conv_output = convolution(pixels, kernel)
    pooled_output = max_pooling(conv_output)
    fc_weights = np.random.randn(num_classes, pooled_output.size)
    fc_bias = np.random.randn(num_classes)
    fc_output = fully_connected(pooled_output, fc_weights, fc_bias)
    predictions = softmax(fc_output)
    print("System Predicting.......")
    for epoch in range(train):
        fc_weights, fc_bias = backpropagate(pooled_output, fc_weights, fc_bias, predictions, label)
    pixels = np.array(image) / 255.0
    conv_output = convolution(pixels, kernel)
    pooled_output = max_pooling(conv_output)
    fc_output = fully_connected(pooled_output, fc_weights, fc_bias)
    predictions = softmax(fc_output)
    predicted_label = np.argmax(predictions)
    print(f"True Label: {label}, Predicted Label: {predicted_label}")
    if predicted_label==0 and label==0:
        accuracy = 100.00000
        print(f"Accuracy: {accuracy:.2f}%")
    elif predicted_label==0 or label==0:
        accuracy = 0
        print(f"Accuracy: {accuracy:.2f}%")
    elif predicted_label > label:
        accuracy = (((label / predicted_label) * 100) - 100)
        print(f"Accuracy: {accuracy:.2f}%")        
    else:
        accuracy = (label / predicted_label) * 100
        print(f"Accuracy: {accuracy:.2f}%")
while n!=0:
    clear_screen()
    root = tk.Tk()
    root.withdraw()
    path =  filedialog.askopenfilename(title="Select a file")
    label = int(input("Enter Digit Label :- "))
    train = int(input("How Many Times To Get The Output :- "))
    main(path,label,train)
    n = int(input("Enter Your Choice (0 for Exit) and (1 for Repeat) :- "))