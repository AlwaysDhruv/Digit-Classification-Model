import os
import numpy as np
from PIL import Image

class CNN:
    def __init__(self, num_classes=10):
        self.kernel = np.array([[1, 0, -1],
                               [1, 0, -1],
                               [1, 0, -1]])
        self.num_classes = num_classes
        # These will be initialized when training starts or a model is loaded.
        self.fc_weights = None
        self.fc_bias = None

    def _process_image(self, input_image_path):
        image = Image.open(input_image_path)
        image = image.convert('L')
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        pixel_array = np.array(image)
        pixel_array = np.clip(pixel_array, 0, 255)
        processed_image = Image.fromarray(pixel_array.astype('uint8'))
        return processed_image

    def _convolution(self, pixels, kernel):
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

    def _max_pooling(self, input_matrix, kernel_size=2, stride=2):
        input_height, input_width = input_matrix.shape
        output_height = (input_height - kernel_size) // stride + 1
        output_width = (input_width - kernel_size) // stride + 1
        output_matrix = np.zeros((output_height, output_width))
        for i in range(0, output_height):
            for j in range(0, output_width):
                region = input_matrix[i * stride:i * stride + kernel_size, j * stride:j * stride + kernel_size]
                output_matrix[i, j] = np.max(region)
        return output_matrix.flatten()

    def _fully_connected(self, input_vector, weights, bias):
        return np.dot(weights, input_vector) + bias

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def _backpropagate(self, pooled_output, predictions, label, learning_rate=0.01):
        d_loss_d_pred = predictions
        d_loss_d_pred[label] -= 1
        d_loss_d_fc_weights = np.outer(d_loss_d_pred, pooled_output)
        d_loss_d_fc_bias = d_loss_d_pred
        self.fc_weights -= learning_rate * d_loss_d_fc_weights
        self.fc_bias -= learning_rate * d_loss_d_fc_bias

    def forward_pass(self, image_path):
        image = self._process_image(image_path)
        pixels = np.array(image)
        conv_output = self._convolution(pixels, self.kernel)
        pooled_output = self._max_pooling(conv_output)
        return pooled_output

    def train(self, image_paths, labels, epochs=10):
        # Initialize weights if they don't exist
        if self.fc_weights is None:
            # Get the size of the pooled output to determine the weights size
            sample_pooled_output = self.forward_pass(image_paths[0])
            self.fc_weights = np.random.randn(self.num_classes, sample_pooled_output.size)
            self.fc_bias = np.random.randn(self.num_classes)

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            for image_path, label in zip(image_paths, labels):
                pooled_output = self.forward_pass(image_path)
                fc_output = self._fully_connected(pooled_output, self.fc_weights, self.fc_bias)
                predictions = self._softmax(fc_output)
                self._backpropagate(pooled_output, predictions, label)
        print("Training complete.")

    def predict(self, image_path):
        if self.fc_weights is None or self.fc_bias is None:
            print("Model is not trained or loaded.")
            return None

        pooled_output = self.forward_pass(image_path)
        fc_output = self._fully_connected(pooled_output, self.fc_weights, self.fc_bias)
        predictions = self._softmax(fc_output)
        return np.argmax(predictions)

    def save_model(self, path="model.npz"):
        if self.fc_weights is not None and self.fc_bias is not None:
            np.savez(path, weights=self.fc_weights, bias=self.fc_bias)
            print(f"Model saved to {path}")
        else:
            print("Cannot save a model that has not been trained.")

    def load_model(self, path="model.npz"):
        if os.path.exists(path):
            data = np.load(path)
            self.fc_weights = data['weights']
            self.fc_bias = data['bias']
            print(f"Model loaded from {path}")
        else:
            print(f"No model found at {path}. Please train a new model.")
            # Initialize weights so that training can start
            self.fc_weights = None
            self.fc_bias = None
