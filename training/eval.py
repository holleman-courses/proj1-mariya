import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Path to the model file and dataset
model_path = 'cat_detector.h5'  
cat_dir = 'cat/'         
non_cat_dir = 'non-cat/' 

# Initialize image size
img_size = 120  # Resize all images to 120x120

# Load the images and labels manually
def load_images_and_labels(cat_dir, non_cat_dir, img_size):
    images = []
    labels = []
    
    # Load cat images
    for img_name in os.listdir(cat_dir):
        img_path = os.path.join(cat_dir, img_name)
        img = image.load_img(img_path, target_size=(img_size, img_size))
        img_array = image.img_to_array(img) / 255.0  # Normalize the image
        images.append(img_array)
        labels.append(1)  # Label 1 for 'cat'
    
    # Load non-cat images
    for img_name in os.listdir(non_cat_dir):
        img_path = os.path.join(non_cat_dir, img_name)
        img = image.load_img(img_path, target_size=(img_size, img_size))
        img_array = image.img_to_array(img) / 255.0  # Normalize the image
        images.append(img_array)
        labels.append(0)  # Label 0 for 'non-cat'
    
    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels

# Load the images and labels
images, labels = load_images_and_labels(cat_dir, non_cat_dir, img_size)

# One-hot encode the labels for evaluation
labels = to_categorical(labels, num_classes=2)

# Load the trained model
model = load_model(model_path)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(images, labels, batch_size=32)

# Print the evaluation result
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images[i])
    plt.title('Predicted: ' + ('Cat' if np.argmax(model.predict(images[i:i+1])) == 1 else 'Non-Cat'))
    plt.axis('off')
plt.show()
