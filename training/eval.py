import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to the model file and dataset
model_path = 'cat_detector.h5'  
cat_dir = 'cat/'         
non_cat_dir = 'non-cat/' 
test_cat_dir = 'test/cat/'         
test_non_cat_dir = 'test/non-cat/' 

# Initialize image size
img_size = 96  # Resize all images to 120x120


def load_test_images(image_dir):
    images = []
    # Load cat images
    for img_name in os.listdir(image_dir):
        if img_name.lower().endswith(".jpg"):
            img_path = os.path.join(image_dir, img_name)
            img = image.load_img(img_path, target_size=(img_size, img_size))
            img_array = image.img_to_array(img) / 255.0  # Normalize the image
            img_array = tf.image.rgb_to_grayscale(img_array)                        
            images.append(img_array)
    return np.array(images)

# Load the images and labels manually
def load_images_and_labels(cat_dir, non_cat_dir, img_size):
    images = []
    labels = []
    
    # Load cat images
    for img_name in os.listdir(cat_dir):
        img_path = os.path.join(cat_dir, img_name)
        img = image.load_img(img_path, target_size=(img_size, img_size))
        img_array = image.img_to_array(img) / 255.0  # Normalize the image
        img_array = tf.image.rgb_to_grayscale(img_array)                        
        images.append(img_array)
        labels.append(1)  # Label 1 for 'cat'
    
    # Load non-cat images
    for img_name in os.listdir(non_cat_dir):
        img_path = os.path.join(non_cat_dir, img_name)
        img = image.load_img(img_path, target_size=(img_size, img_size))
        img_array = image.img_to_array(img) / 255.0  # Normalize the image
        img_array = tf.image.rgb_to_grayscale(img_array)                        
        images.append(img_array)
        labels.append(0)  # Label 0 for 'non-cat'
    
    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels

# Prepare the representative dataset 
def representative_data_gen():
    images, labels = load_images_and_labels(cat_dir, non_cat_dir, img_size)
    num_calibration_steps = len(images)
    for i in range(num_calibration_steps):
        next_input = images[i:i+1,:,:,:]
        yield [next_input.astype(np.float32)]  

def convert_model_to_int8(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Set the representative dataset
    converter.representative_dataset = representative_data_gen
    
    # Set the target for integer quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # Ensure input and output tensors are also quantized
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the quantized TFLite model to a file
    with open('cat_detector_int8.tflite', "wb") as fpo:
      fpo.write(tflite_model)
    print(f"Wrote to {tflite_model}")   

def test_quantized_model():
    # Load the quantized model
    interpreter = tf.lite.Interpreter(model_path='cat_detector_int8.tflite')

    # Allocate tensors
    interpreter.allocate_tensors()

    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_scale, input_zero_point = input_details[0]["quantization"]
    
    images, labels = load_images_and_labels(test_cat_dir, test_non_cat_dir, img_size)

    TP = FP = TN = FN = 0

    for image, true_label in zip(images, labels):
        image = np.expand_dims(image, axis=0) 
        image_q = np.array(image/input_scale + input_zero_point, dtype=np.int8)

        interpreter.set_tensor(input_details[0]['index'], image_q)

        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(output_data)
        predicted_label = np.argmax(output_data)
        print(f'Predicted value: {predicted_label}, true value: {true_label}')
    
        if true_label == 1 and predicted_label == 1:
            TP += 1  # True Positive
        elif true_label == 1 and predicted_label == 0:
            FN += 1  # False Negative
        elif true_label == 0 and predicted_label == 0:
            TN += 1  # True Negative
        elif true_label == 0 and predicted_label == 1:
            FP += 1  # False Positive
    
    # Calculate the metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    frr = FN / (FN + TP) if TP + FN != 0 else 0
    fpr = FP/ (FP + TN) if FP + TN != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

    # Print the results
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'FRR: {frr:.4f}')
    print(f'FPR: {fpr:.4f}')
    
    print(f'True Positives (TP): {TP}')
    print(f'False Positives (FP): {FP}')
    print(f'True Negatives (TN): {TN}')
    print(f'False Negatives (FN): {FN}')

    plt.figure(figsize=(10, 5))
    '''
    for i in range(10):
        dat = np.expand_dims(input_data[i], axis=0)
        dat_q = np.array(dat/input_scale + input_zero_point, dtype=np.int8)
        #print(f"min = {np.min(dat_q)}, max = {np.max(dat_q)}")
        
        # Set the input tensor with the sample data
        interpreter.set_tensor(input_details[0]['index'], dat_q)

        # Invoke the interpreter (run inference)
        interpreter.invoke()

        # Get and print the output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = np.argmax(output_data)
        print("prediction =", prediction)
    plt.show()
    '''

# Load the images and labels
#images, labels = load_images_and_labels(cat_dir, non_cat_dir, img_size)

# One-hot encode the labels for evaluation
#labels = to_categorical(labels, num_classes=2)

import tensorflow as tf
print(tf.__version__)

# Load the trained model
model = tf.keras.models.load_model(model_path)
# Evaluate the model on the test set
#loss, accuracy = model.evaluate(images, labels, batch_size=64)

model.summary()

convert_model_to_int8(model)
test_quantized_model()