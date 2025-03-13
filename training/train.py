import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import matplotlib.pyplot as plt

# Set the paths for your dataset
cat_dir = 'cat/'
non_cat_dir = 'non-cat/'

img_size = 120  # Resize all images to 120x120
batch_size = 32

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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Check the shapes of the data
print(f'Training images shape: {X_train.shape}')
print(f'Testing images shape: {X_test.shape}')

train_datagen = ImageDataGenerator(rescale=1./255, 
                                   rotation_range=40, 
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2, 
                                   zoom_range=0.2,
                                   horizontal_flip=True, 
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

# Create training and testing generators
train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
test_generator = test_datagen.flow(X_test, y_test, batch_size=batch_size)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # Additional Convolutional Layers
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # More Convolutional Layers
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Flatten the output from the convolutional layers
    tf.keras.layers.Flatten(),
    
    # Fully connected layers
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')  
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary to see the details of the layers
model.summary()

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // batch_size,
    epochs=50,
    validation_data=test_generator,
    validation_steps=len(X_test) // batch_size
)


# Save the model
model.save('cat_detector.h5')

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()
