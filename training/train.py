import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import regularizers

# Set the paths for your dataset
cat_dir = 'cat/'
non_cat_dir = 'non-cat/'

img_width = 96   
img_height = 96 
batch_size = 32

# Rename pictures to catNNNN.jpg
def rename_images(directory, name):

    files = os.listdir(directory)
    
    # Filter only jpg files (or you can adjust this for other formats as needed)
    jpg_files = [f for f in files] 
    
    # Sort the files if you want them to be renamed sequentially
    jpg_files.sort()
    
    # Loop through the jpg files and rename them
    for index, file in enumerate(jpg_files):        
        # Generate the new file name with zero-padded number
        new_name = f"{name}{index:04d}.jpg"
        
        # Get the full paths for old and new names
        old_file_path = os.path.join(directory, file)
        new_file_path = os.path.join(directory, new_name)
        
        # Check if the file with the new name already exists
        if os.path.exists(new_file_path):
            print(f"Skipping {file}: {new_file_path} already exists.")
        else:
            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {old_file_path} -> {new_file_path}")

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def count_images(directory):
    image_count = 0
    for img_name in os.listdir(directory):
        image_count = image_count + 1
    return image_count    

# Load the images and labels manually
def load_images_and_labels(cat_dir, non_cat_dir, width, height):
    images = []
    labels = []
    
    # Load cat images
    for img_name in os.listdir(cat_dir):
        img_path = os.path.join(cat_dir, img_name)
        try:
            img = image.load_img(img_path, target_size=(width, height))
            img_array = image.img_to_array(img)# / 255.0  # Normalize the image
            img_array = tf.image.rgb_to_grayscale(img_array)
            images.append(img_array)
            labels.append(1)  # Label 1 for 'cat'
        except:
            print(img_path)
    # Load non-cat images
    for img_name in os.listdir(non_cat_dir):
        img_path = os.path.join(non_cat_dir, img_name)
        try:
            img = image.load_img(img_path, target_size=(width, height))
            img_array = image.img_to_array(img)# / 255.0  # Normalize the image
            img_array = tf.image.rgb_to_grayscale(img_array)
            images.append(img_array)
            labels.append(0)  # Label 0 for 'non-cat'
        except:
            print(img_path)
    
    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels

def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(8, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(img_width, img_height, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.AveragePooling2D((2,2)),
        tf.keras.layers.Dropout(0.2),
                        
        tf.keras.layers.SeparableConv2D(16,kernel_size=(3, 3), depthwise_regularizer=regularizers.l2(0.0001), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.AveragePooling2D((2,2)),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.SeparableConv2D(32, kernel_size=(3, 3), depthwise_regularizer=regularizers.l2(0.0001), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.AveragePooling2D((2,2)),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.SeparableConv2D(64, kernel_size=(3, 3), depthwise_regularizer=regularizers.l2(0.0001), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.AveragePooling2D((2,2)),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.SeparableConv2D(128, kernel_size=(3, 3), depthwise_regularizer=regularizers.l2(0.001), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.AveragePooling2D((2,2)),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.SeparableConv2D(64, kernel_size=(3, 3), depthwise_regularizer=regularizers.l2(0.001), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        # Flatten the output from the convolutional layers
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        
        # Fully connected hidden layer 
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')  
    ])
    return model 

import tensorflow as tf
print(tf.__version__)

cats = count_images(cat_dir)
non_cats = count_images(non_cat_dir)

categories = ['Cat', 'Non-Cat']
counts = [cats, non_cats]

plt.figure(figsize=(6, 4))
plt.bar(categories, counts, color=['blue', 'orange'])
plt.xlabel('Category')
plt.ylabel('Number of Images')
plt.title('Image Counts in Cat vs Non-Cat Directories')
plt.show()

#rename_images(non_cat_dir, "non-cat")
# Load the images and labels
images, labels = load_images_and_labels(cat_dir, non_cat_dir, img_width, img_height)

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

model = build_model()
print(model)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary to see the details of the layers
model.summary()

#early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# Train the model
history = model.fit(
    train_generator,
    shuffle=True,
    steps_per_epoch=len(X_train) // batch_size,
    epochs=70,
    #callbacks=[early_stopping, lr_scheduler],
    callbacks=[lr_scheduler],
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
