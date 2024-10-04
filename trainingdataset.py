import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the path to your dataset directories
train_dir = './dataset/train'
test_dir = './dataset/test'

# Data augmentation for the training dataset
train_datagen = ImageDataGenerator(
    rescale=1./255,                # Normalize the images
    rotation_range=40,             # Randomly rotate images
    width_shift_range=0.2,         # Randomly shift images horizontally
    height_shift_range=0.2,        # Randomly shift images vertically
    shear_range=0.2,               # Randomly shear images
    zoom_range=0.2,                # Randomly zoom images
    horizontal_flip=True,          # Randomly flip images horizontally
    fill_mode='nearest'            # Fill pixels that are newly created during transformations
)

# No data augmentation for validation/test dataset
test_datagen = ImageDataGenerator(rescale=1./255)

# Create the training data generator
train_generator = train_datagen.flow_from_directory(
    train_dir,                     # Directory for the training data
    target_size=(150, 150),         # Resize images to 150x150
    batch_size=32,                 # Batch size for training
    class_mode='categorical'        # For multi-class classification
)

# Create the validation data generator
validation_generator = test_datagen.flow_from_directory(
    test_dir,                      # Directory for the validation data
    target_size=(150, 150),        # Resize images to 150x150
    batch_size=32,                 # Batch size for validation
    class_mode='categorical'        # For multi-class classification
)

# Build a simple CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 output neurons for weapon, mask, and none
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Multi-class classification
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,  # Increase or decrease based on your needs
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Save the trained model
model.save("mask_weapon_none_detection_modelwithmaazanew.h5")

print("Training completed and model saved as mask_weapon_none_detection_model.h5")

