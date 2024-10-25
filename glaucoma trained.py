import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Paths to the directories
train_dir = r'C:\Users\kukum\OneDrive\Documents\glaucoma_detection_project\dataset'
test_dir = r'C:\Users\kukum\OneDrive\Documents\glaucoma_detection_project\train'

# Image Data Generator for Data Augmentation and Rescaling
datagen = ImageDataGenerator(
    rescale=1./255,       # Rescale the pixel values from [0, 255] to [0, 1]
    shear_range=0.2,      # Randomly shear images
    zoom_range=0.2,       # Randomly zoom into images
    horizontal_flip=True  # Randomly flip images horizontally
)

# Training and Testing Generators
train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Resize images to 224x224
    batch_size=32,
    class_mode='binary'      # Binary classification (glaucoma or normal)
)

test_gen = datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Model Architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    
    Dense(128, activation='relu'),
    Dropout(0.5),
    
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the Model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Checkpoints and Early Stopping
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, mode='min')
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the Model
history = model.fit(
    train_gen,
    steps_per_epoch=train_gen.samples // train_gen.batch_size,
    validation_data=test_gen,
    validation_steps=test_gen.samples // test_gen.batch_size,
    epochs=25,  # You can increase this based on your data and computation power
    callbacks=[checkpoint, early_stop]
)

# Save the final model
model.save('glaucoma_detection_model.keras')

# Evaluate the Model on the Test Set
test_loss, test_acc = model.evaluate(test_gen)
print(f'Test Accuracy: {test_acc * 100:.2f}%')