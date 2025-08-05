import os
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
import keras_cv
import matplotlib.pyplot as plt

# Load CSV, skip bad lines
df = pd.read_csv('Data/styles.csv', on_bad_lines='skip')

# Ensure id is string and build file paths (add extension if missing)
def get_filepath(x):
    if not x.lower().endswith(('.jpg', '.jpeg', '.png')):
        x = f"{x}.jpg"
    return os.path.join('Data/images', x)

df['id'] = df['id'].astype(str)
df['filename'] = df['id'].apply(get_filepath)

# Filter out missing files
print(f"Rows before filtering: {len(df)}")
df = df[df['filename'].apply(os.path.exists)].reset_index(drop=True)
print(f"Rows after filtering files that exist: {len(df)}")

# Encode classes explicitly
class_names = sorted(df['subCategory'].unique())
class_to_index = {name: idx for idx, name in enumerate(class_names)}
df['label'] = df['subCategory'].map(class_to_index)

# Split dataset
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=None)

IMG_SIZE = (150, 150)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
NUM_CLASSES = len(class_names)

# Data augmentation pipeline (using keras_cv)
augmentation = keras.Sequential([
    keras_cv.layers.RandomFlip(),
    keras_cv.layers.RandomRotation(factor=0.15),
    keras_cv.layers.RandomZoom(height_factor=0.1, width_factor=0.1),
])

# Preprocess image function
def preprocess_image(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image

# Create dataset function
def create_dataset(df, shuffle=True, augment=False):
    filenames = df['filename'].values
    labels = tf.keras.utils.to_categorical(df['label'].values, num_classes=NUM_CLASSES)
    ds = tf.data.Dataset.from_tensor_slices((filenames, labels))

    def load_and_preprocess(filename, label):
        image = preprocess_image(filename)
        if augment:
            image = augmentation(image)
        return image, label

    ds = ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)
    return ds

train_ds = create_dataset(train_df, shuffle=True, augment=True)
val_ds = create_dataset(val_df, shuffle=False, augment=False)

# Build CNN model
model = keras.Sequential([
    keras.layers.Input(shape=(*IMG_SIZE, 3)),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(NUM_CLASSES, activation='softmax'),
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[early_stop]
)

loss, acc = model.evaluate(val_ds)
print(f"Validation Accuracy: {acc:.2f}")

# Save model and label mapping
model.save('image_classifier_model.keras')
with open('class_indices.json', 'w') as f:
    json.dump(class_to_index, f)

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.title('Loss')

    plt.show()

plot_history(history)



