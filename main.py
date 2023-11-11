import pathlib
import numpy as np
from PIL import Image
import scipy
import tensorflow as tf
from utils import helpers

# Setup train and test directory
train_dir = "data/10_food_classes_all_data/train/"
test_dir = "data/10_food_classes_all_data/test/"

data_dir = pathlib.Path(train_dir)
class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))


# Rescale
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.0)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.0)

# Load data into batches
train_data = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32, class_mode="categorical"
)

test_data = test_datagen.flow_from_directory(
    test_dir, target_size=(224, 224), batch_size=32, class_mode="categorical"
)


# Replicating a model which is at CNN explainer website
model_1 = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(10, 3, input_shape=(224, 224, 3), activation="relu"),
        tf.keras.layers.Conv2D(10, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(10, 3, activation="relu"),
        tf.keras.layers.Conv2D(10, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

# Compile the model
model_1.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"],
)

# Fit the model
history_1 = model_1.fit(
    train_data,
    epochs=5,
    steps_per_epoch=len(train_data),
    validation_data=test_data,
    validation_steps=len(test_data),
)

model_1_result = model_1.evaluate(test_data)

# Save model1 details
model_1.save("models/model_1/model_1.h5")
helpers.plot_loss_curves(history_1, "data/model_1")


model_2 = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(10, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(10, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

model_2.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"],
)

history_2 = model_2.fit(
    train_data,
    epochs=5,
    steps_per_epoch=len(train_data),
    validation_data=test_data,
    validation_steps=len(test_data),
)

model_2_result = model_1.evaluate(test_data)

# Save model1 details
model_2.save("models/model_2/model_2.h5")
helpers.plot_loss_curves(history_1, "data/model_2")

"""
As **simplifying the model** didn't work too well, and the overfitting continued. Now we should  try **data augmentation**. 
"""
# Create an augmented data generator instance

train_datagen_augmented = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1 / 255.0,
    rotation_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)


train_data_augmented = train_datagen_augmented.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32, class_mode="categorical"
)


# Let's create a model for augmented training data

model_3 = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(10, 3, input_shape=(224, 224, 3), activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(10, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

model_3.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"],
)
history_3 = model_3.fit(
    train_data_augmented,
    epochs=5,
    steps_per_epoch=len(train_data_augmented),
    validation_data=test_data,
    validation_steps=len(test_data),
)

model_3_result = model_1.evaluate(test_data)

# Save model1 details
model_2.save("models/model_3/model_3.h5")
helpers.plot_loss_curves(history_1, "data/model_3")
