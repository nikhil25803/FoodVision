import matplotlib.pyplot as plt
import matplotlib.image as mpig
import random
import os
import tensorflow as tf


def view_random_image(target_dir: str, target_class: str):
    # Setup the target directory
    target_folder = target_dir + target_class
    random_image = random.sample(os.listdir(target_folder), 1)

    # Read into image and plot it using matplotlib
    img = mpig.imread(target_folder + "/" + random_image[0])
    plt.imshow(img)
    plt.title(target_class)
    plt.axis("off")

    print(f"Image shape: {img.shape}")

    return img


def plot_loss_curves(history, save_path=None):
    """
    Returns separate loss curves for training and saves the plot if save_path is provided.
    """
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]

    epochs = range(len(history.history["loss"]))

    # Loss
    plt.plot(epochs, loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validity Loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    if save_path:
        plt.savefig(save_path + "/loss_plot.png")

    # Accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label="Train Accuracy")
    plt.plot(epochs, val_accuracy, label="Validity Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    if save_path:
        plt.savefig(save_path + "/accuracy_plot.png")


def load_and_prep_image(filename, img_shape=224):
    """
    Read an image from the filename and turn it into (img_shape, img_shape, colour_channel) tensor
    """
    # Read in the image
    img = tf.io.read_file(filename)

    #     # Decode the read file into a tensor
    img = tf.image.decode_image(img)

    # Resize the image
    img = tf.image.resize(img, size=[img_shape, img_shape])

    # Rescale the image and get all values between 0 and 1
    img = img / 255.0
    return img


def pred_and_plot(model_number, filename, class_names):
    # Load the model
    model_path = f"models/model_{model_number}/model_{model_number}.h5"
    model = tf.keras.models.load_model(model_path)

    # Import the target image and preprocess it
    img = load_and_prep_image(filename)

    # Make a prediction
    pred = model.predict(tf.expand_dims(img, axis=0))

    # Get the predicted class index
    pred_class_index = tf.argmax(pred, axis=1)

    # Get the predicted class
    pred_class = class_names[pred_class_index.numpy()[0]]
    print(f"Predicted class: {pred_class}")

    # Plot the image
    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False)
    plt.show()
