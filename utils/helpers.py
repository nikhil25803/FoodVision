import matplotlib.pyplot as plt
import matplotlib.image as mpig
import random
import os


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