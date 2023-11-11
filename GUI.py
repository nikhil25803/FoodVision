import os
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from utils import helpers

class_names = [
    "chicken_curry",
    "chicken_wings",
    "fried_rice",
    "grilled_salmon",
    "hamburger",
    "ice_cream",
    "pizza",
    "ramen",
    "steak",
    "sushi",
]


def predict_image(model_number, image_path):
    pclass = helpers.pred_and_plot(
        model_number=model_number, filename=image_path, class_names=class_names
    )
    return pclass


def browse_image():
    file_path = filedialog.askopenfilename()
    entry_image_path.delete(0, tk.END)
    entry_image_path.insert(0, file_path)
    display_image(file_path)


def display_image(image_path):
    if os.path.exists(image_path):
        image = Image.open(image_path)
        image = image.resize((200, 200), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)

        label_image.configure(image=photo)
        label_image.image = photo


def predict():
    model_number = entry_model.get()
    image_path = entry_image_path.get()

    if model_number not in ["1", "2", "3"]:
        result_var.set("Invalid model number. Please enter 1, 2, or 3.")
        return

    if not os.path.exists(image_path):
        result_var.set("Invalid image path. Please select a valid image.")
        return

    predicted_class = predict_image(model_number, image_path)
    result_var.set(f"Predicted Class: {predicted_class}")


# Create the main window
window = tk.Tk()
window.title("Food Image Prediction")
window.geometry("500x400")

# Create and place widgets
label_model = ttk.Label(window, text="Model Number:")
label_model.grid(row=0, column=0, padx=10, pady=10, sticky="e")

entry_model = ttk.Entry(window)
entry_model.grid(row=0, column=1, padx=10, pady=10, sticky="w")

label_image_path = ttk.Label(window, text="Image Path:")
label_image_path.grid(row=1, column=0, padx=10, pady=10, sticky="e")

entry_image_path = ttk.Entry(window)
entry_image_path.grid(row=1, column=1, padx=10, pady=10, sticky="w")

button_browse = ttk.Button(window, text="Browse", command=browse_image)
button_browse.grid(row=1, column=2, padx=10, pady=10)

label_image = ttk.Label(window)
label_image.grid(row=2, column=0, columnspan=3, pady=10)

button_predict = ttk.Button(window, text="Predict", command=predict)
button_predict.grid(row=3, column=1, padx=10, pady=10)

result_var = tk.StringVar()
label_result = ttk.Label(window, textvariable=result_var, font=("Arial", 12, "bold"))
label_result.grid(row=4, column=0, columnspan=3, pady=10)

# Start the Tkinter event loop
window.mainloop()
