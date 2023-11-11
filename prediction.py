import argparse
import os
import random
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


def get_random_image(food):
    folder_path = f"data/10_food_classes_all_data/test/{food}"
    files = os.listdir(folder_path)
    random_image = random.choice(files)
    return os.path.join(folder_path, random_image)


def main(model, food):
    filename = get_random_image(food)
    helpers.pred_and_plot(
        model_number=model,
        filename=filename,
        class_names=class_names,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict and plot food images.")
    parser.add_argument("--model", type=str, help="Model name or number")
    parser.add_argument("--food", type=str, help="Food class name")

    args = parser.parse_args()

    if args.food not in class_names:
        print(f"Invalid.\nAvailable class names are:\n {class_names}")

    if args.model not in ["1", "2", "3"]:
        print(f"Invalid.\nAvailable class names are:\n {['1', '2', '3']}")

    if args.model and args.food:
        main(model=args.model, food=args.food)
    else:
        print("Please provide both --model and --food arguments.")
