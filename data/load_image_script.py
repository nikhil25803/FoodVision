import zipfile
import os
import shutil
import requests
from tqdm import tqdm

url = "https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_all_data.zip"

# Get the file size for tqdm
response = requests.head(url)
file_size = int(response.headers.get("content-length", 0))

# Download the file with tqdm progress bar
with requests.get(url, stream=True) as response, open(
    "10_food_classes_all_data.zip", "wb"
) as outfile, tqdm(
    desc="Downloading",
    total=file_size,
    unit="B",
    unit_scale=True,
    unit_divisor=1024,
) as bar:
    for data in response.iter_content(chunk_size=1024):
        bar.update(len(data))
        outfile.write(data)

# Unzip the data
zip_ref = zipfile.ZipFile("10_food_classes_all_data.zip", "r")
zip_ref.extractall()
zip_ref.close()


# Walk through  classes of food image data
food_images = 0
for dirpath, dirnames, filenames in os.walk("10_food_classes_all_data"):
    food_images += len(filenames)

print(f"Successfully loaded {food_images} food images!!")
