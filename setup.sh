echo "Food Vision Project Setup"

echo "Installing Dependencies"
pip install -r requirements.txt

echo "Changing directory to 'data' folder."
cd data
echo "Downloading images"
python load_image_script.py

echo "Deleting unnecessary data"
rm "10_food_classes_all_data.zip"
rm -r "__MACOSX"
rmdir "__MACOSX"

echo "Rooting back to root directory"
cd ..