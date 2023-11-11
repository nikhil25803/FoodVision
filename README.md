## FoodVision
FoodVision is a **Python** project that effectively utilizes **Convolutional Neural Networks (CNN)** for food
**image classification**. The adoption of best practices, automation through
shell scripting, and the inclusion of both command line and GUI interfaces
contribute to its accessibility and usability. Future improvements aim to enhance
the model's performance and user experience.


## Tools and Software Used
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black) ![Shell Script](https://img.shields.io/badge/shell_script-%23121011.svg?style=for-the-badge&logo=gnu-bash&logoColor=white)

## Project Structure

```md
FoodVision
 ├── 📄  LICENSE
 ├── 📄  README.md
 ├── 📄  .gitignore
 ├── 📄  requirements.txt - Contains all the dependencies
 ├── 📄  GUI.py - Generates a user-friendly GUI for image classification.
 ├── 📄  prediction.py - Allows image prediction using command line arguments.
 ├── 📄  main.py -  Builds and trains CNN models.
 └── 📂  data/ - Contains datasets and Python files for automating the download and structuring.
 └── 📂  models/ - Holds various CNN models along with their accuracy and loss curves.
 └── 📂  utils/ - Includes helper functions for data preprocessing and model evaluation.
```

## Project Setup

+ Create and activate the virtual environment
```bash
python -m venv env
```

> For windows
```bash
source env/Scripts/Activate
```

> For Linux
```bash
source env/bin/activate
```

+ Install Dependencies
```bash
pip install -r requirements.txt
```

+ Make a prediction - Command Line Arguments (CLI)
```bash
python prediction.py --help
```
> Response
```bash
Predict and plot food images.

options:
  -h, --help     show this help message and exit
  --model MODEL  Model name or number
  --food FOOD    Food class name
```

```bash
python prediction.py --model 1 --food pizza
```
> Response
![image](https://github.com/nikhil25803/FoodVision/assets/93156825/7c06c01a-8c38-4490-a002-3e7f8338c100)

+ Make a prediction - Through GUI
```bash
python GUI.py
```

> Response
![image](https://github.com/nikhil25803/FoodVision/assets/93156825/c7c1377e-78df-49c9-a412-d026c4a41685)

## Customization?
If you are looking to enhance the model accuracy, run the `setup.sh` bash file to avail all the images you'll need for building the model.
```bash
bash setup.sh
```
> This will automatically download and unpack the zip file and will delete unnecessary data as well.
