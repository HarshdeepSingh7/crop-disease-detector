# Crop Disease Detection Project

This repository contains an end-to-end Python pipeline to train a Deep Learning model (MobileNetV2) to classify 4 specific crop diseases using the Kaggle `plantdisease` dataset.

## Setup and Requirements

1. **Install dependencies**:
   ```bash
   pip install tensorflow numpy matplotlib seaborn scikit-learn kaggle
   ```

2. **Kaggle API Setup**:
   To download the dataset automatically, you must configure the Kaggle API.
   - Go to Kaggle -> Settings -> Create New API Token.
   - Download the `kaggle.json` file.
   - Place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `C:\Users\<User>\.kaggle\` (Windows).
   - Ensure permissions are secure (`chmod 600 ~/.kaggle/kaggle.json` on Linux/Mac).

## Running Locally

Simply execute the main script:
```bash
python main.py
```
This script will:
- Download the `plantdisease` dataset.
- Extract the 4 requested classes and delete the rest to save space.
- Train a MobileNetV2 transfer learning model.
- Evaluate the model, generate a confusion matrix in `models/`, and print the classification report.

## Running on Google Colab

If you want to run this in Google Colab, create a new Notebook and run the following cells:

**Cell 1: Setup Kaggle API**
```python
from google.colab import files
import os

print("Upload your kaggle.json file:")
files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

**Cell 2: Run the code**
Upload the `main.py` file to your Colab environment, then run:
```python
!python main.py
```

## Structure

- `data/`: Contains the filtered dataset (created at runtime).
- `models/`: Stores the best trained model `crop_disease_model.h5` and evaluation plots.
- `notebooks/`: Directory for any Jupyter notebooks you might create.
- `main.py`: The complete end-to-end pipeline script.
