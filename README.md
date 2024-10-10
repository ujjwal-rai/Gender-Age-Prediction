---

# Age and Gender Prediction using Convolutional Neural Network (CNN)

This repository contains the code for predicting the age and gender of individuals using facial images. The model is built using Convolutional Neural Networks (CNN) and trained on the UTKFace dataset.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Results](#results)
- [Dependencies](#dependencies)
- [Contributing](#contributing)

## Overview

This project predicts a person's age and gender from facial images using a deep learning CNN model. The model is trained on the UTKFace dataset, which includes images labeled with age, gender, and ethnicity. The network is implemented using TensorFlow and Keras, with age prediction treated as a regression task and gender prediction as a binary classification task.

## Dataset

The dataset used in this project is [UTKFace](https://www.kaggle.com/datasets/jangedoo/utkface-new), which contains over 20,000 images of faces labeled with age, gender, and ethnicity. We primarily use the age and gender labels for this project.

To download the dataset:
```bash
kaggle datasets download -d jangedoo/utkface-new
```

## Model Architecture

The model is a Convolutional Neural Network (CNN) that consists of the following layers:
- 4 convolutional layers with MaxPooling.
- Fully connected (Dense) layers for extracting high-level features.
- Dropout layers to prevent overfitting.
- Two output layers: one for gender classification (binary classification) and another for age prediction (regression).

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/age-gender-prediction.git
   cd age-gender-prediction
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset:
   ```bash
   kaggle datasets download -d jangedoo/utkface-new
   unzip utkface-new.zip -d ./UTKFace
   ```

4. Run the notebook or the Python script to train the model:
   ```bash
   python train_model.py
   ```

## Usage

You can run the model to predict gender and age from a set of images. Below is an example of how to use the trained model to make predictions on a given image.

```python
# Load an image
image_index = 100
pred = model.predict(X[image_index].reshape(1, 128, 128, 1))

# Predicted gender and age
pred_gender = gender_dict[round(pred[0][0][0])]
pred_age = round(pred[1][0][0])

print(f"Predicted Gender: {pred_gender}, Predicted Age: {pred_age}")
```

## Results

The model achieves reasonable accuracy in predicting gender and age from facial images. Below are the performance metrics:
- Gender classification accuracy: **(mention accuracy here)**.
- Age prediction mean absolute error (MAE): **(mention MAE here)**.

Visualization of training progress is shown through loss and accuracy plots.

## Dependencies

- Python 3.x
- TensorFlow
- Keras
- Matplotlib
- Seaborn
- NumPy
- Pandas
- TQDM

Install the dependencies using the following command:
```bash
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements, bug fixes, or suggestions.
---
