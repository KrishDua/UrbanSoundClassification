# UrbanSound Classification Project

This project focuses on environmental sound classification using the **UrbanSound8K** dataset. It uses **MFCC-based feature extraction** and a **Convolutional Neural Network (CNN)** to identify urban sounds like dog barks, sirens, drilling, and more.

It showcases a pipeline combining **data preprocessing**, **deep learning**, and **visualization** in Python using TensorFlow and supporting libraries.

---

## Project Overview

This project allows you to:

- Load and preprocess audio files from the UrbanSound8K dataset
- Extract MFCC features using `librosa`
- Train a CNN model on the extracted features
- Evaluate the model using accuracy, AUC, precision, recall
- Visualize training history, confusion matrix, and classification report
- Achieve over **92% classification accuracy** on the test set

---

## Features

- MFCC-based feature extraction from WAV files
- Min-Max feature scaling
- One-hot label encoding
- CNN model with multiple Conv1D + Dense layers
- Model evaluation using:
  - Accuracy
  - Precision
  - Recall
  - AUC
- Visualization of:
  - Loss/Accuracy Curves
  - Confusion Matrix
  - Classification Report Heatmap

---

## Technologies Used

### Core Libraries & Tools

- Python
- TensorFlow / Keras
- Librosa
- NumPy / Pandas
- Matplotlib / Seaborn
- scikit-learn
- tqdm

---

## Model Architecture

- `Conv1D (64 filters, kernel size=5, ReLU)`
- `MaxPooling1D`
- `Conv1D (64 filters, kernel size=5, ReLU)`
- `MaxPooling1D`
- `Flatten`
- `Dense (256, ReLU) + Dropout`
- `Dense (128, ReLU) + Dropout`
- `Dense (64, ReLU)`
- `Dense (10, Softmax)`

**Optimizer**: Adam  
**Loss Function**: Categorical Crossentropy  
**Epochs**: 100  
**Batch Size**: 128

---

## Results

| Metric       | Value         |
|--------------|---------------|
| Accuracy     | 92.33%        |
| AUC          | 0.9859        |
| Precision    | 0.9253        |
| Recall       | 0.9222        |
| Test Loss    | 0.4465        |

---

## Dataset

- **Name**: UrbanSound8K  
- **Size**: 8732 labeled .wav audio clips  
- **Labels**: 10 sound categories  
- **Metadata File**: `UrbanSound8K/metadata/UrbanSound8K.csv`  
- **Audio Files Folder**: `UrbanSound8K/audio/`

You can download the dataset from:  
https://urbansounddataset.weebly.com/urbansound8k.html

---

## APIs / Integrations

- `librosa` – Audio loading and MFCC feature extraction  
- `scikit-learn` – Train/test split, scaling, and evaluation  
- `matplotlib` & `seaborn` – Plotting model performance  

---

## Author

**Krish Dua**  
Portfolio: https://krishdua.vercel.app  
LinkedIn: https://www.linkedin.com/in/krish-dua

---
