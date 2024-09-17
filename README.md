# Retina-blood-vessel-detection-using-UNET
This repository provides a dataset and deep learning implementation for detecting blood vessels in retina images. It includes retina images with corresponding mask images highlighting the vessels. A Jupyter notebook demonstrates the binary segmentation task, and pre-trained models in .pth format are available for use or fine-tuning.
Retina Blood Vessel Detection
Project Overview
This repository contains a binary segmentation project aimed at detecting blood vessels in retina images. The goal is to segment out blood vessels from retina images using deep learning techniques. The dataset includes paired images of retinas and their corresponding binary masks, where blood vessels are marked as the foreground and everything else as the background.


Dataset
The dataset contains retina images and their corresponding masks. The images are split into training and test sets, with each set containing two subfolders:

images/: Retina images.
masks/: Binary mask images where blood vessels are marked.
Structure of the dataset:
Train: 80 retina images and 80 corresponding masks.
Test: 20 retina images and 20 corresponding masks.
Both the training and test data are provided in zip files (train.zip and test.zip).

Models
The repository includes pre-trained models saved in .pth format:

model_v1.pth: Initial model trained on the dataset.
model_v2.pth: Fine-tuned model with improved segmentation performance.
Getting Started

Running the notebook: Open the Retina_Blood_Vessel_Detection.ipynb notebook and follow the steps to train the model or evaluate it using the pre-trained models.

Training the Model: The notebook allows you to train a binary segmentation model from scratch using the training data. You can fine-tune the pre-trained models or train a new one by modifying the code inside the notebook.

Evaluating the Model: The notebook includes a section for evaluating the model on the test set using metrics such as Dice coefficient, Intersection over Union (IoU), and pixel-wise accuracy.

Requirements
Python 3.x
PyTorch
NumPy
Matplotlib
OpenCV (for image preprocessing)
Results
The model is trained for blood vessel segmentation in retina images, with performance evaluated using standard segmentation metrics. You can explore and visualize the results in the notebook.

Future Work
Hyperparameter tuning for improved model performance.
Integration of additional segmentation metrics.
Exploration of more advanced deep learning architectures for better results.
License
This project is licensed under the MIT License.

Contact
For any questions or contributions, feel free to reach out at [ahsanejazbutt420@gmail.com].
