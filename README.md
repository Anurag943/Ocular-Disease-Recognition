# Ocular Disease Recognition

## Introduction
This project focuses on the recognition of ocular diseases using a deep learning model. Ocular diseases encompass a wide range of eye-related conditions, including cataracts, macular degeneration, diabetic retinopathy, and more. To address this, we have implemented a solution using the ResNet50 pretrained model, which is known for its efficiency and effectiveness in image classification tasks. The model was implemented using TensorFlow and achieved an accuracy of 98%.

## Code Structure
- **`Ocular_disease_recognition.ipynb`**: This Jupyter notebook contains the core implementation of the Ocular Disease Recognition model. It includes data preprocessing, model training, evaluation, and testing using various libraries and frameworks.
- **`app.py`**: This script is for a Streamlit web application that demonstrates the model's functionality. Users can upload images, and the app will predict the ocular disease based on the trained model.

## Dependencies
The following libraries are required to run the project:

- **`Matplotlib`**: Used for data visualization, including plotting training/validation accuracy and loss curves.
- **`PIL` (Pillow)**: Utilized for image processing tasks such as loading and resizing images.
- **`TensorFlow`**: The primary deep learning framework used for building, training, and evaluating the model.
- **`Pandas`**: Employed for handling and manipulating data, particularly when working with CSV files.

## Implementation Details
1. **Data Preprocessing**:
   - Images are preprocessed by resizing and normalizing them to ensure consistency.
   - Data augmentation techniques are applied to enhance the dataset, given the limited amount of available data.

2. **Model Architecture**:
   - We utilized the ResNet50 architecture, a widely recognized convolutional neural network (CNN) that is pretrained on the ImageNet dataset.
   - The model was fine-tuned on the ocular disease dataset to adapt the features learned from ImageNet to our specific task.

3. **Training and Evaluation**:
   - The model was trained on a dataset of ocular disease images using TensorFlow.
   - Hyperparameters were optimized through experimentation to achieve the best possible accuracy while avoiding overfitting.
   - The final model achieved an accuracy of 98%, which is considered excellent for this type of medical image classification task.

4. **Deployment**:
   - The trained model was integrated into a Streamlit web application to allow users to interact with it easily.
   - The web app allows users to upload images, and the model provides a prediction of the ocular disease depicted in the image.
