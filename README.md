
# Image Classification with Transfer Learning

This project builds a CNN for image classification using transfer learning with EfficientNetB0 and the CIFAR-10 dataset. A pre-trained EfficientNetB0 extracts features from images (resized to 224×224), while custom layers on top perform the classification into 10 classes.

## Overview

- **Data:**  
  - **Dataset:** CIFAR-10  
  - **Images:** Originally 32×32, resized to 224×224  
  - **Classes:** 10 classes (e.g., airplane, automobile, etc.)

- **Methodology:**  
  1. **Preprocessing:**  
     - Load data, resize images, normalize pixels, and one-hot encode labels.  
  2. **Model Building:**  
     - Utilize EfficientNetB0 with frozen weights and add global average pooling, a 128-neuron dense layer (ReLU), dropout (0.5), and a softmax output layer.  
  3. **Hyperparameter Tuning:**  
     - Use Keras Tuner and EarlyStopping to optimize dense layer size, dropout rate, and learning rate.
  4. **Evaluation & Interpretability:**  
     - Achieved ~82.75% test accuracy, generated a classification report and confusion matrix, and use Grad-CAM for visualizing model decisions.


![image](https://github.com/user-attachments/assets/eea70ec7-c2cf-4809-9050-7c09147b2496)

- **Output:**  
  - The final model is saved as `final_tuned_transfer_model.h5` for deployment.

