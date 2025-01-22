# Traffic Sign Classification with Neural Networks

This repository contains a solution for the Traffic Sign Classification problem, inspired by the [CS50’s Introduction to Artificial Intelligence with Python](https://cs50.harvard.edu/ai/2024/notes/5/). The goal of this project is to classify images of traffic signs using a neural network built with TensorFlow.

---

## Problem Overview

The primary objective is to classify images of traffic signs from the German Traffic Sign Recognition Benchmark (GTSRB) dataset, which includes **43 different categories** of traffic signs. This problem is highly relevant for applications such as autonomous vehicles, where accurate identification of traffic signs is crucial.

### **Dataset Details**
- The GTSRB dataset consists of thousands of labeled images of traffic signs.
- Each image is categorized into one of 43 classes.
- Images vary in size, illumination, and quality, making the classification task challenging.

---

## Approach

### **1. Preprocessing**
- **Image Resizing**: All images were resized to 30x30 pixels for consistency.
- **Normalization**: Pixel values were scaled to a range of 0 to 1 for faster training.
- **One-Hot Encoding**: Labels were converted to one-hot encoded vectors.

### **2. Neural Network Architecture**
We experimented with different architectures and hyperparameters to achieve the best results. The final model used the following structure:

| Layer Type         | Filters/Units | Kernel Size | Activation | Additional Info           |
|--------------------|---------------|-------------|------------|---------------------------|
| Conv2D             | 32            | 3x3         | ReLU       | Input shape: (30, 30, 3)  |
| MaxPooling2D       | -             | 2x2         | -          | -                         |
| Conv2D             | 64            | 3x3         | ReLU       | -                         |
| MaxPooling2D       | -             | 2x2         | -          | -                         |
| Conv2D             | 128           | 3x3         | ReLU       | -                         |
| MaxPooling2D       | -             | 2x2         | -          | -                         |
| Flatten            | -             | -           | -          | -                         |
| Dense              | 128           | -           | ReLU       | -                         |
| Dropout            | -             | -           | -          | Dropout rate: 0.5         |
| Dense (Output)     | 43            | -           | Softmax    | -                         |

### **3. Hyperparameters**
- **Epochs**: 20
- **Batch Size**: 32
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy

---

## Results

### **1. Our Results**
| Metric        | Value       |
|---------------|-------------|
| Test Accuracy | **98.39%**  |
| Test Loss     | **0.0911**  |

### **2. Professor's Results**
| Metric        | Value       |
|---------------|-------------|
| Test Accuracy | **95.35%**  |
| Test Loss     | **0.1616**  |

### **Analysis**
- Our approach achieved higher accuracy (**98.39%**) and lower test loss (**0.0911**) compared to the professor's results.
- The improvement can be attributed to:
  - An additional convolutional layer (128 filters) in our architecture.
  - A higher dropout rate (0.5) to prevent overfitting.
  - Training for 20 epochs, providing the model more time to learn.

These results demonstrate that careful tuning of hyperparameters and model complexity can significantly improve performance on image classification tasks.

---

## Observations

This project shows the importance of deep learning in solving real-world problems, such as traffic sign classification for autonomous vehicles. While our model achieved excellent results, further improvements could be made by:
- Using data augmentation to increase robustness.
- Experimenting with transfer learning using pretrained models.

---

## Disclaimer

This project was created and made publicly available for **collaboration between programmers**. It is not intended for plagiarism and must not be used to violate academic integrity policies. All work is subject to the rules and regulations of **Harvard University** and **edX**. Violations may result in severe consequences.

---

Thank you for reviewing this project. Contributions and suggestions are always welcome!
