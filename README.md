# Image-Classification-cnn
Image Classification using CNN
📌 Project Overview

This project implements a Convolutional Neural Network (CNN) to classify images of cats and dogs using a dataset of pet images. The model is trained using TensorFlow and Keras, leveraging image preprocessing, data augmentation, and multiple CNN architectures to achieve optimal classification accuracy.

🏗️ Tech Stack

Programming Language: Python
Deep Learning Framework: TensorFlow, Keras
Image Processing: OpenCV, NumPy
Data Storage: Pickle for saving training data
Logging & Monitoring: TensorBoard
Model Evaluation: Matplotlib for visualization
📌 Key Features

Data Preprocessing: Converts images to grayscale, resizes to 50x50 pixels.
Dataset Handling: Extracts and labels images of cats and dogs from a directory.
CNN Model Architecture:
Multiple convolutional layers with ReLU activation.
Max-pooling for feature extraction.
Fully connected layers with Sigmoid activation for binary classification.
Hyperparameter Tuning:
Experiments with different numbers of convolutional layers, dense layers, and neuron sizes.
Training and Evaluation:
Trained with binary cross-entropy loss and Adam optimizer.
Uses validation split to monitor overfitting.
Performance Tracking:
Logs training results using TensorBoard for model comparison.
🚀 Results & Insights

Multiple CNN architectures were tested, varying layer depth and neuron count.
Model accuracy improved with increased convolutional layers and proper pooling strategies.
The best-performing model achieved high accuracy on the validation set.
Further improvements could be made using data augmentation and transfer learning.
🔮 Future Enhancements

Implement data augmentation to improve generalization.
Experiment with pre-trained CNN models like VGG16 or ResNet.
Deploy the model as an API for real-time image classification.
👨‍💻 Author

Shravan Sundar Ravi – Data Scientist & Deep Learning Enthusiast
