# Deepfake Detection System

## 📌 Overview
The **Deepfake Detection System** is an advanced deep learning pipeline designed for image forgery detection. It utilizes a **fine-tuned InceptionResNetV1 model**, trained on deepfake and authentic image datasets, to classify images as real or fake. The model leverages **convolutional neural networks (CNNs)** for feature extraction and **fully connected layers** for binary classification. Integrated **Grad-CAM visualization** provides interpretability, and a **Gradio-based web interface** enables easy testing. The system supports **CUDA acceleration** for enhanced performance.

## 🚀 Features
- **Pretrained InceptionResNetV1** for deepfake detection
- **Binary classification** (Real vs Fake) using deep learning
- **Grad-CAM support** for interpretability and model explainability
- **Optimized training with Adam optimizer and Cross-Entropy loss**
- **Batch normalization and dropout** for improved generalization
- **CUDA acceleration** for faster inference on GPUs
- **TensorBoard integration** for real-time performance monitoring
- **Gradio-based UI** for accessible web-based interaction

## 🛠 Tech Stack
- **Python**, **PyTorch**, **torchvision**
- **TensorBoard** for training visualization
- **pytorch-grad-cam** for model interpretability
- **Gradio** for user-friendly deployment
 
## 📌 Acknowledgments
- **Facenet-PyTorch** for pretrained models
- **pytorch-grad-cam** for explainability
- **Gradio** for an easy-to-use web UI

## 📜 License
This project is open-source under the **MIT License**.

