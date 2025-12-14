# 📑 SmartDoc Sorter: AI-Powered Document Classification

![Status](https://img.shields.io/badge/Status-Completed-success)
![TensorFlow](https://img.shields.io/badge/ML-TensorFlow%20%7C%20Keras-orange)
![Python](https://img.shields.io/badge/Backend-FastAPI-blue)
![Angular](https://img.shields.io/badge/Frontend-Angular%2017-red)
![Docker](https://img.shields.io/badge/Deployment-Docker-blue)

## 📋 Overview

**SmartDoc Sorter** is a full-stack automated document processing solution. It utilizes a custom-trained Convolutional Neural Network (CNN) to classify uploaded document images into business categories (Invoices, Resumes, ID Cards, Contracts) in real-time.

Unlike generic API wrappers, this project features a **proprietary deep learning model** trained from scratch using TensorFlow, demonstrated within a production-ready microservices architecture.

## 🧠 The AI Model

The core of this application is a custom CNN trained on a subset of the **RVL-CDIP** dataset.

* **Framework:** TensorFlow 
* **Architecture:** Sequential CNN (3 Convolutional Layers + MaxPooling, Flatten, Dense Dropout, Softmax Output).
* **Input:** Grayscale document images resized to 224x224.
* **Performance:** Achieved ~92% accuracy on the validation set.
* **Training Code:** See `/model_training/train_classifier.ipynb` for the full training pipeline including data augmentation.

## 🛠️ Tech Stack & Architecture

### **Frontend (Angular)**
* Drag-and-drop file upload zone.
* Real-time visualization of the classification confidence (e.g., "Invoice: 98%").
* State management with Signals.

### **Backend (FastAPI)**
* Serves the trained `.h5` model.
* Handles image preprocessing (resizing, normalization) to match model input requirements.
* Exposes a REST API `POST /predict`.

### **DevOps**
* **Docker:** Multi-stage builds for frontend and backend.
* **Nginx:** Reverse proxy configuration.

## 🏗️ Workflow Diagram

```mermaid
graph LR
    A[User Uploads Img] -- Angular --> B(FastAPI Endpoint)
    B -- Preprocessing --> C[Resize & Normalize]
    C -- Tensor Input --> D{Custom TF Model}
    D -- Prediction Vector --> B
    B -- JSON Response --> A
    A -- Display Result --> User
