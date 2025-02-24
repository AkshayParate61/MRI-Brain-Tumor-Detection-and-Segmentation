# MRI-Brain-Tumor-Detection-and-Segmentation


## 📌 Project Overview

This project is an AI-based **Brain Tumor Detection and Segmentation** system using **Flask, TensorFlow, and OpenCV**. The system classifies MRI scans to detect tumors and performs segmentation to highlight affected regions.

## 🚀 Features

- ✅ Upload MRI images for **tumor classification** and **segmentation**.
- ✅ Uses **ResNet-50, VGG16, and ResUNet** for accurate predictions.
- ✅ **Flask-based web interface** for user-friendly interaction.
- ✅ Generates visual **tumor masks** on MRI scans.
- ✅ Fully automated **deep learning pipeline**.

## 🏗️ Project Structure

```bash
📂 Brain-Tumor-Detection
│── 📂 static/                 # Contains uploaded and processed images
│── 📂 saved_models/           # Pretrained models and weights
│── 📂 templates/              # HTML templates for Flask UI
│── 📄 app.py                  # Flask web server
│── 📄 server.py               # Runs the prediction pipeline
│── 📄 model.py                # CNN-based classification model
│── 📄 model_segmentation.py   # ResUNet-based segmentation model
│── 📄 loss.py                 # Custom loss functions (Tversky, Focal Tversky)
│── 📄 prediction.py           # Performs model inference
│── 📄 plot_mri.py             # Generates tumor mask visualizations
│── 📄 my_model.json           # Pretrained model architecture
│── 📄 requirements.txt        # Dependencies
│── 📄 favicon.ico             # Web app icon
```

## 🖥️ How to Run

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Run the Flask Server

```bash
python app.py
```

Open your browser and go to `http://127.0.0.1:5000/`.

### 3️⃣ Upload an MRI Scan

- Click **Upload Image**
- Get **tumor classification results**
- If detected, view the **segmented mask**

## 🧠 Model Details

- **Classification**: Uses **ResNet-50** and **VGG16** models.
- **Segmentation**: Uses **ResUNet** with custom **Tversky & Focal Tversky Loss**.
- **Dataset**: Trained on **MRI brain tumor datasets**.


## 📜 License

This project is **open-source** and available under the **MIT License**.

