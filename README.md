# 🚦 Traffic Sign Recognition Using CNN

This project is a deep learning-based Traffic Sign Recognition system developed using **Keras** and **TensorFlow**. It classifies traffic signs into **43 different categories**. The model is trained on images using **ImageDataGenerator** for data augmentation and the **Adam optimizer** for efficient convergence.

## 📂 Dataset
* **Total Classes:** 43 traffic sign categories
* **Data Format:** Images organized in class-wise folders
* **Preprocessing:** Resized, normalized, and augmented using `ImageDataGenerator`

## 🧠 Model Architecture

* **Architecture:**
  * Convolutional layers (ReLU + MaxPooling)
  * Dropout layers to reduce overfitting
  * Fully connected (Dense) layers
  * Softmax activation for multi-class classification
* **Optimizer:** Adam
* **Loss Function:** Categorical Crossentropy
* **Evaluation Metrics:** Accuracy

## 📈 Training

* **Augmentation:** Rotation, zoom, shift, shear, etc.
* **Epochs:** 30 (based on training)
* **Batch Size:** 42

## 📊 Results

* **Accuracy:** 99.35%
* **Model Saved:** `traffic_sign_model.h5`

## 🚀 Future Improvements

* Convert to TensorFlow Lite or ONNX for mobile deployment
* Add real-time video stream detection using OpenCV
* Improve accuracy with advanced architectures (ResNet, MobileNet)

