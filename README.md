# Facial Emotion Recognition Using CNN

## 👋 Project Overview
This project focuses on building a **Facial Emotion Recognition (FER) system** using **Convolutional Neural Networks (CNNs)**. The model can detect and classify human emotions from facial images in real-time, enabling applications in **human-computer interaction, mental health monitoring, and AI-powered emotion analysis**.

---

## 📂 Dataset
- **Dataset Used:** FER-2013 (Facial Expression Recognition 2013)  
- **Classes:**  
  - Angry 😠  
  - Disgust 🤢  
  - Fear 😨  
  - Happy 😀  
  - Sad 😢  
  - Surprise 😲  
  - Neutral 😐  
- **Preprocessing Steps:**  
  - Converted images to **grayscale**.  
  - Resized all images to **48x48 pixels**.  
  - Normalized pixel values to range **0–1**.  
  - Applied **data augmentation** for better generalization.

---

## 🛠 Methodology
1. **Data Loading & Preprocessing:**  
   Load the FER dataset, convert to grayscale, normalize, and split into **training, validation, and test sets**.  

2. **CNN Architecture:**  
   - **Input Layer:** 48x48 grayscale images  
   - **Conv Layer 1:** 32 filters, 3x3 kernel, ReLU activation  
   - **MaxPooling 1:** 2x2 pool size  
   - **Conv Layer 2:** 64 filters, 3x3 kernel, ReLU activation  
   - **MaxPooling 2:** 2x2 pool size  
   - **Conv Layer 3:** 128 filters, 3x3 kernel, ReLU activation  
   - **Flatten Layer**  
   - **Fully Connected Layer:** 128 neurons, ReLU activation  
   - **Dropout Layer:** 0.5 (to prevent overfitting)  
   - **Output Layer:** 7 neurons, Softmax activation (for emotion classification)

3. **Model Compilation:**  
   - Loss Function: `categorical_crossentropy`  
   - Optimizer: `Adam`  
   - Metrics: `accuracy`

4. **Training:**  
   - Batch size: 64  
   - Epochs: 50–100 (depending on dataset size and GPU availability)  
   - Validation used to monitor overfitting  

---

## 🧠 Technologies Used
- **Programming Language:** Python  
- **Libraries:** TensorFlow/Keras, OpenCV, NumPy, Matplotlib, Seaborn  
- **IDE:** Google Colab / VS Code  

---

## 📊 Results
- The CNN model achieved high accuracy on the validation set (typically around **65–70%** depending on training).  
- Real-time emotion detection implemented using **OpenCV webcam feed**.  
- Confusion Matrix and Accuracy/Loss plots were used to evaluate performance.

---

## 🔮 Future Scope
- Improve accuracy using **deeper CNN architectures** or **pretrained models** like VGG16 or ResNet.  
- Extend to **video-based emotion recognition**.  
- Integrate with **chatbots or virtual assistants** for enhanced human-computer interaction.  
- Apply in **mental health monitoring**, **driver drowsiness detection**, or **classroom engagement analysis**.

---

## 📫 Contact
- Email: mfaizankh007@gmail.com  
- GitHub: https://github.com/FAIZAVENGER

---

> "Emotions are the window to the mind, and AI helps us understand them."  
