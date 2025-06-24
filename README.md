
# 🤟 Hand Sign Language Detection using MediaPipe, OpenCV & Machine Learning

> **“Bridging Communication Gaps with Real-Time AI”**

This project is a real-time **American Sign Language (ASL) hand gesture detection system** that uses your **webcam** to recognize signs (A–Z and 0–9). It leverages **MediaPipe** for hand tracking and a **Random Forest Classifier** for gesture classification. The system is lightweight, fast, and runs directly in your browser (via Streamlit) or desktop.

---

## 🚀 Features

- 🖐️ Real-time hand sign recognition from webcam
- 🔍 Detects 36+ signs (A-Z, 0–9, Empty)
- 🎯 100% model accuracy on test data
- ⚡ Built with MediaPipe + OpenCV for fast and robust landmark detection
- 🧠 Machine learning with scikit-learn (Random Forest Classifier)
- 🌐 Web-based interface using Streamlit + WebRTC
- 💻 Desktop version also included

---

## 📂 Project Structure

📁 data/ → Image dataset per class (captured via webcam)
📄 collect_imgs.py → Capture images for each gesture
📄 create_dataset.py → Process images using MediaPipe & generate features
📄 train_classifier.py → Train the Random Forest Classifier
📄 Accuricy_Check.py → Evaluate model with accuracy & confusion matrix
📄 ASL_Detect_App.py → Real-time detection using OpenCV GUI
📄 Streamlit_App.py → Web-based interface for live demo

---

## 🎓 Tech Stack

| Tech | Purpose |
|------|---------|
| **Python 3.x** | Core programming language |
| **MediaPipe** | Real-time hand landmark detection |
| **OpenCV** | Webcam handling and image processing |
| **scikit-learn** | Model training (Random Forest) |
| **Streamlit** | Deploy real-time detection in web browser |
| **pickle** | Model & data serialization |

---

## 📈 Model Performance

- 📊 **Accuracy:** 100%
- ✔️ Perfect precision, recall, and f1-score across all 37 classes
- 🧪 Trained on normalized landmark data (21 points × 2 = 42 features)
- 📉 Lightweight & no deep learning framework dependency

---

## 📸 Screenshots

> *(Add screenshots here in your GitHub repo)*

- Confusion Matrix  
- Classification Report  
- Real-time prediction in webcam feed  

---

## 🔮 Future Scope

- 🔤 Sentence-level sign recognition (gesture sequences)
- 🌍 Multilingual support (ISL, BSL, etc.)
- 📱 Mobile deployment with TensorFlow Lite
- 🔈 Voice output for gesture-to-speech conversion
- 👨‍💻 Custom gesture training by users

---

## 👨‍💻 Author

**Tanmay Dhar**  
📚 MBA – Business Analytics & Data Science  
📍 BIBS Kolkata
