# YOLO Object Detection with Streamlit

🚀 **Real-Time Object Detection App using YOLOv3-Tiny and Streamlit**

## 📌 Overview
This project implements an **object detection web app** using **YOLO (You Only Look Once) and Streamlit**. Users can **upload videos** for real-time detection, and the app processes each frame to identify objects using the YOLO model. The UI is built using Streamlit for an interactive experience.

## 🎯 Features
✅ Upload video for real-time object detection  
✅ Uses **YOLOv3-Tiny** for fast and efficient inference  
✅ Displays **bounding boxes, labels & confidence scores**  
✅ **Stop Detection button** for better control  
✅ FPS counter to track real-time performance  
✅ Simple and interactive **Streamlit UI**  
✅ Optimized **temporary file handling** for smooth execution  
✅ Supports **multiple object detection** per frame  

## 🛠 Tech Stack
- **Python**
- **OpenCV**
- **Streamlit**
- **YOLOv3-Tiny**
- **NumPy**

## 📥 Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/yolo-object-detection-streamlit.git
   cd yolo-object-detection-streamlit
   ```
2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv env
   source env/bin/activate  # MacOS/Linux
   env\Scripts\activate    # Windows
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Download YOLO weights and config** (if not included in repo)
   - Download **yolov3-tiny.weights** and **yolov3-tiny.cfg** from [YOLO official site](https://pjreddie.com/darknet/yolo/)
   - Place them in the project directory

## 🚀 Usage
1. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```
2. **Upload a video and start object detection!**
3. **Use the stop button to end detection at any time**

## 📸 Demo
📷 **Screenshot of the UI in action** (Attach a sample image)
📽️ **Demo Video** (Attach a link to a video demonstration)

## 🔍 Future Improvements
- [ ] Upgrade to **YOLOv8** for enhanced accuracy and speed
- [ ] Add **real-time Webcam support**
- [ ] Implement **custom object detection** for specific use cases
- [ ] Deploy on **Streamlit Cloud / Hugging Face Spaces**
- [ ] Optimize inference speed for large videos
- [ ] Add a **dark mode UI option**

## 🤝 Contributing
Contributions are welcome! Feel free to fork this repository, create a new branch, and submit a pull request.  

## 📜 License
This project is licensed under the **MIT License**.

---

🔗 **Let's connect!**  
If you have suggestions or improvements, feel free to open an **issue** or submit a **pull request**! 🎯
