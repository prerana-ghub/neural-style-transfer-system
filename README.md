## 🎨 Neural Style Transfer System – Deep Learning (Streamlit)

This project is a **Neural Style Transfer (NST) system** that blends the content of one image with the artistic style of another using **Deep Learning**.
The application is built using **Python**, **TensorFlow/Keras (VGG19)**, and **Streamlit** for an interactive web-based interface.

## 📌 Features

* Upload **content image** and **style image**
* Generate a **stylized output image**
* Adjust **style intensity** using a slider 🎚️
* View results directly in the browser
* Download the generated stylized image 📥
* Simple and user-friendly web interface

## 🛠 Requirements

* **Python 3.10 or higher**
* **TensorFlow / Keras**
* **Streamlit**
* **Pillow (PIL)**
* **NumPy**
* **Matplotlib** (for loss visualization, if enabled)
* Code editor: **VS Code** (recommended)

All required libraries are listed in `requirements.txt`.

## ✅ Steps to Run the Project

### 1. 📦 Clone the Repository

```bash
git clone https://github.com/yourusername/neural-style-transfer-streamlit.git
cd neural-style-transfer-streamlit
```

### 2. 📁 Open Project Folder

* Open **VS Code**
* Click **File → Open Folder**
* Select the project root folder

### 3. 📦 Create Virtual Environment (Optional but Recommended)

```bash
python -m venv .venv
```

Activate it:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### 4. 📥 Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. ▶️ Run the code

```bash
streamlit run app.py
```

## 📁 Project Structure

```
├── app.py                # Streamlit frontend
├── nst.py                # Neural Style Transfer backend logic
├── requirements.txt      # Project dependencies
├── images/               # Sample input/output images
├── .streamlit/           # Streamlit configuration
├── Neural Style Transfer.png
```

## 📚 Technologies Used

* Python
* TensorFlow / Keras
* VGG19 (Pretrained CNN)
* Streamlit
* NumPy
* Pillow (PIL)

## 📌 Outcome

The application successfully demonstrates **Neural Style Transfer** by preserving the structural content of an image while applying artistic textures from a style image.
It provides an easy-to-use platform for experimenting with deep learning–based image stylization.

* or a **deployment-ready version**

just say it 👍
