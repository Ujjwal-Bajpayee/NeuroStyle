# 🧠🎨 NeuroStyle

**NeuroStyle** is a neural style transfer web application built with Streamlit and PyTorch. It allows users to blend the content of one image with the artistic style of another — creating beautiful, AI-generated art in just seconds!

## 🌈 What is Neural Style Transfer?

Neural Style Transfer uses deep learning to apply the visual style of one image (typically artwork) to the content of another (such as a photograph), producing a new image that merges both.

## 🚀 Features

- 📁 Upload your own **content** and **style** images
- 🖼 Use built-in sample images
- 🎛 Adjust **style intensity** with an interactive slider
- ⚡ Fast processing with pretrained VGG-19 model
- 💾 Download the final stylized image

## 🛠️ Tech Stack

- [Python](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [PyTorch](https://pytorch.org/)
- [torchvision](https://pytorch.org/vision/stable/)
- [Pillow](https://pillow.readthedocs.io/)

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/neurostyle.git
   cd neurostyle

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate        
    
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt

4. **Run the app**
   ```bash
   streamlit run Style.py

If no images are uploaded, the app defaults to:
image.jpg – content image
demo.jpg – style image
These are loaded automatically when the "Use Sample Images" toggle is on.

 **Project Structure**
   ```bash
   ├── Style.py            # Main Streamlit app
   ├── image.jpg           # Default content image
   ├── demo.jpg            # Default style image
   ├── requirements.txt    # Dependencies
   ├── README.md           # Documentation
   └── .gitignore          # Git ignore rules

