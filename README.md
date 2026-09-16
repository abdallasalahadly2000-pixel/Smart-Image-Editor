# Smart-Image-Editor

A professional browser-based image editing suite built with **Streamlit** and **OpenCV**.
Crop, color-grade, retouch, and swap backgrounds — all without leaving your browser.

- - - -

## ✨ Features

### 🎨 Editing
- Crop with drag-to-select, rotate (90° / 180° / custom), flip
- Resize with aspect-ratio lock + presets (FHD, Square, Portrait, Banner)
- Perspective correction (4-corner straightening)

### 🌈 Color & Light
- **Color Studio** — HSL, temperature, tint, vibrance, RGB curves
- **Light** — brightness, contrast, gamma, auto-fix
- Live histogram and live preview before applying

### 🖌️ Filters & Retouch
- B&W, Sepia, Blur, Sharpen, Invert, Emboss, Warm/Cool tones
- Object removal with freehand mask + inpainting

### 🖼️ AI Background
- Subject cut-out using cvzone + MediaPipe
- Replace background with solid color or custom image

### ⚙️ Workflow
- Full undo / reset history
- Non-destructive preview before apply
- Export as PNG or JPEG (quality control)

---


## 🚀 Live Demo


link of project : [Smart Image Editor](https://smart-image-editor-first-version.streamlit.app/)
---

## 🛠️ Installation

### Requirements
- Python **3.9 – 3.11** (⚠️ MediaPipe does not support 3.12+ yet)
- numpy
- pandas
- streamlit
- openCV

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/abdallasalahadly2000-pixel/Smart-Image-Editor.git
cd Smart-Image-Editor

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
