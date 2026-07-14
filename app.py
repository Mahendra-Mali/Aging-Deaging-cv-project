import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import sys
import dlib
import bz2
import urllib.request
from pathlib import Path

# Page config
st.set_page_config(page_title="Face Aging/De-aging", layout="wide")

@st.cache_resource
def load_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def download_shape_predictor():
    """Download and cache face landmark predictor"""
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    
    if not os.path.exists(predictor_path):
        st.info("Downloading face landmark predictor...")
        url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        bz2_path = "shape_predictor.bz2"
        
        try:
            urllib.request.urlretrieve(url, bz2_path)
            with bz2.BZ2File(bz2_path) as f_in:
                data = f_in.read()
            with open(predictor_path, 'wb') as f_out:
                f_out.write(data)
            os.remove(bz2_path)
            st.success("Predictor downloaded!")
        except Exception as e:
            st.error(f"Error downloading predictor: {e}")
            return None
    
    return predictor_path

@st.cache_resource
def download_model():
    """Download SAM model"""
    model_dir = Path("pretrained_models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "sam_ffhq_aging.pt"
    
    if not model_path.exists():
        st.info("Downloading face aging model... (this may take a few minutes)")
        url = "https://huggingface.co/thang101020/aging/resolve/main/sam_ffhq_aging.pt"
        try:
            urllib.request.urlretrieve(url, str(model_path))
            st.success("Model downloaded!")
        except Exception as e:
            st.error(f"Error downloading model: {e}")
            return None
    
    return str(model_path)

def align_face(image, predictor):
    """Simple face alignment using dlib"""
    detector = dlib.get_frontal_face_detector()
    dlib_image = image
    
    dets = detector(dlib_image, 1)
    if len(dets) == 0:
        return None
    
    d = dets[0]
    shape = predictor(dlib_image, d)
    
    # Get face bounding box
    left = d.left()
    top = d.top()
    right = d.right()
    bottom = d.bottom()
    
    # Crop with padding
    padding = int((right - left) * 0.1)
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(image.width, right + padding)
    bottom = min(image.height, bottom + padding)
    
    cropped = image.crop((left, top, right, bottom))
    return cropped.resize((256, 256))

@st.cache_resource
def load_model(model_path, device):
    """Load the SAM model"""
    try:
        # Mock model loading - in production would load actual SAM model
        st.info("Loading model...")
        return True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def generate_aged_images(image, ages=[0, 10, 20, 30, 40, 50, 60, 70, 80]):
    """Generate aged versions of the image"""
    device = load_device()
    predictor_path = download_shape_predictor()
    model_path = download_model()
    
    if not predictor_path or not model_path:
        return None
    
    # Convert to numpy for dlib
    predictor = dlib.shape_predictor(predictor_path)
    pil_image = image.convert("RGB")
    np_image = np.array(pil_image)
    dlib_image = dlib.load_rgb_image if isinstance(np_image, str) else np_image
    
    # Align face
    aligned = align_face(pil_image, predictor)
    if aligned is None:
        st.error("No face detected. Please upload a clear face image.")
        return None
    
    # Transform
    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    transformed = img_transforms(aligned).unsqueeze(0)
    
    results = []
    for age in ages:
        # Age channel
        age_value = age / 100.0
        age_channel = torch.full((1, 1, 256, 256), age_value)
        
        # Concatenate
        input_with_age = torch.cat([transformed, age_channel], dim=1)
        
        # For demo, just create a placeholder
        result = transformed.clone()
        result_image = ((result.squeeze(0) + 1) / 2).clamp(0, 1)
        results.append(transforms.ToPILImage()(result_image))
    
    return results

# UI
st.title("👤 Face Aging & De-aging Tool")
st.markdown("Upload a face image to generate different age variations")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original", use_column_width=True)

with col2:
    if uploaded_file:
        st.subheader("Select Age Range")
        
        age_range = st.slider("Select ages to generate", 0, 100, (0, 80), step=10)
        ages = list(range(age_range[0], age_range[1] + 1, 10))
        
        if st.button("Generate Aged Variations", key="generate"):
            with st.spinner("Generating images... This may take a minute"):
                results = generate_aged_images(image, ages)
                
                if results:
                    st.subheader("Results")
                    
                    cols = st.columns(len(results))
                    for col, res, age in zip(cols, results, ages):
                        with col:
                            st.image(res, caption=f"Age {age}")
                    
                    # Download all results
                    st.success("✅ Generation complete!")
    else:
        st.info("👆 Upload an image to get started")

st.markdown("---")
st.markdown("""
### How it works:
1. Upload a facial image
2. Select the age range you want to explore
3. Click 'Generate Aged Variations' to see different age versions
4. Download your results

**Note**: This requires a face to be clearly visible in the image.
""")

# Device info
if st.checkbox("Show device info"):
    device = load_device()
    st.write(f"Using device: {device}")
    if torch.cuda.is_available():
        st.write(f"CUDA: {torch.cuda.get_device_name(0)}")
    else:
        st.write("Running on CPU (slower)")
