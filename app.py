"""
DermScan AI - Skin Lesion Classification App
8-Class Dermoscopic Image Classifier using BEiT v2 Large (Vision Transformer)
Trained on 44,000+ images with patient-level splitting and SOTA techniques
"""

# ✅ STREAMLIT MUST BE IMPORTED FIRST
import streamlit as st

# Then all other imports
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import requests
from io import BytesIO
import numpy as np
import plotly.graph_objects as go
import cv2
import base64

# -------------------------
# Configuration
# -------------------------
MODEL_URL = "https://huggingface.co/Skindoc/streamlit14/resolve/main/best_model.pth"
MODEL_NAME = "beitv2_large_patch16_224"
NUM_CLASSES = 8
IMG_SIZE = 384

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'scc', 'vasc']

CLASS_INFO = {
    'akiec': {'full_name': 'Actinic Keratoses (AKIEC)', 'description': 'Pre-cancerous lesions caused by sun damage. Requires monitoring and treatment.', 'risk': 'Medium', 'color': '#FFA500'},
    'bcc': {'full_name': 'Basal Cell Carcinoma (BCC)', 'description': 'Most common skin cancer. Slow-growing, rarely spreads, highly treatable.', 'risk': 'High', 'color': '#FF4444'},
    'bkl': {'full_name': 'Benign Keratosis (BKL)', 'description': 'Non-cancerous skin growth. Generally harmless but may be removed for cosmetic reasons.', 'risk': 'Low', 'color': '#90EE90'},
    'df': {'full_name': 'Dermatofibroma (DF)', 'description': 'Benign fibrous nodule. Usually harmless and does not require treatment.', 'risk': 'Low', 'color': '#87CEEB'},
    'mel': {'full_name': 'Melanoma (MEL)', 'description': 'Most dangerous skin cancer. Can spread rapidly. Requires immediate medical attention.', 'risk': 'Critical', 'color': '#8B0000'},
    'nv': {'full_name': 'Melanocytic Nevi (NV)', 'description': 'Common moles. Generally benign but should be monitored for changes.', 'risk': 'Low', 'color': '#98FB98'},
    'scc': {'full_name': 'Squamous Cell Carcinoma (SCC)', 'description': 'Second most common skin cancer. Can spread if untreated. Requires treatment.', 'risk': 'High', 'color': '#FF6347'},
    'vasc': {'full_name': 'Vascular Lesions (VASC)', 'description': 'Blood vessel abnormalities. Usually benign (e.g., cherry angiomas, hemangiomas).', 'risk': 'Low', 'color': '#DDA0DD'}
}

# -------------------------
# Custom CSS
# -------------------------
def set_theme(background_color='#0E1117'):
    css = f"""
    <style>
    .stApp {{ background-color: {background_color}; background-image: none; }}
    .main .block-container {{ background-color: rgba(18, 18, 18, 0.8); padding: 4rem; border-radius: 12px; }}
    h1, h2, h3, h4, .stMarkdown, .stText, label, p {{ color: #F0F2F6 !important; }}
    [data-testid="stSidebar"] {{ background-color: rgba(30, 30, 30, 0.95); color: #F0F2F6; }}
    hr {{ border-top: 1px solid #333; }}

    /* Confidence indicator styling */
    .confidence-badge {{
        display: inline-block;
        padding: 5px 12px;
        border-radius: 15px;
        font-weight: bold;
        margin-left: 10px;
    }}
    .high-confidence {{
        background-color: #4CAF50;
        color: white;
    }}
    .medium-confidence {{
        background-color: #FFC107;
        color: black;
    }}
    .low-confidence {{
        background-color: #FF5722;
        color: white;
    }}

    /* Heatmap animation */
    .gradcam-container {{
        position: relative;
        margin-top: 1.5rem;
    }}
    .gradcam-overlay {{
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        opacity: 0;
        animation: gradcamFadeIn 0.3s ease-in 2.5s forwards;
    }}
    .gradcam-reveal-mask {{
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(255, 255, 255, 0.12);
        clip-path: polygon(0% 0%, 0% 100%, 0% 100%, 0% 0%);
        animation: gradcamReveal 2.2s ease-in-out 0.3s forwards;
    }}
    @keyframes gradcamReveal {{
        0% {{ clip-path: polygon(0% 0%, 0% 100%, 0% 100%, 0% 0%); }}
        100% {{ clip-path: polygon(0% 0%, 100% 0%, 100% 100%, 0% 100%); }}
    }}
    @keyframes gradcamFadeIn {{
        to {{ opacity: 1; }}
    }}
    .gradcam-caption {{
        text-align: center;
        margin-top: 10px;
        font-size: 0.95em;
        color: #aaa;
        font-style: italic;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    
# -------------------------
# Model Loading
# -------------------------
@st.cache_resource
def load_model():
    try:
        with st.spinner("Downloading BEiT v2 Large model (this may take a minute on first run)..."):
            response = requests.get(MODEL_URL, timeout=120)
            response.raise_for_status()
        
        checkpoint = torch.load(BytesIO(response.content), map_location='cpu')
        
        # Create BEiT v2 Large model (with img_size=384 for proper interpolation)
        model = timm.create_model(
            MODEL_NAME, 
            pretrained=False, 
            num_classes=NUM_CLASSES,
            img_size=IMG_SIZE
        )
        
        # Load state dict
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# -------------------------
# Grad-CAM for Vision Transformers
# -------------------------
def generate_gradcam_vit(model, image_tensor, predicted_class):
    """
    Grad-CAM adapted for Vision Transformers (BEiT v2).
    Targets the last layer norm before the classification head.
    """
    try:
        # For BEiT, use the layer norm before head
        target_layer_name = "norm"
        activations = {}
        gradients = {}

        def forward_hook(module, inp, out):
            activations['act'] = out.detach()

        def backward_hook(module, grad_in, grad_out):
            gradients['grad'] = grad_out[0].detach()

        # Register hooks
        target_layer = dict(model.named_modules())[target_layer_name]
        fh = target_layer.register_forward_hook(forward_hook)
        bh = target_layer.register_full_backward_hook(backward_hook)

        # Forward + backward
        outputs = model(image_tensor)
        score = outputs[0, predicted_class]
        score.backward()

        # Get activations and gradients
        act = activations['act'][0, 1:, :]  # Remove CLS token, shape: [num_patches, embed_dim]
        grad = gradients['grad'][0, 1:, :]   # Remove CLS token
        
        # Compute importance weights
        weights = grad.mean(dim=0)  # [embed_dim]
        cam = (act * weights).sum(dim=1)  # [num_patches]
        
        # Reshape to spatial dimensions
        num_patches = int(np.sqrt(cam.shape[0]))
        cam = cam.reshape(num_patches, num_patches).numpy()
        
        # Resize to image size
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        cam = np.maximum(cam, 0)  # ReLU
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # Create overlay
        img = transforms.ToPILImage()(image_tensor.squeeze(0).cpu())
        img = np.array(img)
        
        # Denormalize image (ViT uses mean=0.5, std=0.5)
        img = (img * 0.5 + 0.5) * 255
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

        fh.remove()
        bh.remove()
        return Image.fromarray(overlay)
    except Exception as e:
        st.warning(f"Grad-CAM generation failed: {e}")
        return None

# -------------------------
# Image Preprocessing & Prediction
# -------------------------
def get_transform():
    """Transform for BEiT v2 (uses ViT normalization: mean=0.5, std=0.5)"""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # ViT standard
    ])

def preprocess_image(image):
    transform = get_transform()
    return transform(image.convert('RGB')).unsqueeze(0)

def predict_with_tta(model, image_tensor, use_tta=True):
    """Prediction with optional Test-Time Augmentation"""
    model.eval()
    with torch.no_grad():
        if not use_tta:
            outputs = model(image_tensor)
            return F.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
        
        # TTA: Average predictions across augmentations
        preds = []
        
        # Original
        preds.append(F.softmax(model(image_tensor), dim=1))
        
        # Horizontal flip
        preds.append(F.softmax(model(torch.flip(image_tensor, [3])), dim=1))
        
        # Vertical flip
        preds.append(F.softmax(model(torch.flip(image_tensor, [2])), dim=1))
        
        # Average all predictions
        avg_pred = torch.stack(preds).mean(dim=0).squeeze(0).cpu().numpy()
        return avg_pred

def get_confidence_level(confidence):
    """Determine confidence level for clinical interpretation"""
    if confidence >= 0.85:
        return "High", "high-confidence"
    elif confidence >= 0.70:
        return "Medium", "medium-confidence"
    else:
        return "Low", "low-confidence"

# -------------------------
# Visualization
# -------------------------
def create_probability_chart(probs, class_names):
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices] * 100
    sorted_names = [CLASS_INFO[class_names[i]]['full_name'] for i in sorted_indices]
    colors = [CLASS_INFO[class_names[i]]['color'] for i in sorted_indices]
    
    fig = go.Figure(go.Bar(
        x=sorted_probs,
        y=sorted_names,
        orientation='h',
        marker=dict(color=colors),
        text=[f'{p:.1f}%' for p in sorted_probs],
        textposition='outside'
    ))
    fig.update_layout(
        title="Classification Probabilities",
        xaxis_title="Confidence (%)",
        yaxis_title="Lesion Type",
        height=400,
        plot_bgcolor='rgba(30, 30, 30, 0.8)',
        paper_bgcolor='rgba(18, 18, 18, 0.1)',
        font=dict(color='#F0F2F6'),
        xaxis=dict(range=[0, 105])
    )
    return fig
    
def create_risk_indicator(top_class: str, confidence: float):
    """Create risk indicator with confidence level"""
    risk = CLASS_INFO[top_class]['risk']
    conf_level, conf_class = get_confidence_level(confidence)
    
    risk_colors = {
        'Low': '#4CAF50', 
        'Medium': '#FFC107',
        'High': '#FF5722',
        'Critical': '#F44336'
    }
    color = risk_colors.get(risk, '#808080')
    
    html = f"""
    <div style="padding: 20px; border-radius: 10px; background-color: {color}; color: white; text-align: center; margin-bottom: 20px;">
        <h2 style="margin: 0; color: white !important;">Risk Level: {risk}</h2>
        <p style="margin: 5px 0; color: white !important;">Model Confidence: <span class="confidence-badge {conf_class}">{conf_level}</span></p>
    </div>
    """
    return html, risk, conf_level

# -------------------------
# Streamlit UI
# -------------------------
def main():
    st.set_page_config(
        page_title="DermScan AI - Skin Lesion Analyzer",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    set_theme()

    st.markdown("""
        # 🔬 DermScan AI - Dermoscopic Image Analyzer
        <p style='font-size: 18px; color: #aaa; margin-top: -10px;'>
        State-of-the-Art 8-Class Classification | BEiT v2 Large (Vision Transformer) trained on 44,000+ images | 
        Macro F1 89.2% | Macro AUC 97.4% | High-Confidence F1 94.2%
        </p>
        <hr>
        """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("ℹ️ Information")
        st.divider()
        st.warning("⚠️ **Medical Disclaimer**\n\nThis tool is for educational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional.")
        st.divider()
        
        st.header("⚙️ Settings")
        use_tta = st.checkbox("Use Test-Time Augmentation", value=True, 
                              help="Averages predictions across multiple image orientations for more robust results")
        show_all_probabilities = st.checkbox("Show detailed probability chart", value=True)
        show_gradcam = st.checkbox("Show AI Attention Map", value=True,
                                   help="Visualizes which regions the AI focuses on for its decision")
        st.divider()
        
        st.header("📊 Model Performance")
        st.markdown("**Overall Metrics:**")
        st.metric("Macro F1 Score", "89.2%")
        st.metric("Macro AUC-ROC", "97.4%")
        st.metric("Balanced Accuracy", "86.3%")
        
        st.markdown("**High-Confidence Predictions (≥85%):**")
        st.metric("F1 Score", "94.2%")
        st.metric("Balanced Accuracy", "92.2%")
        st.metric("Rejection Rate", "10.8%")
        
        st.divider()
        st.markdown("**Training Dataset:**")
        st.text("• ISIC2019: 25,331 images")
        st.text("• BCN20000: 19,424 images")
        st.text("• BOSQUE: 146 images")
        st.text("• Derm7pt: 828 images")
        st.text("Total: ~44,000 images")
        
        st.divider()
        st.markdown("**Key Features:**")
        st.text("✓ Patient-level data splitting")
        st.text("✓ Multi-dataset validation")
        st.text("✓ Confidence thresholding")
        st.text("✓ Vision Transformer architecture")

    model = load_model()
    if model is None:
        st.error("Failed to load model. Please refresh the page.")
        return

    st.subheader("📤 Upload Dermoscopic Image")
    uploaded_file = st.file_uploader("Choose a dermoscopic image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        try:
            col1, col2 = st.columns([1, 1])
            image = Image.open(uploaded_file)

            with col1:
                st.subheader("Uploaded Image")
                st.image(image, use_container_width=True)
                st.caption(f"Image size: {image.size[0]} x {image.size[1]} pixels")

                # Generate heatmap
                if show_gradcam:
                    with st.spinner("Generating AI attention map..."):
                        tensor = preprocess_image(image)
                        tensor.requires_grad = True
                        top_idx = np.argmax(predict_with_tta(model, tensor, use_tta=False))
                        gradcam_img = generate_gradcam_vit(model, tensor, top_idx)
                    
                    if gradcam_img:
                        buffered = BytesIO()
                        gradcam_img.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        
                        st.markdown(f"""
                        <div class="gradcam-container">
                            <img src="data:image/png;base64,{img_str}" 
                                 style="width:100%; height:auto; display:block; border-radius: 8px;"
                                 alt="Dermoscopic image with AI attention map">
                            <div class="gradcam-overlay">
                                <div class="gradcam-reveal"></div>
                            </div>
                            <div class="gradcam-caption">
                                AI Attention Map (shows where the model focuses)
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("AI attention map unavailable for this image.")

            with col2:
                st.subheader("Classification Results")
                with st.spinner("Analyzing image with Vision Transformer..."):
                    tensor = preprocess_image(image)
                    probs = predict_with_tta(model, tensor, use_tta=use_tta)
                
                top_idx = np.argmax(probs)
                top_class = CLASS_NAMES[top_idx]
                top_confidence = probs[top_idx]

                risk_html, risk_level, conf_level = create_risk_indicator(top_class, top_confidence)
                st.markdown(risk_html, unsafe_allow_html=True)
                
                # Confidence warning
                if conf_level == "Low":
                    st.error("⚠️ **Low Confidence Prediction** - This result should be interpreted with caution. Expert review strongly recommended.")
                elif conf_level == "Medium":
                    st.warning("⚡ **Medium Confidence** - Consider seeking expert dermatology opinion for confirmation.")
                
                # Urgent action for high-risk lesions
                if risk_level in ['High', 'Critical'] and conf_level in ['Medium', 'High']:
                    st.error("🚨 **Seek urgent Dermatology opinion**", icon="⚠️")
                
                st.markdown("---")
                st.markdown(f"### **Predicted Diagnosis:**\n## {CLASS_INFO[top_class]['full_name']}")
                st.markdown(f"**Confidence:** <span style='font-size: 1.2em; color: #00FF7F;'>{top_confidence*100:.1f}%</span>", unsafe_allow_html=True)
                st.progress(float(top_confidence))
                st.markdown("---")
                st.markdown(f"**Description:** {CLASS_INFO[top_class]['description']}")

            if show_all_probabilities:
                st.subheader("📊 Detailed Probability Distribution")
                st.plotly_chart(create_probability_chart(probs, CLASS_NAMES), use_container_width=True)

            st.subheader("🩺 Clinical Recommendations")
            
            # Adjust recommendations based on confidence level
            if conf_level == "Low":
                st.error("""**⚠️ LOW CONFIDENCE RESULT**
                
**This prediction has low confidence and should NOT be used for clinical decisions.**

**Recommended Actions:**
- **Immediate dermatologist consultation required**
- Do not rely on this result for diagnosis
- Consider the image quality or lesion characteristics may be ambiguous
- Expert human evaluation is essential""")
            elif risk_level in ['Critical', 'High']:
                st.error(f"""**⚠️ URGENT: This lesion shows characteristics of {CLASS_INFO[top_class]['full_name']}**

**Recommended Actions:**
- Schedule an appointment with a **dermatologist immediately**
- Do not delay - early detection is crucial
- Bring this analysis to your appointment
- Avoid sun exposure until evaluated
- Monitor for any rapid changes""")
            elif risk_level == 'Medium':
                st.warning(f"""**⚡ This lesion shows characteristics of {CLASS_INFO[top_class]['full_name']}**

**Recommended Actions:**
- Schedule a dermatologist appointment within **1-2 weeks**
- Monitor for any changes in size, color, or shape
- Protect from sun exposure
- Document with photos for comparison""")
            else:
                st.info(f"""**✓ This lesion appears to be {CLASS_INFO[top_class]['full_name']}**

**Recommended Actions:**
- Continue regular skin monitoring
- Annual dermatology check-ups recommended
- Report any changes to your doctor
- Practice sun safety (SPF 30+, protective clothing)""")

            st.subheader("🔍 Top 3 Predictions")
            top3 = np.argsort(probs)[::-1][:3]
            cols = st.columns(3)
            for i, idx in enumerate(top3):
                name = CLASS_NAMES[idx]
                with cols[i]:
                    st.markdown(f"""
                    <div style="padding: 15px; border-radius: 10px; border: 2px solid {CLASS_INFO[name]['color']};">
                        <h4>#{i+1}: {CLASS_INFO[name]['full_name']}</h4>
                        <p><strong>Confidence:</strong> {probs[idx]*100:.1f}%</p>
                        <p><strong>Risk:</strong> {CLASS_INFO[name]['risk']}</p>
                    </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"⚠️ An error occurred while processing the image: {e}")
            st.info("Please try uploading a different image or contact support if the problem persists.")
    else:
        st.info("""👆 **Please upload a dermoscopic image to begin analysis**

**Tips for best results:**
- Use high-quality dermoscopic images with good lighting and focus
- Ensure the lesion occupies most of the frame
- Avoid blurry or poorly lit images
- Not validated for subungual (under nail) or mucosal lesions""")

    st.subheader("📸 What is a dermoscopic image?")
    st.markdown("""Dermoscopic images are captured using a **dermatoscope**, a specialized medical device that:
- Uses 10-100x magnification
- Employs polarized or non-polarized light
- Reveals subsurface skin structures invisible to the naked eye
- Enables detection of early melanoma and other skin conditions""")
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #999; padding: 20px;">
        <p><strong>Model:</strong> BEiT v2 Large (Vision Transformer) | 304M parameters</p>
        <p><strong>Training:</strong> 44,000+ images from 4 datasets (ISIC2019, BCN20000, BOSQUE, Derm7pt) | Patient-level splitting</p>
        <p><strong>Performance:</strong> 97.4% Macro AUC | 89.2% F1 | 94.2% F1 at high confidence (≥85%)</p>
        <p><strong>Architecture:</strong> 24 transformer layers | 16x16 patches | 384px input resolution</p>
        <p><strong>Developed by:</strong> Dr Tom Hutchinson, Oxford, England | For educational and research purposes only</p>
    </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
