import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw
import io
import os
import json
from pathlib import Path
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import requests
from datetime import datetime
import hashlib

# NEW: Background removal imports
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

# NEW: CLIP imports (using Hugging Face instead of OpenAI CLIP)
try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="🤖 Learning AI Stylist",
    page_icon="🎨",
    layout="wide"
)

# Constants
STYLE_DATA_FILE = "aesthetic_style.pkl"
FEEDBACK_DATA_FILE = "feedback_history.json"
STICKERS_DIR = Path("stickers")
REFERENCE_IMAGES_DIR = Path("reference_images")

# NEW: Initialize directories
STICKERS_DIR.mkdir(exist_ok=True)
REFERENCE_IMAGES_DIR.mkdir(exist_ok=True)

# NEW: Load CLIP model (cached) - Using Hugging Face
@st.cache_resource
def load_clip_model():
    """Load CLIP model for image embeddings using Hugging Face"""
    if not CLIP_AVAILABLE:
        return None, None, None
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.to(device)
        return model, processor, device
    except Exception as e:
        st.error(f"❌ Error loading CLIP model: {e}")
        return None, None, None

# NEW: Style data management
class StyleLearner:
    def __init__(self):
        self.style_vector = None
        self.reference_embeddings = []
        self.feedback_history = []
        self.load_style_data()
    
    def load_style_data(self):
        """Load existing style data"""
        if os.path.exists(STYLE_DATA_FILE):
            try:
                with open(STYLE_DATA_FILE, 'rb') as f:
                    data = pickle.load(f)
                    self.style_vector = data.get('style_vector')
                    self.reference_embeddings = data.get('reference_embeddings', [])
            except Exception as e:
                st.warning(f"Could not load style data: {e}")
        
        if os.path.exists(FEEDBACK_DATA_FILE):
            try:
                with open(FEEDBACK_DATA_FILE, 'r') as f:
                    self.feedback_history = json.load(f)
            except Exception as e:
                st.warning(f"Could not load feedback history: {e}")
    
    def save_style_data(self):
        """Save style data to disk"""
        try:
            data = {
                'style_vector': self.style_vector,
                'reference_embeddings': self.reference_embeddings,
                'last_updated': datetime.now().isoformat()
            }
            with open(STYLE_DATA_FILE, 'wb') as f:
                pickle.dump(data, f)
            
            with open(FEEDBACK_DATA_FILE, 'w') as f:
                json.dump(self.feedback_history, f)
            return True
        except Exception as e:
            st.error(f"Error saving style data: {e}")
            return False
    
    def extract_embedding(self, image_pil, model, processor, device):
        """Extract CLIP embedding from PIL image using Hugging Face"""
        try:
            # Process image
            inputs = processor(images=image_pil, return_tensors="pt").to(device)
            
            # Get image embedding
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
                # Normalize the features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy().flatten()
        except Exception as e:
            st.error(f"Error extracting embedding: {e}")
            return None
    
    def learn_from_references(self, reference_images, model, processor, device):
        """Learn aesthetic style from reference images"""
        embeddings = []
        
        for img_file in reference_images:
            try:
                img = Image.open(img_file).convert('RGB')
                embedding = self.extract_embedding(img, model, processor, device)
                if embedding is not None:
                    embeddings.append(embedding)
                    
                    # NEW: Save reference image for future use
                    img_hash = hashlib.md5(img_file.getvalue()).hexdigest()[:8]
                    ref_path = REFERENCE_IMAGES_DIR / f"ref_{img_hash}.jpg"
                    img.save(ref_path, 'JPEG', quality=85)
                    
            except Exception as e:
                st.warning(f"Skipping invalid reference image: {e}")
        
        if embeddings:
            self.reference_embeddings = embeddings
            self.style_vector = np.mean(embeddings, axis=0)
            self.save_style_data()
            return len(embeddings)
        return 0
    
    def update_from_feedback(self, image_pil, model, processor, device, liked=True):
        """Update style based on user feedback"""
        if not liked:
            return False
        
        embedding = self.extract_embedding(image_pil, model, processor, device)
        if embedding is not None:
            # Add to reference embeddings
            self.reference_embeddings.append(embedding)
            
            # Recalculate style vector
            self.style_vector = np.mean(self.reference_embeddings, axis=0)
            
            # Record feedback
            feedback_entry = {
                'timestamp': datetime.now().isoformat(),
                'liked': liked,
                'embedding_count': len(self.reference_embeddings)
            }
            self.feedback_history.append(feedback_entry)
            
            self.save_style_data()
            return True
        return False
    
    def get_style_similarity(self, image_pil, model, processor, device):
        """Get similarity score between image and learned style"""
        if self.style_vector is None:
            return 0.5  # Default medium similarity
        
        embedding = self.extract_embedding(image_pil, model, processor, device)
        if embedding is not None:
            similarity = cosine_similarity([embedding], [self.style_vector])[0][0]
            return max(0, min(1, (similarity + 1) / 2))  # Normalize to 0-1
        return 0.5

# NEW: AI Sticker Generation
def generate_ai_sticker(prompt, api_key=None):
    """Generate sticker using DALL-E API (placeholder for now)"""
    # NOTE: This is a placeholder. In production, you'd use:
    # - OpenAI DALL-E API
    # - Stability AI
    # - Local models like Stable Diffusion
    
    st.info("🤖 AI sticker generation coming soon! For now, upload your own stickers.")
    return None

# NEW: Background removal for user stickers
def remove_background(image_pil):
    """Remove background from uploaded sticker"""
    if not REMBG_AVAILABLE:
        st.warning("Background removal not available. Using image as-is.")
        return image_pil
    
    try:
        # Convert PIL to bytes
        img_bytes = io.BytesIO()
        image_pil.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Remove background
        output_bytes = remove(img_bytes.getvalue())
        
        # Convert back to PIL
        output_image = Image.open(io.BytesIO(output_bytes)).convert('RGBA')
        return output_image
    except Exception as e:
        st.error(f"Background removal failed: {e}")
        return image_pil.convert('RGBA')

# Enhanced image processing functions
def apply_smart_warm_filter(image_cv, similarity_score):
    """Apply warm filter with intensity based on style similarity"""
    # Map similarity to filter intensity
    if similarity_score > 0.8:
        intensity = 1.2  # Strong warm filter
        brightness = 25
    elif similarity_score > 0.6:
        intensity = 0.9  # Medium warm filter
        brightness = 15
    else:
        intensity = 0.6  # Light warm filter
        brightness = 8
    
    img_float = image_cv.astype(np.float32) / 255.0
    
    warm_matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.1 * intensity, 1.0 + 0.1 * intensity, 0.0],
        [0.0, 0.0, 1.0 - 0.2 * intensity]
    ])
    
    warm_img = cv2.transform(img_float, warm_matrix)
    return cv2.convertScaleAbs(warm_img * 255, alpha=1.0 + intensity * 0.1, beta=brightness)

def apply_smart_outline(image_cv, similarity_score):
    """Apply outline with thickness based on style similarity"""
    if similarity_score > 0.8:
        thickness, blur = 3, 5  # Bold outlines
    elif similarity_score > 0.6:
        thickness, blur = 2, 3  # Medium outlines
    else:
        thickness, blur = 1, 2  # Light outlines
    
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur*2+1, blur*2+1), 0)
    
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((thickness, thickness), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def get_available_stickers():
    """Get list of available sticker files"""
    sticker_files = []
    for ext in ['*.png', '*.PNG']:
        sticker_files.extend(STICKERS_DIR.glob(ext))
    return sticker_files

def cv2_to_pil(cv2_image):
    """Convert OpenCV image to PIL"""
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)

def pil_to_cv2(pil_image):
    """Convert PIL image to OpenCV"""
    rgb_array = np.array(pil_image.convert('RGB'))
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

def overlay_smart_stickers(base_image_pil, similarity_score):
    """Overlay stickers based on style similarity"""
    available_stickers = get_available_stickers()
    if not available_stickers:
        return base_image_pil
    
    img_width, img_height = base_image_pil.size
    result_image = base_image_pil.convert('RGBA')
    
    # Determine sticker style based on similarity
    if similarity_score > 0.8:
        sticker_count = min(3, len(available_stickers))  # Bold: more stickers
        opacity = 0.9
    elif similarity_score > 0.6:
        sticker_count = min(2, len(available_stickers))  # Medium: some stickers
        opacity = 0.8
    else:
        sticker_count = min(1, len(available_stickers))  # Minimal: few stickers
        opacity = 0.7
    
    # Add stickers
    for i in range(sticker_count):
        sticker_path = available_stickers[i % len(available_stickers)]
        try:
            sticker = Image.open(sticker_path).convert('RGBA')
            sticker = sticker.resize((80, 80), Image.Resampling.LANCZOS)
            
            # Apply opacity
            alpha = sticker.split()[-1]
            alpha = alpha.point(lambda p: int(p * opacity))
            sticker.putalpha(alpha)
            
            # Position stickers
            if i == 0:
                pos = (int(img_width * 0.8), int(img_height * 0.1))
            elif i == 1:
                pos = (int(img_width * 0.1), int(img_height * 0.85))
            else:
                pos = (int(img_width * 0.7), int(img_height * 0.6))
            
            result_image.paste(sticker, pos, sticker)
        except Exception as e:
            st.warning(f"Could not load sticker {sticker_path}: {e}")
    
    return result_image.convert('RGB')

def process_image_with_ai_style(uploaded_image, style_learner, model, processor, device):
    """Process image using learned AI style"""
    pil_image = Image.open(uploaded_image)
    cv_image = pil_to_cv2(pil_image)
    
    # Get style similarity
    similarity = style_learner.get_style_similarity(pil_image, model, processor, device)
    
    # Apply smart filters
    warm_image = apply_smart_warm_filter(cv_image, similarity)
    outline = apply_smart_outline(cv_image, similarity)
    
    # Blend warm image with outline
    outline_gray = cv2.cvtColor(outline, cv2.COLOR_BGR2GRAY)
    outline_mask = outline_gray > 0
    
    final_cv = warm_image.copy()
    final_cv[outline_mask] = final_cv[outline_mask] * 0.7
    
    final_pil = cv2_to_pil(final_cv)
    
    # Add smart stickers
    final_pil = overlay_smart_stickers(final_pil, similarity)
    
    return final_pil, similarity

def main():
    st.title("🤖 Learning AI Stylist")
    st.markdown("*An AI that learns your aesthetic and gets better over time*")
    st.markdown("---")
    
    # Initialize components
    model, processor, device = load_clip_model()
    if model is None:
        st.error("❌ Could not load CLIP model. Please check your internet connection.")
        st.info("💡 The app will work with basic functionality, but AI learning features will be disabled.")
        # You can still provide basic image processing here
        st.stop()
    
    # NEW: Initialize style learner
    style_learner = StyleLearner()
    
    # Sidebar for AI learning controls
    with st.sidebar:
        st.header("🧠 AI Style Learning")
        
        # Show current style status
        if style_learner.style_vector is not None:
            ref_count = len(style_learner.reference_embeddings)
            feedback_count = len(style_learner.feedback_history)
            likes = sum(1 for f in style_learner.feedback_history if f.get('liked', False))
            
            st.success("✅ AI Style Learned!")
            st.metric("Reference Images", ref_count)
            st.metric("Feedback Given", feedback_count)
            st.metric("Liked Images", likes)
            
            if st.button("🗑️ Reset Style Learning"):
                if os.path.exists(STYLE_DATA_FILE):
                    os.remove(STYLE_DATA_FILE)
                if os.path.exists(FEEDBACK_DATA_FILE):
                    os.remove(FEEDBACK_DATA_FILE)
                st.success("Style data reset! Refresh the page.")
                st.stop()
        else:
            st.warning("🤔 No style learned yet")
            st.info("👇 Upload reference images to start learning!")
        
        st.markdown("---")
        
        # NEW: Reference image learning
        st.subheader("📁 Style Learning")
        reference_images = st.file_uploader(
            "Upload 3-10 reference images (your aesthetic style)",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Upload images that represent your preferred editing style"
        )
        
        if reference_images and len(reference_images) >= 3:
            if st.button("🎯 Learn My Style"):
                with st.spinner("🧠 Learning your aesthetic preferences..."):
                    learned_count = style_learner.learn_from_references(
                        reference_images, model, processor, device
                    )
                    if learned_count > 0:
                        st.success(f"✅ Learned from {learned_count} images!")
                        st.rerun()
                    else:
                        st.error("❌ Could not learn from reference images")
        elif reference_images:
            st.warning(f"Upload at least 3 images (you have {len(reference_images)})")
        
        st.markdown("---")
        
        # NEW: Sticker management
        st.subheader("🎪 Sticker Management")
        
        # Show available stickers
        available_stickers = get_available_stickers()
        st.info(f"📁 {len(available_stickers)} stickers available")
        
        # User sticker upload with background removal
        user_sticker = st.file_uploader(
            "Upload custom sticker",
            type=['jpg', 'jpeg', 'png'],
            help="Upload any image - background will be removed automatically"
        )
        
        if user_sticker:
            if st.button("✂️ Process & Save Sticker"):
                with st.spinner("🔄 Removing background..."):
                    try:
                        img = Image.open(user_sticker)
                        processed_img = remove_background(img)
                        
                        # Save processed sticker
                        sticker_name = f"custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        sticker_path = STICKERS_DIR / sticker_name
                        processed_img.save(sticker_path)
                        
                        st.success(f"✅ Sticker saved as {sticker_name}")
                        st.image(processed_img, caption="Processed sticker", width=100)
                    except Exception as e:
                        st.error(f"Error processing sticker: {e}")
        
        # NEW: AI sticker generation (placeholder)
        if st.button("🤖 Generate AI Stickers"):
            generate_ai_sticker("cute kawaii sticker")
        
        st.markdown("---")
        
        # Main image upload
        uploaded_file = st.file_uploader(
            "📸 Upload photo to style",
            type=['jpg', 'jpeg', 'png']
        )
    
    # Main content area
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Original")
            original_image = Image.open(uploaded_file)
            st.image(original_image, use_column_width=True)
        
        with col2:
            st.subheader("🎨 AI Styled")
            
            if style_learner.style_vector is not None:
                # Process with learned style
                with st.spinner("🎨 Applying your learned aesthetic..."):
                    styled_image, similarity = process_image_with_ai_style(
                        uploaded_file, style_learner, model, processor, device
                    )
                    
                    st.image(styled_image, use_column_width=True)
                    
                    # Show style analysis
                    st.metric("Style Similarity", f"{similarity:.2%}")
                    
                    if similarity > 0.8:
                        st.success("🎯 Perfect match! Applied bold styling")
                    elif similarity > 0.6:
                        st.info("👍 Good match! Applied medium styling")
                    else:
                        st.warning("🤔 Different style. Applied light styling")
                    
                    # NEW: Feedback section
                    st.markdown("---")
                    st.write("**How do you like this result?**")
                    
                    feedback_col1, feedback_col2 = st.columns(2)
                    with feedback_col1:
                        if st.button("👍 Love it!", use_container_width=True):
                            if style_learner.update_from_feedback(
                                styled_image, model, processor, device, liked=True
                            ):
                                st.success("✅ Thanks! AI learned from your feedback")
                                st.rerun()
                    
                    with feedback_col2:
                        if st.button("👎 Not my style", use_container_width=True):
                            st.info("📝 Feedback noted. AI won't learn from this image.")
                    
                    # Download button
                    img_buffer = io.BytesIO()
                    styled_image.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    
                    st.download_button(
                        label="💾 Download Styled Image",
                        data=img_buffer,
                        file_name=f"ai_styled_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        use_container_width=True
                    )
            else:
                st.warning("🤔 No style learned yet. Upload reference images first!")
                st.info("👈 Use the sidebar to upload reference images and teach the AI your style")
    
    else:
        st.info("👆 Upload an image to see AI styling in action!")
    
    # Information section
    st.markdown("---")
    with st.expander("🤖 How the Learning AI Works"):
        st.markdown("""
        **This AI stylist learns and improves over time:**
        
        1. **🎯 Initial Learning**: Upload 3-10 reference images showing your preferred aesthetic
        2. **🧠 Style Embedding**: AI creates a "style vector" representing your preferences using CLIP
        3. **📊 Smart Analysis**: For each new image, AI measures similarity to your learned style
        4. **🎨 Adaptive Styling**: Applies filters, outlines, and stickers based on similarity:
           - High similarity (80%+): Bold, strong effects
           - Medium similarity (60-80%): Balanced styling  
           - Low similarity (<60%): Gentle, minimal effects
        5. **📈 Continuous Learning**: Your 👍/👎 feedback updates the AI's understanding
        6. **🎪 Smart Stickers**: Auto background removal for custom stickers
        
        **The more you use it, the better it gets at matching your style!**
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "🤖 Learning AI Stylist • Gets smarter with every interaction"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()