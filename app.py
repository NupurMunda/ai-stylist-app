import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import pickle
import os
import requests
from io import BytesIO
import base64
from sklearn.metrics.pairwise import cosine_similarity
import torch
import clip
from rembg import remove
import random
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# Configuration
STYLE_VECTOR_PATH = "style_vector.pkl"
REFERENCE_IMAGES_DIR = "reference_images"
STICKERS_DIR = "stickers"
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")  # Add your OpenAI API key to Streamlit secrets

# Create directories if they don't exist
os.makedirs(REFERENCE_IMAGES_DIR, exist_ok=True)
os.makedirs(STICKERS_DIR, exist_ok=True)

@st.cache_resource
def load_clip_model():
    """Load CLIP model for image embedding extraction"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

class AIStyler:
    def __init__(self):
        self.model, self.preprocess, self.device = load_clip_model()
        self.style_vector = None
        self.reference_embeddings = []
        
    def extract_image_embedding(self, image: Image.Image) -> np.ndarray:
        """Extract CLIP embedding from an image"""
        # Preprocess image for CLIP
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model.encode_image(image_input)
            embedding = embedding.cpu().numpy().flatten()
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
        return embedding
    
    def save_style_vector(self, embeddings: List[np.ndarray]):
        """Save style vector computed from reference embeddings"""
        if embeddings:
            # Average all embeddings to create style vector
            style_vector = np.mean(embeddings, axis=0)
            style_vector = style_vector / np.linalg.norm(style_vector)  # Normalize
            
            # Save both the style vector and individual embeddings
            style_data = {
                'style_vector': style_vector,
                'reference_embeddings': embeddings
            }
            
            with open(STYLE_VECTOR_PATH, 'wb') as f:
                pickle.dump(style_data, f)
            
            self.style_vector = style_vector
            self.reference_embeddings = embeddings
            return True
        return False
    
    def load_style_vector(self) -> bool:
        """Load existing style vector"""
        if os.path.exists(STYLE_VECTOR_PATH):
            try:
                with open(STYLE_VECTOR_PATH, 'rb') as f:
                    style_data = pickle.load(f)
                    self.style_vector = style_data['style_vector']
                    self.reference_embeddings = style_data.get('reference_embeddings', [])
                return True
            except:
                return False
        return False
    
    def calculate_similarity(self, image_embedding: np.ndarray) -> float:
        """Calculate cosine similarity between image and style vector"""
        if self.style_vector is None:
            return 0.0
        
        similarity = cosine_similarity([image_embedding], [self.style_vector])[0][0]
        return max(0, similarity)  # Ensure non-negative
    
    def update_style_with_feedback(self, image_embedding: np.ndarray, liked: bool):
        """Update style vector based on user feedback"""
        if liked and image_embedding is not None:
            # Add this embedding to our reference set
            self.reference_embeddings.append(image_embedding)
            
            # Recompute style vector
            self.save_style_vector(self.reference_embeddings)
            return True
        return False

def apply_warm_filter(image: Image.Image, intensity: float = 0.3) -> Image.Image:
    """Apply warm filter with adjustable intensity"""
    # Convert to array for OpenCV processing
    img_array = np.array(image)
    
    # Create warm filter effect
    warm_filter = np.zeros_like(img_array, dtype=np.float32)
    warm_filter[:, :, 0] = intensity * 50   # Add red
    warm_filter[:, :, 1] = intensity * 30   # Add green
    warm_filter[:, :, 2] = intensity * -10  # Reduce blue slightly
    
    # Apply filter
    result = img_array.astype(np.float32) + warm_filter
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return Image.fromarray(result)

def apply_vintage_filter(image: Image.Image, intensity: float = 0.3) -> Image.Image:
    """Apply vintage filter effect"""
    # Reduce saturation
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(1 - intensity * 0.3)
    
    # Add sepia tone
    img_array = np.array(image).astype(np.float32)
    sepia_filter = np.array([
        [0.393 + 0.607 * (1 - intensity), 0.769 - 0.769 * (1 - intensity), 0.189 - 0.189 * (1 - intensity)],
        [0.349 - 0.349 * (1 - intensity), 0.686 + 0.314 * (1 - intensity), 0.168 - 0.168 * (1 - intensity)],
        [0.272 - 0.272 * (1 - intensity), 0.534 - 0.534 * (1 - intensity), 0.131 + 0.869 * (1 - intensity)]
    ])
    
    result = img_array.dot(sepia_filter.T)
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return Image.fromarray(result)

def apply_dreamy_filter(image: Image.Image, intensity: float = 0.3) -> Image.Image:
    """Apply dreamy/soft filter effect"""
    # Apply gaussian blur
    blurred = image.filter(ImageFilter.GaussianBlur(radius=2 * intensity))
    
    # Blend original with blurred
    result = Image.blend(image, blurred, intensity * 0.5)
    
    # Enhance brightness slightly
    enhancer = ImageEnhance.Brightness(result)
    result = enhancer.enhance(1 + intensity * 0.2)
    
    return result

def apply_light_filter(image: Image.Image, intensity: float = 0.3) -> Image.Image:
    """Apply light/bright filter effect"""
    # Increase brightness and slight contrast
    enhancer = ImageEnhance.Brightness(image)
    result = enhancer.enhance(1 + intensity * 0.4)
    
    enhancer = ImageEnhance.Contrast(result)
    result = enhancer.enhance(1 + intensity * 0.2)
    
    return result

def create_doodle_outline(image: Image.Image, strength: float = 0.5, outline_type: str = "full") -> Image.Image:
    """Create doodle-style outline overlay"""
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # Apply edge detection with adjustable parameters based on strength
    low_threshold = int(50 + (1 - strength) * 50)
    high_threshold = int(150 + (1 - strength) * 100)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    
    # Create outline image
    outline_img = Image.fromarray(edges, mode='L')
    outline_rgba = Image.new('RGBA', image.size, (0, 0, 0, 0))
    
    # Draw outline with transparency
    for y in range(outline_img.height):
        for x in range(outline_img.width):
            if outline_img.getpixel((x, y)) > 128:  # Edge pixel
                alpha = int(255 * strength)
                outline_rgba.putpixel((x, y), (0, 0, 0, alpha))
    
    # Convert original to RGBA and composite
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    result = Image.alpha_composite(image, outline_rgba)
    return result.convert('RGB')

def remove_background(image: Image.Image) -> Image.Image:
    """Remove background from image using rembg"""
    try:
        # Convert PIL image to bytes
        img_bytes = BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()
        
        # Remove background
        result_bytes = remove(img_bytes)
        
        # Convert back to PIL Image
        result_image = Image.open(BytesIO(result_bytes))
        return result_image
    except Exception as e:
        st.error(f"Background removal failed: {str(e)}")
        return image

def generate_ai_sticker(prompt: str, api_key: str) -> Optional[Image.Image]:
    """Generate sticker using OpenAI DALL-E API"""
    if not api_key:
        st.warning("OpenAI API key not configured. Cannot generate AI stickers.")
        return None
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "dall-e-3",
            "prompt": f"{prompt}, transparent background, sticker style, high quality",
            "n": 1,
            "size": "1024x1024",
            "response_format": "url"
        }
        
        response = requests.post(
            "https://api.openai.com/v1/images/generations",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            image_url = result["data"][0]["url"]
            
            # Download and process image
            img_response = requests.get(image_url)
            if img_response.status_code == 200:
                image = Image.open(BytesIO(img_response.content))
                
                # Remove background to ensure transparency
                processed_image = remove_background(image)
                
                # Save to stickers directory
                filename = f"ai_sticker_{len(os.listdir(STICKERS_DIR))}.png"
                filepath = os.path.join(STICKERS_DIR, filename)
                processed_image.save(filepath, "PNG")
                
                return processed_image
        
        st.error("Failed to generate AI sticker")
        return None
        
    except Exception as e:
        st.error(f"AI sticker generation error: {str(e)}")
        return None

def load_stickers() -> List[Tuple[str, Image.Image]]:
    """Load all available stickers"""
    stickers = []
    if os.path.exists(STICKERS_DIR):
        for filename in os.listdir(STICKERS_DIR):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    filepath = os.path.join(STICKERS_DIR, filename)
                    sticker = Image.open(filepath)
                    if sticker.mode != 'RGBA':
                        sticker = sticker.convert('RGBA')
                    stickers.append((filename, sticker))
                except:
                    continue
    return stickers

def apply_sticker(base_image: Image.Image, sticker: Image.Image, position: Tuple[int, int], size: Tuple[int, int]) -> Image.Image:
    """Apply sticker to base image at specified position and size"""
    # Resize sticker
    sticker_resized = sticker.resize(size, Image.Resampling.LANCZOS)
    
    # Convert base image to RGBA if needed
    if base_image.mode != 'RGBA':
        base_image = base_image.convert('RGBA')
    
    # Create a copy to work with
    result = base_image.copy()
    
    # Paste sticker
    if sticker_resized.mode == 'RGBA':
        result.paste(sticker_resized, position, sticker_resized)
    else:
        result.paste(sticker_resized, position)
    
    return result.convert('RGB')

def main():
    st.set_page_config(
        page_title="AI Stylist - Learning Image Editor",
        page_icon="🎨",
        layout="wide"
    )
    
    st.title("🎨 AI Stylist - Learning Image Editor")
    st.markdown("*An AI-powered image editor that learns your aesthetic preferences over time*")
    
    # Initialize AI Styler
    if 'ai_styler' not in st.session_state:
        st.session_state.ai_styler = AIStyler()
        st.session_state.current_image_embedding = None
        st.session_state.processed_image = None
    
    ai_styler = st.session_state.ai_styler
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Check if style vector exists
        style_exists = ai_styler.load_style_vector()
        
        if style_exists:
            st.success(f"✅ Style learned from {len(ai_styler.reference_embeddings)} reference images")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Update Style"):
                    st.session_state.update_style = True
            with col2:
                if st.button("🗑️ Reset Style"):
                    if os.path.exists(STYLE_VECTOR_PATH):
                        os.remove(STYLE_VECTOR_PATH)
                    ai_styler.style_vector = None
                    ai_styler.reference_embeddings = []
                    st.rerun()
        else:
            st.warning("📚 No style learned yet. Upload reference images to get started!")
            st.session_state.update_style = True
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📸 Upload & Configure")
        
        # Handle style learning/updating
        if not style_exists or st.session_state.get('update_style', False):
            st.subheader("🎭 Learn Your Aesthetic Style")
            st.markdown("Upload 3+ reference images that represent your preferred aesthetic:")
            
            reference_files = st.file_uploader(
                "Choose reference images",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True,
                key="reference_uploader"
            )
            
            if reference_files and len(reference_files) >= 3:
                if st.button("🧠 Learn Style from References"):
                    with st.spinner("Analyzing your aesthetic preferences..."):
                        embeddings = []
                        for i, ref_file in enumerate(reference_files):
                            try:
                                ref_image = Image.open(ref_file).convert('RGB')
                                embedding = ai_styler.extract_image_embedding(ref_image)
                                embeddings.append(embedding)
                                
                                # Save reference image
                                ref_path = os.path.join(REFERENCE_IMAGES_DIR, f"ref_{i}_{ref_file.name}")
                                ref_image.save(ref_path)
                                
                            except Exception as e:
                                st.error(f"Error processing {ref_file.name}: {str(e)}")
                        
                        if embeddings:
                            ai_styler.save_style_vector(embeddings)
                            st.success(f"✅ Style learned from {len(embeddings)} images!")
                            st.session_state.update_style = False
                            st.rerun()
            elif reference_files:
                st.warning(f"Please upload at least 3 reference images (uploaded: {len(reference_files)})")
        
        # Image upload for editing
        if ai_styler.style_vector is not None:
            st.subheader("🖼️ Upload Image to Edit")
            uploaded_file = st.file_uploader(
                "Choose an image to style",
                type=['png', 'jpg', 'jpeg'],
                key="main_uploader"
            )
            
            if uploaded_file:
                # Load and display original image
                original_image = Image.open(uploaded_file).convert('RGB')
                st.image(original_image, caption="Original Image", use_column_width=True)
                
                # Extract embedding and calculate similarity
                with st.spinner("Analyzing image..."):
                    image_embedding = ai_styler.extract_image_embedding(original_image)
                    similarity = ai_styler.calculate_similarity(image_embedding) * 100
                    st.session_state.current_image_embedding = image_embedding
                
                st.info(f"📊 Style Similarity: {similarity:.1f}%")
                
                # Editing controls
                st.subheader("🎛️ Editing Controls")
                
                # Filter selection
                filter_type = st.selectbox(
                    "Filter Type",
                    ["warm", "vintage", "dreamy", "light"],
                    help="Choose the type of filter to apply"
                )
                
                # Filter intensity based on similarity
                base_intensity = 0.3 + (similarity / 100) * 0.4  # 0.3 to 0.7 range
                filter_intensity = st.slider(
                    "Filter Intensity",
                    0.0, 1.0, base_intensity,
                    help="AI-suggested intensity based on your style"
                )
                
                # Outline controls
                outline_option = st.selectbox(
                    "Outline Application",
                    ["none", "full", "subject_only"],
                    help="Where to apply doodle outlines"
                )
                
                if outline_option != "none":
                    outline_strength = st.slider(
                        "Outline Strength",
                        0.1, 1.0, 0.3 + (similarity / 100) * 0.4
                    )
                
                # Process image
                if st.button("✨ Apply AI Styling"):
                    with st.spinner("Applying AI styling..."):
                        result_image = original_image.copy()
                        
                        # Apply selected filter
                        if filter_type == "warm":
                            result_image = apply_warm_filter(result_image, filter_intensity)
                        elif filter_type == "vintage":
                            result_image = apply_vintage_filter(result_image, filter_intensity)
                        elif filter_type == "dreamy":
                            result_image = apply_dreamy_filter(result_image, filter_intensity)
                        elif filter_type == "light":
                            result_image = apply_light_filter(result_image, filter_intensity)
                        
                        # Apply outline if requested
                        if outline_option == "full":
                            result_image = create_doodle_outline(result_image, outline_strength, "full")
                        elif outline_option == "subject_only":
                            # For subject-only, we'd ideally use object detection
                            # For now, apply to full image with reduced strength
                            result_image = create_doodle_outline(result_image, outline_strength * 0.7, "subject")
                        
                        st.session_state.processed_image = result_image
    
    with col2:
        st.header("🎨 Results & Stickers")
        
        # Display processed image
        if st.session_state.processed_image is not None:
            st.subheader("✨ Styled Image")
            st.image(st.session_state.processed_image, caption="AI Styled Result", use_column_width=True)
            
            # Feedback section
            st.subheader("👍 Feedback")
            col_like, col_dislike = st.columns(2)
            
            with col_like:
                if st.button("👍 Love it!", key="like_btn"):
                    if st.session_state.current_image_embedding is not None:
                        ai_styler.update_style_with_feedback(
                            st.session_state.current_image_embedding, 
                            True
                        )
                        st.success("Thanks! I've learned from this image to better match your style.")
            
            with col_dislike:
                if st.button("👎 Not quite", key="dislike_btn"):
                    st.info("Got it! I won't learn from this image.")
            
            # Download button
            if st.session_state.processed_image:
                buf = BytesIO()
                st.session_state.processed_image.save(buf, format="PNG")
                st.download_button(
                    label="💾 Download Styled Image",
                    data=buf.getvalue(),
                    file_name=f"ai_styled_{uploaded_file.name}",
                    mime="image/png"
                )
        
        # Sticker section
        st.subheader("🔖 Stickers")
        
        # AI Sticker Generation
        with st.expander("🤖 Generate AI Stickers"):
            sticker_prompts = [
                "aesthetic doodle of a flower",
                "hand-drawn yellow star outline in cute style",
                "warm-toned vintage sparkle sticker",
                "minimalist heart doodle",
                "boho-style moon and stars",
                "kawaii cloud with face"
            ]
            
            selected_prompt = st.selectbox("Choose prompt", sticker_prompts)
            custom_prompt = st.text_input("Or enter custom prompt:")
            
            final_prompt = custom_prompt if custom_prompt else selected_prompt
            
            if st.button("🎨 Generate AI Sticker"):
                if OPENAI_API_KEY:
                    with st.spinner("Generating AI sticker..."):
                        ai_sticker = generate_ai_sticker(final_prompt, OPENAI_API_KEY)
                        if ai_sticker:
                            st.success("AI sticker generated!")
                            st.image(ai_sticker, caption="Generated Sticker", width=200)
                else:
                    st.warning("OpenAI API key not configured")
        
        # Custom Sticker Upload
        with st.expander("📤 Upload Custom Sticker"):
            sticker_file = st.file_uploader(
                "Upload sticker (background will be removed)",
                type=['png', 'jpg', 'jpeg'],
                key="sticker_uploader"
            )
            
            if sticker_file:
                sticker_image = Image.open(sticker_file)
                
                if st.button("🗑️ Remove Background & Save"):
                    with st.spinner("Processing sticker..."):
                        processed_sticker = remove_background(sticker_image)
                        
                        # Save processed sticker
                        filename = f"custom_{len(os.listdir(STICKERS_DIR))}_{sticker_file.name}"
                        filepath = os.path.join(STICKERS_DIR, filename)
                        processed_sticker.save(filepath, "PNG")
                        
                        st.success("Sticker saved!")
                        st.image(processed_sticker, caption="Processed Sticker", width=200)
        
        # Available Stickers
        stickers = load_stickers()
        if stickers:
            st.subheader("📚 Available Stickers")
            
            # Display stickers in grid
            cols = st.columns(3)
            for i, (name, sticker) in enumerate(stickers[:9]):  # Show first 9
                with cols[i % 3]:
                    st.image(sticker, caption=name[:15], width=100)
                    
                    if st.session_state.processed_image and st.button(f"Add", key=f"add_sticker_{i}"):
                        # Simple sticker application (center position)
                        img_width, img_height = st.session_state.processed_image.size
                        sticker_size = (min(img_width, img_height) // 6, min(img_width, img_height) // 6)
                        position = (img_width // 2 - sticker_size[0] // 2, img_height // 2 - sticker_size[1] // 2)
                        
                        st.session_state.processed_image = apply_sticker(
                            st.session_state.processed_image,
                            sticker,
                            position,
                            sticker_size
                        )
                        st.rerun()

if __name__ == "__main__":
    main()