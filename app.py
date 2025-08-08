import streamlit as st
import cv2
import numpy as np
import pickle
import os
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import io
from typing import List, Dict, Tuple, Optional
import torch
from transformers import CLIPProcessor, CLIPModel
from rembg import remove
import json

# Initialize session state
if 'style_vector' not in st.session_state:
    st.session_state.style_vector = None
if 'style_name' not in st.session_state:
    st.session_state.style_name = None
if 'preview_image' not in st.session_state:
    st.session_state.preview_image = None

class StyleLearner:
    """Handles learning and storing user's aesthetic style preferences"""
    
    def __init__(self):
        self.model_name = "openai/clip-vit-base-patch32"
        self.processor = None
        self.model = None
        self.style_dir = "styles"
        os.makedirs(self.style_dir, exist_ok=True)
    
    @st.cache_resource
    def load_clip_model(_self):
        """Load CLIP model for image embeddings"""
        try:
            processor = CLIPProcessor.from_pretrained(_self.model_name)
            model = CLIPModel.from_pretrained(_self.model_name)
            return processor, model
        except Exception as e:
            st.error(f"Error loading CLIP model: {e}")
            return None, None
    
    def extract_image_embedding(self, image: Image.Image) -> np.ndarray:
        """Extract embedding from a single image"""
        if self.processor is None or self.model is None:
            self.processor, self.model = self.load_clip_model()
        
        if self.processor is None:
            return np.zeros(512)  # Fallback
        
        try:
            inputs = self.processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            return image_features.numpy().flatten()
        except Exception as e:
            st.error(f"Error extracting embedding: {e}")
            return np.zeros(512)
    
    def learn_style_from_images(self, images: List[Image.Image], style_name: str) -> np.ndarray:
        """Learn style vector from multiple reference images"""
        embeddings = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, img in enumerate(images):
            status_text.text(f"Processing image {i+1}/{len(images)}...")
            embedding = self.extract_image_embedding(img)
            embeddings.append(embedding)
            progress_bar.progress((i + 1) / len(images))
        
        # Average all embeddings to create style vector
        style_vector = np.mean(embeddings, axis=0)
        
        # Save style
        self.save_style(style_vector, style_name)
        
        status_text.text("Style learning completed!")
        progress_bar.empty()
        
        return style_vector
    
    def save_style(self, style_vector: np.ndarray, style_name: str):
        """Save learned style to disk"""
        style_path = os.path.join(self.style_dir, f"{style_name}.pkl")
        with open(style_path, 'wb') as f:
            pickle.dump(style_vector, f)
    
    def load_style(self, style_name: str) -> Optional[np.ndarray]:
        """Load saved style from disk"""
        style_path = os.path.join(self.style_dir, f"{style_name}.pkl")
        if os.path.exists(style_path):
            with open(style_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def get_available_styles(self) -> List[str]:
        """Get list of available saved styles"""
        if not os.path.exists(self.style_dir):
            return []
        return [f.replace('.pkl', '') for f in os.listdir(self.style_dir) if f.endswith('.pkl')]

class AestheticEditor:
    """Handles layered aesthetic editing based on style preferences"""
    
    def __init__(self):
        self.filters = {
            'warm': {'brightness': 1.1, 'contrast': 1.2, 'saturation': 1.3, 'warmth': True},
            'cool': {'brightness': 1.0, 'contrast': 1.1, 'saturation': 0.9, 'warmth': False},
            'vintage': {'brightness': 0.9, 'contrast': 1.3, 'saturation': 0.8, 'sepia': True},
            'vivid': {'brightness': 1.2, 'contrast': 1.4, 'saturation': 1.5, 'warmth': False},
            'soft': {'brightness': 1.1, 'contrast': 0.9, 'saturation': 1.1, 'blur': True},
            'dramatic': {'brightness': 0.8, 'contrast': 1.6, 'saturation': 1.2, 'warmth': False}
        }
        
        self.outline_styles = {
            'thin_black': {'color': (0, 0, 0), 'width': 2, 'style': 'solid'},
            'thick_white': {'color': (255, 255, 255), 'width': 4, 'style': 'solid'},
            'dotted_red': {'color': (255, 0, 0), 'width': 3, 'style': 'dotted'},
            'sketchy': {'color': (100, 100, 100), 'width': 2, 'style': 'sketchy'}
        }
        
        self.doodle_patterns = ['hearts', 'stars', 'flowers', 'geometric', 'minimal']
    
    def calculate_style_similarity(self, image_embedding: np.ndarray, style_vector: np.ndarray) -> float:
        """Calculate cosine similarity between image and style vector"""
        if style_vector is None or len(style_vector) == 0:
            return 0.5  # Default similarity
        
        # Normalize vectors
        norm_img = image_embedding / np.linalg.norm(image_embedding)
        norm_style = style_vector / np.linalg.norm(style_vector)
        
        # Calculate cosine similarity
        similarity = np.dot(norm_img, norm_style)
        return max(0, min(1, (similarity + 1) / 2))  # Normalize to 0-1
    
    def apply_filter_layer(self, image: Image.Image, filter_name: str, intensity: float = 1.0) -> Image.Image:
        """Apply aesthetic filter based on similarity score"""
        if filter_name not in self.filters:
            return image
        
        filter_settings = self.filters[filter_name]
        edited = image.copy()
        
        # Adjust intensity based on similarity
        intensity = max(0.3, min(1.0, intensity))
        
        # Apply brightness
        if 'brightness' in filter_settings:
            enhancer = ImageEnhance.Brightness(edited)
            factor = 1 + (filter_settings['brightness'] - 1) * intensity
            edited = enhancer.enhance(factor)
        
        # Apply contrast
        if 'contrast' in filter_settings:
            enhancer = ImageEnhance.Contrast(edited)
            factor = 1 + (filter_settings['contrast'] - 1) * intensity
            edited = enhancer.enhance(factor)
        
        # Apply saturation
        if 'saturation' in filter_settings:
            enhancer = ImageEnhance.Color(edited)
            factor = 1 + (filter_settings['saturation'] - 1) * intensity
            edited = enhancer.enhance(factor)
        
        # Apply special effects
        if filter_settings.get('sepia'):
            edited = self.apply_sepia(edited, intensity)
        
        if filter_settings.get('warmth'):
            edited = self.apply_warmth(edited, intensity)
        
        if filter_settings.get('blur'):
            edited = edited.filter(ImageFilter.GaussianBlur(radius=0.5 * intensity))
        
        return edited
    
    def apply_sepia(self, image: Image.Image, intensity: float) -> Image.Image:
        """Apply sepia tone effect"""
        pixels = np.array(image)
        
        # Sepia transformation matrix
        sepia_filter = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        
        sepia_img = pixels.dot(sepia_filter.T)
        sepia_img = np.clip(sepia_img, 0, 255)
        
        # Blend with original based on intensity
        blended = (1 - intensity) * pixels + intensity * sepia_img
        return Image.fromarray(blended.astype(np.uint8))
    
    def apply_warmth(self, image: Image.Image, intensity: float) -> Image.Image:
        """Apply warm color temperature"""
        pixels = np.array(image)
        
        # Increase red and yellow tones
        warm_pixels = pixels.copy()
        warm_pixels[:, :, 0] = np.clip(pixels[:, :, 0] * (1 + 0.1 * intensity), 0, 255)  # Red
        warm_pixels[:, :, 1] = np.clip(pixels[:, :, 1] * (1 + 0.05 * intensity), 0, 255)  # Green
        
        return Image.fromarray(warm_pixels.astype(np.uint8))
    
    def add_outline_layer(self, image: Image.Image, outline_style: str, intensity: float) -> Image.Image:
        """Add outline around detected subjects"""
        if outline_style not in self.outline_styles:
            return image
        
        style = self.outline_styles[outline_style]
        
        # Convert to OpenCV format for edge detection
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges based on width and intensity
        kernel_size = int(style['width'] * intensity)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Create outline overlay
        outline_overlay = np.zeros_like(cv_image)
        outline_overlay[edges > 0] = style['color'][::-1]  # BGR format
        
        # Convert back to PIL
        outline_pil = Image.fromarray(cv2.cvtColor(outline_overlay, cv2.COLOR_BGR2RGB))
        
        # Blend with original image
        result = Image.alpha_composite(
            image.convert('RGBA'),
            outline_pil.convert('RGBA')
        ).convert('RGB')
        
        return result
    
    def add_doodle_layer(self, image: Image.Image, pattern: str, intensity: float) -> Image.Image:
        """Add decorative doodles based on pattern and intensity"""
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        width, height = image.size
        num_doodles = int(5 * intensity)  # More doodles with higher intensity
        
        for _ in range(num_doodles):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            
            if pattern == 'hearts':
                self.draw_heart(draw, x, y, intensity)
            elif pattern == 'stars':
                self.draw_star(draw, x, y, intensity)
            elif pattern == 'flowers':
                self.draw_flower(draw, x, y, intensity)
            elif pattern == 'geometric':
                self.draw_geometric(draw, x, y, intensity)
            elif pattern == 'minimal':
                self.draw_minimal(draw, x, y, intensity)
        
        # Blend with original
        result = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
        return result
    
    def draw_heart(self, draw, x, y, intensity):
        """Draw a heart shape"""
        size = int(10 * intensity)
        color = (255, 192, 203, int(100 * intensity))  # Pink with alpha
        
        # Simple heart approximation using circles and triangle
        draw.ellipse([x-size//2, y-size//2, x, y], fill=color)
        draw.ellipse([x, y-size//2, x+size//2, y], fill=color)
        points = [(x-size//2, y), (x+size//2, y), (x, y+size//2)]
        draw.polygon(points, fill=color)
    
    def draw_star(self, draw, x, y, intensity):
        """Draw a star shape"""
        size = int(8 * intensity)
        color = (255, 255, 0, int(120 * intensity))  # Yellow with alpha
        
        points = []
        for i in range(10):
            angle = i * 36 * np.pi / 180
            if i % 2 == 0:
                r = size
            else:
                r = size // 2
            px = x + r * np.cos(angle)
            py = y + r * np.sin(angle)
            points.append((px, py))
        
        draw.polygon(points, fill=color)
    
    def draw_flower(self, draw, x, y, intensity):
        """Draw a simple flower"""
        size = int(6 * intensity)
        color = (255, 182, 193, int(90 * intensity))  # Light pink with alpha
        
        # Draw petals as small circles
        for angle in range(0, 360, 60):
            rad = angle * np.pi / 180
            px = x + size * np.cos(rad)
            py = y + size * np.sin(rad)
            draw.ellipse([px-size//3, py-size//3, px+size//3, py+size//3], fill=color)
        
        # Center
        center_color = (255, 255, 0, int(100 * intensity))  # Yellow center
        draw.ellipse([x-size//4, y-size//4, x+size//4, y+size//4], fill=center_color)
    
    def draw_geometric(self, draw, x, y, intensity):
        """Draw geometric shapes"""
        size = int(8 * intensity)
        colors = [
            (0, 255, 255, int(80 * intensity)),    # Cyan
            (255, 0, 255, int(80 * intensity)),    # Magenta
            (255, 255, 0, int(80 * intensity))     # Yellow
        ]
        
        shape_type = np.random.choice(['circle', 'triangle', 'square'])
        color = np.random.choice(colors)
        
        if shape_type == 'circle':
            draw.ellipse([x-size, y-size, x+size, y+size], fill=color)
        elif shape_type == 'triangle':
            points = [(x, y-size), (x-size, y+size), (x+size, y+size)]
            draw.polygon(points, fill=color)
        else:  # square
            draw.rectangle([x-size, y-size, x+size, y+size], fill=color)
    
    def draw_minimal(self, draw, x, y, intensity):
        """Draw minimal lines and dots"""
        size = int(15 * intensity)
        color = (100, 100, 100, int(60 * intensity))  # Gray with alpha
        
        if np.random.random() > 0.5:
            # Draw line
            x2 = x + np.random.randint(-size, size)
            y2 = y + np.random.randint(-size, size)
            draw.line([(x, y), (x2, y2)], fill=color, width=int(2 * intensity))
        else:
            # Draw dot
            dot_size = int(3 * intensity)
            draw.ellipse([x-dot_size, y-dot_size, x+dot_size, y+dot_size], fill=color)

def remove_background(image: Image.Image) -> Image.Image:
    """Remove background from image for stickers"""
    try:
        # Convert PIL to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Remove background
        result_bytes = remove(img_bytes.getvalue())
        
        # Convert back to PIL
        result_image = Image.open(io.BytesIO(result_bytes))
        return result_image
    except Exception as e:
        st.error(f"Error removing background: {e}")
        return image

def apply_stickers(base_image: Image.Image, stickers: List[Image.Image], intensity: float) -> Image.Image:
    """Apply stickers to the base image"""
    result = base_image.copy().convert('RGBA')
    
    num_stickers = max(1, int(len(stickers) * intensity))
    selected_stickers = np.random.choice(stickers, size=min(num_stickers, len(stickers)), replace=False)
    
    for sticker in selected_stickers:
        # Random position
        max_x = result.width - sticker.width
        max_y = result.height - sticker.height
        
        if max_x > 0 and max_y > 0:
            x = np.random.randint(0, max_x)
            y = np.random.randint(0, max_y)
            
            # Random size (50% to 100% of original)
            scale = 0.5 + 0.5 * intensity
            new_size = (int(sticker.width * scale), int(sticker.height * scale))
            scaled_sticker = sticker.resize(new_size, Image.Resampling.LANCZOS)
            
            # Apply sticker
            result.paste(scaled_sticker, (x, y), scaled_sticker)
    
    return result.convert('RGB')

def create_default_styles():
    """Create some default style presets"""
    default_styles = {
        'vintage_warm': {
            'filter': 'vintage',
            'outline': 'sketchy',
            'doodle': 'minimal',
            'description': 'Warm, vintage aesthetic with sketchy outlines'
        },
        'modern_vivid': {
            'filter': 'vivid',
            'outline': 'thin_black',
            'doodle': 'geometric',
            'description': 'Modern, vivid colors with geometric elements'
        },
        'soft_romantic': {
            'filter': 'soft',
            'outline': 'thick_white',
            'doodle': 'hearts',
            'description': 'Soft, romantic feel with heart doodles'
        },
        'dramatic_cool': {
            'filter': 'dramatic',
            'outline': 'dotted_red',
            'doodle': 'stars',
            'description': 'Dramatic contrast with cool tones'
        }
    }
    
    # Save default styles
    os.makedirs('styles', exist_ok=True)
    with open('styles/default_styles.json', 'w') as f:
        json.dump(default_styles, f, indent=2)
    
    return default_styles

def main():
    st.set_page_config(
        page_title="AI Photo Editor",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎨 AI-Powered Aesthetic Photo Editor")
    st.markdown("*Learn your style, edit with AI*")
    
    # Initialize components
    style_learner = StyleLearner()
    aesthetic_editor = AestheticEditor()
    
    # Create default styles if they don't exist
    if not os.path.exists('styles/default_styles.json'):
        default_styles = create_default_styles()
    else:
        with open('styles/default_styles.json', 'r') as f:
            default_styles = json.load(f)
    
    # Sidebar for style management
    with st.sidebar:
        st.header("🎯 Style Management")
        
        # Style selection
        style_option = st.radio(
            "Choose Style Option:",
            ["Use Default Style", "Load Saved Style", "Learn New Style"]
        )
        
        if style_option == "Use Default Style":
            selected_default = st.selectbox(
                "Select Default Style:",
                list(default_styles.keys()),
                format_func=lambda x: f"{x.replace('_', ' ').title()}"
            )
            
            if selected_default:
                st.info(default_styles[selected_default]['description'])
                st.session_state.style_name = selected_default
                # Create a dummy style vector for default styles
                st.session_state.style_vector = np.random.normal(0, 1, 512)
        
        elif style_option == "Load Saved Style":
            available_styles = style_learner.get_available_styles()
            if available_styles:
                selected_style = st.selectbox("Select Saved Style:", available_styles)
                if st.button("Load Style"):
                    st.session_state.style_vector = style_learner.load_style(selected_style)
                    st.session_state.style_name = selected_style
                    st.success(f"Loaded style: {selected_style}")
            else:
                st.info("No saved styles available. Learn a new style first!")
        
        elif style_option == "Learn New Style":
            st.subheader("📚 Learn From Reference Images")
            
            style_name = st.text_input("Style Name:", placeholder="e.g., my_aesthetic")
            
            uploaded_refs = st.file_uploader(
                "Upload 3-10 reference images:",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True
            )
            
            if uploaded_refs and len(uploaded_refs) >= 3 and style_name:
                if st.button("Learn Style"):
                    ref_images = []
                    for ref_file in uploaded_refs[:10]:  # Limit to 10 images
                        ref_image = Image.open(ref_file).convert('RGB')
                        ref_images.append(ref_image)
                    
                    # Learn style
                    style_vector = style_learner.learn_style_from_images(ref_images, style_name)
                    st.session_state.style_vector = style_vector
                    st.session_state.style_name = style_name
                    
                    st.success(f"Successfully learned style: {style_name}")
            
            elif uploaded_refs and len(uploaded_refs) < 3:
                st.warning("Please upload at least 3 reference images.")
    
    # Main editing interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📷 Upload Image to Edit")
        
        uploaded_file = st.file_uploader(
            "Choose an image to edit:",
            type=['png', 'jpg', 'jpeg']
        )
        
        if uploaded_file:
            original_image = Image.open(uploaded_file).convert('RGB')
            st.image(original_image, caption="Original Image", use_column_width=True)
            
            # Manual adjustment controls
            with st.expander("🎛️ Manual Adjustments", expanded=False):
                manual_intensity = st.slider("Overall Effect Intensity:", 0.0, 1.0, 0.8)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    custom_filter = st.selectbox(
                        "Override Filter:",
                        ['auto'] + list(aesthetic_editor.filters.keys())
                    )
                    custom_outline = st.selectbox(
                        "Override Outline:",
                        ['auto'] + list(aesthetic_editor.outline_styles.keys())
                    )
                
                with col_b:
                    custom_doodle = st.selectbox(
                        "Override Doodles:",
                        ['auto'] + aesthetic_editor.doodle_patterns
                    )
                    add_stickers = st.checkbox("Add Stickers", value=False)
                
                # Sticker upload
                if add_stickers:
                    sticker_files = st.file_uploader(
                        "Upload Stickers:",
                        type=['png', 'jpg', 'jpeg'],
                        accept_multiple_files=True
                    )
            
            # Preview and Edit buttons
            col_btn1, col_btn2, col_btn3 = st.columns(3)
            
            with col_btn1:
                if st.button("🔮 Preview AI Edit", use_container_width=True):
                    if st.session_state.style_vector is not None:
                    st.write(f"**Vector Size:** {len(st.session_state.style_vector)} dimensions")

if __name__ == "__main__":
    main() None:
                        with st.spinner("Creating AI preview..."):
                            # Get image embedding
                            img_embedding = style_learner.extract_image_embedding(original_image)
                            
                            # Calculate similarity
                            similarity = aesthetic_editor.calculate_style_similarity(
                                img_embedding, st.session_state.style_vector
                            )
                            
                            # Determine editing parameters
                            if st.session_state.style_name in default_styles:
                                style_config = default_styles[st.session_state.style_name]
                                filter_name = style_config['filter']
                                outline_name = style_config['outline']
                                doodle_name = style_config['doodle']
                            else:
                                # AI-determined parameters based on similarity
                                filter_name = 'vivid' if similarity > 0.7 else 'warm' if similarity > 0.4 else 'soft'
                                outline_name = 'thin_black' if similarity > 0.6 else 'sketchy'
                                doodle_name = 'geometric' if similarity > 0.7 else 'minimal'
                            
                            # Apply manual overrides
                            if custom_filter != 'auto':
                                filter_name = custom_filter
                            if custom_outline != 'auto':
                                outline_name = custom_outline
                            if custom_doodle != 'auto':
                                doodle_name = custom_doodle
                            
                            intensity = manual_intensity if manual_intensity else similarity
                            
                            # Apply layers
                            edited_image = original_image.copy()
                            
                            # Layer 1: Filter
                            edited_image = aesthetic_editor.apply_filter_layer(
                                edited_image, filter_name, intensity
                            )
                            
                            # Layer 2: Outline
                            edited_image = aesthetic_editor.add_outline_layer(
                                edited_image, outline_name, intensity * 0.8
                            )
                            
                            # Layer 3: Doodles
                            edited_image = aesthetic_editor.add_doodle_layer(
                                edited_image, doodle_name, intensity * 0.6
                            )
                            
                            # Layer 4: Stickers
                            if add_stickers and 'sticker_files' in locals() and sticker_files:
                                stickers = []
                                for sticker_file in sticker_files:
                                    sticker_img = Image.open(sticker_file).convert('RGBA')
                                    # Remove background
                                    sticker_img = remove_background(sticker_img)
                                    stickers.append(sticker_img)
                                
                                if stickers:
                                    edited_image = apply_stickers(edited_image, stickers, intensity * 0.4)
                            
                            st.session_state.preview_image = edited_image
                            
                            # Display similarity score
                            st.info(f"Style Similarity: {similarity:.2%} | Intensity: {intensity:.2%}")
                    else:
                        st.warning("Please select or learn a style first!")
            
            with col_btn2:
                if st.button("✅ Apply Edit", use_container_width=True):
                    if st.session_state.preview_image is not None:
                        st.session_state.final_image = st.session_state.preview_image
                        st.success("Edit applied!")
                    else:
                        st.warning("Generate a preview first!")
            
            with col_btn3:
                if st.button("🔄 Reset", use_container_width=True):
                    st.session_state.preview_image = None
                    if 'final_image' in st.session_state:
                        del st.session_state.final_image
                    st.success("Reset complete!")
    
    with col2:
        st.header("✨ Edited Result")
        
        # Show preview or final image
        if 'final_image' in st.session_state:
            st.image(st.session_state.final_image, caption="Final Edited Image", use_column_width=True)
            
            # Download button
            img_buffer = io.BytesIO()
            st.session_state.final_image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            st.download_button(
                label="📥 Download Edited Image",
                data=img_buffer,
                file_name=f"edited_{st.session_state.style_name}_{uploaded_file.name if uploaded_file else 'image'}.png",
                mime="image/png",
                use_container_width=True
            )
            
        elif st.session_state.preview_image is not None:
            st.image(st.session_state.preview_image, caption="Preview (Click 'Apply Edit' to finalize)", use_column_width=True)
        
        else:
            st.info("Upload an image and generate a preview to see results here!")
        
        # Show current style info
        if st.session_state.style_name:
            with st.expander("ℹ️ Current Style Info"):
                st.write(f"**Style Name:** {st.session_state.style_name}")
                if st.session_state.style_name in default_styles:
                    st.write(f"**Type:** Default Preset")
                    st.write(f"**Description:** {default_styles[st.session_state.style_name]['description']}")
                else:
                    st.write(f"**Type:** Learned from user images")
                
                if st.session_state.style_vector is not