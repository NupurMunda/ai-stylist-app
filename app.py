import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import torch
from transformers import CLIPProcessor, CLIPModel
from rembg import remove
import io
import pickle

# --- Setup and Initialization ---

# Check for CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model and processor from Hugging Face
@st.cache_resource
def load_clip_model():
    """Loads and caches the CLIP model and processor."""
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return model, processor
    except Exception as e:
        st.error(f"Error loading CLIP model: {e}")
        return None, None

model, processor = load_clip_model()

# Session State for managing app flow
if 'style_vector' not in st.session_state:
    st.session_state.style_vector = None
if 'edited_image' not in st.session_state:
    st.session_state.edited_image = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None

# --- Aesthetic Filters (Placeholder) ---
def apply_warm_filter(image):
    """Applies a simple warm filter."""
    hsv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    
    # Increase saturation and slightly shift hue towards red/orange
    s = cv2.add(s, 20)
    v = cv2.add(v, 10)
    
    final_hsv = cv2.merge([h, s, v])
    return Image.fromarray(cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB))

def apply_vintage_filter(image):
    """Applies a vintage, sepia-like filter."""
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia_img = cv2.transform(img_cv, sepia_filter)
    sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(sepia_img, cv2.COLOR_BGR2RGB))

def apply_vivid_filter(image):
    """Applies a vivid filter using PIL."""
    enhancer = ImageEnhance.Color(image)
    enhanced_image = enhancer.enhance(1.5)
    return enhanced_image

filters_dict = {
    'Warm Tones': apply_warm_filter,
    'Vintage': apply_vintage_filter,
    'Vivid': apply_vivid_filter,
}

# --- Core Editing Functions ---

def get_image_embedding(image):
    """Generates a CLIP embedding for a single image."""
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        embedding = model.get_image_features(**inputs)
    return embedding.squeeze(0).cpu().numpy()

def learn_style_from_images(reference_images):
    """Computes a style vector by averaging embeddings of reference images."""
    if not reference_images:
        st.warning("Please upload reference images to learn a style.")
        return None

    embeddings = []
    for img_file in reference_images:
        img = Image.open(img_file).convert('RGB')
        embedding = get_image_embedding(img)
        embeddings.append(embedding)

    # Average the embeddings
    style_vector = np.mean(embeddings, axis=0)
    
    # Save the style vector
    with open('style_vector.pkl', 'wb') as f:
        pickle.dump(style_vector, f)
    
    st.success("Aesthetic style learned and saved successfully!")
    return style_vector

def load_preset_style(style_name):
    """Loads a pretrained style vector."""
    # Placeholder for loading preset style vectors from files
    # In a real app, you would have files like 'vintage_style.pkl', etc.
    try:
        with open(f'{style_name.lower().replace(" ", "_")}_style.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.warning(f"Preset style '{style_name}' not found. Using a default placeholder.")
        # Create and return a dummy vector for demonstration
        return np.random.rand(512)

def apply_outline(image):
    """Adds a simple outline to the main subject using Canny edge detection."""
    img_cv = np.array(image.convert('L')) # Convert to grayscale
    edges = cv2.Canny(img_cv, 100, 200)
    
    # Dilate edges for a thicker line
    kernel = np.ones((2,2), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Create a mask for the outline
    outline_mask = Image.fromarray(dilated_edges)
    
    # Create an RGBA image for the outline
    outline_color = (255, 0, 0, 255) # Red outline with full transparency
    outline_rgba = Image.new('RGBA', image.size, (0, 0, 0, 0))
    outline_rgba.paste(outline_color, box=(0, 0), mask=outline_mask)
    
    return Image.alpha_composite(image.convert('RGBA'), outline_rgba)

def apply_doodles(image):
    """Overlays simple doodle shapes (e.g., stars, hearts)."""
    # Placeholder: In a real app, you would load doodle PNGs from a folder.
    # For now, we'll create a simple shape.
    
    doodle_image = Image.new('RGBA', image.size, (0,0,0,0))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(doodle_image)
    
    # Draw a simple circle as a doodle
    draw.ellipse((50, 50, 150, 150), outline=(255, 255, 0), width=5)
    
    return Image.alpha_composite(image.convert('RGBA'), doodle_image)

def apply_stickers(image, sticker_path="stickers/sticker1.png"):
    """Overlays a sticker on the image."""
    try:
        sticker = Image.open(sticker_path).convert('RGBA')
        
        # Resize sticker to a manageable size
        sticker_width = int(image.width * 0.2)
        sticker_height = int(image.height * 0.2)
        sticker = sticker.resize((sticker_width, sticker_height))
        
        # Position the sticker (e.g., bottom-right)
        position = (image.width - sticker.width - 20, image.height - sticker.height - 20)
        
        composite_image = image.copy().convert('RGBA')
        composite_image.paste(sticker, position, sticker)
        
        return composite_image
    except FileNotFoundError:
        st.warning(f"Sticker file not found: {sticker_path}")
        return image.convert('RGBA')

def ai_sticker_generator(prompt):
    """
    Placeholder for an AI sticker generator using Hugging Face.
    This would call an API like the Diffusion API for transparent images.
    """
    st.info(f"Generating sticker for: '{prompt}' (This is a placeholder feature).")
    # In a real implementation, you would use a library like `diffusers`
    # or the Hugging Face Inference API for image generation.
    # For now, return a placeholder path.
    return "stickers/ai_generated_placeholder.png"


def apply_layered_edits(image, style_vector):
    """
    Applies aesthetic layers based on the learned style vector.
    """
    st.info("Applying AI-powered edits...")
    
    # 1. Get embedding of the new image
    new_image_embedding = get_image_embedding(image)
    
    # 2. Compare embeddings (cosine similarity)
    # Cosine similarity formula: (A . B) / (||A|| * ||B||)
    similarity = np.dot(new_image_embedding, style_vector) / (np.linalg.norm(new_image_embedding) * np.linalg.norm(style_vector))
    
    st.write(f"Style Similarity Score: {similarity:.2f}")

    # 3. Layered Editing Logic based on similarity
    edited_image = image.copy()
    
    # Layer 1: Apply Filter
    # Use similarity to pick a filter. A higher similarity might mean a more 'intense' filter.
    filter_keys = list(filters_dict.keys())
    # Simple logic: pick a filter based on a mapped similarity score
    if similarity > 0.8:
        selected_filter = filter_keys[0] # High similarity -> First filter
    elif similarity > 0.6:
        selected_filter = filter_keys[1] # Medium similarity -> Second filter
    else:
        selected_filter = filter_keys[2] # Low similarity -> Third filter
    
    edited_image = filters_dict[selected_filter](edited_image)
    st.sidebar.write(f"Filter Applied: {selected_filter}")
    
    # Layer 2: Outline
    edited_image = apply_outline(edited_image)
    st.sidebar.write("Outline Added.")
    
    # Layer 3: Doodles
    edited_image = apply_doodles(edited_image)
    st.sidebar.write("Doodles Added.")
    
    # Layer 4: Stickers
    edited_image = apply_stickers(edited_image)
    st.sidebar.write("Sticker Added.")

    st.success("AI edits applied!")
    return edited_image

# --- Streamlit UI Layout ---

st.title("AI-Powered Aesthetic Photo Editor")

# Sidebar for controls
with st.sidebar:
    st.header("1. Learn Your Style")
    reference_images = st.file_uploader(
        "Upload 3-10 reference images to learn your style:",
        type=['jpg', 'png', 'jpeg'], accept_multiple_files=True
    )
    
    if st.button("Learn Style"):
        if len(reference_images) < 3 or len(reference_images) > 10:
            st.warning("Please upload between 3 and 10 reference images.")
        else:
            style_vec = learn_style_from_images(reference_images)
            if style_vec is not None:
                st.session_state.style_vector = style_vec
    
    st.markdown("---")
    
    st.header("2. Choose Style Option")
    style_option = st.radio(
        "Choose an editing style:",
        ('Use Learned Style', 'Warm Vintage Preset', 'Cool Tones Preset')
    )
    
    if style_option == 'Use Learned Style':
        if st.session_state.style_vector is None:
            st.info("Please upload and learn your style first.")
        else:
            st.success("Using your custom learned style.")
    else:
        st.session_state.style_vector = load_preset_style(style_option)
        st.success(f"Using preset style: {style_option}")
    
    if st.button("Reset Style"):
        st.session_state.style_vector = None
        st.session_state.edited_image = None
        st.success("Style has been reset.")
        
    st.markdown("---")
    
    st.header("3. User Sticker Options")
    uploaded_sticker = st.file_uploader("Upload your own sticker (PNG/JPG):", type=['png', 'jpg', 'jpeg'])
    if uploaded_sticker:
        # Remove background from uploaded sticker
        sticker_bytes = uploaded_sticker.read()
        sticker_no_bg = remove(sticker_bytes)
        sticker_pil = Image.open(io.BytesIO(sticker_no_bg)).convert('RGBA')
        
        # Save the processed sticker
        sticker_path = f"stickers/custom_sticker.png"
        sticker_pil.save(sticker_path)
        st.success("Sticker uploaded and background removed.")
        
    ai_sticker_prompt = st.text_input("Or, describe a sticker to generate:")
    if st.button("Generate AI Sticker"):
        if ai_sticker_prompt:
            ai_sticker_generator(ai_sticker_prompt)
        else:
            st.warning("Please enter a description for the sticker.")


# Main content area
st.header("4. Upload and Edit Your Photo")
uploaded_photo = st.file_uploader("Upload a photo to edit:", type=['jpg', 'png', 'jpeg'])

if uploaded_photo:
    st.session_state.original_image = Image.open(uploaded_photo).convert('RGB')
    st.image(st.session_state.original_image, caption="Original Photo", use_column_width=True)

    if st.session_state.style_vector is not None:
        if st.button("Preview AI Edit"):
            # Apply the layered edits
            st.session_state.edited_image = apply_layered_edits(st.session_state.original_image, st.session_state.style_vector)
            st.success("Preview ready!")
    else:
        st.warning("Please learn a style or choose a preset before editing.")

if st.session_state.edited_image:
    st.header("5. Final Result")
    st.image(st.session_state.edited_image, caption="AI Edited Image", use_column_width=True)
    
    # Manual Tweak Options (just a placeholder for now)
    st.subheader("Manual Tweaks")
    brightness = st.slider("Brightness", 0.5, 1.5, 1.0)
    contrast = st.slider("Contrast", 0.5, 1.5, 1.0)
    
    if st.button("Apply Manual Tweaks"):
        enhancer = ImageEnhance.Brightness(st.session_state.edited_image)
        manual_image = enhancer.enhance(brightness)
        enhancer = ImageEnhance.Contrast(manual_image)
        manual_image = enhancer.enhance(contrast)
        st.session_state.edited_image = manual_image
        st.image(st.session_state.edited_image, caption="Edited with Tweaks", use_column_width=True)

    # Download button
    buf = io.BytesIO()
    st.session_state.edited_image.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="Download Final Image",
        data=byte_im,
        file_name="ai_edited_photo.png",
        mime="image/png"
    )