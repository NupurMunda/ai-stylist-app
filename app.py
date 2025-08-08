import streamlit as st
import torch
from PIL import Image, ImageEnhance, ImageFilter
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
import io
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Image Stylizer with ControlNet",
    page_icon="🎨",
    layout="wide"
)

# ---
# UI Elements and Helper Functions
# ---

# Function to simulate a filter effect
def apply_filter(image, filter_name):
    """
    Applies a filter effect to an image using more advanced PIL techniques.
    """
    img = image.convert("RGB")
    enhancer = ImageEnhance.Color(img)
    
    if filter_name == "Warm / Golden Hour":
        img = ImageEnhance.Brightness(img).enhance(1.1)
        img = enhancer.enhance(1.2)
        r, g, b = img.split()
        r = r.point(lambda p: p * 1.1)
        g = g.point(lambda p: p * 1.05)
        img = Image.merge('RGB', (r, g, b))
        overlay = Image.new('RGB', img.size, (255, 180, 50))
        img = Image.blend(img, overlay, alpha=0.1)
    
    elif filter_name == "Paris / Pastel / Soft":
        img = enhancer.enhance(0.8)
        img = ImageEnhance.Contrast(img).enhance(0.9)
        overlay = Image.new('RGB', img.size, (255, 192, 203))
        img = Image.blend(img, overlay, alpha=0.1)
    
    elif filter_name == "Vivid / Vibrant":
        img = enhancer.enhance(1.5)
        img = ImageEnhance.Contrast(img).enhance(1.2)
    
    elif filter_name == "Retro / Film / Grainy":
        img = enhancer.enhance(0.7)
        img = ImageEnhance.Contrast(img).enhance(1.2)
        # Add a light noise/grain overlay
        noise = np.random.normal(0, 15, img.size[::-1] + (3,))
        noise_img = Image.fromarray(np.uint8(np.clip(noise, 0, 255)))
        img = Image.blend(img, noise_img, alpha=0.08)
    
    elif filter_name == "Sepia / Brown":
        img = img.convert("L").convert("RGB")
        r, g, b = img.split()
        r = r.point(lambda p: p * 1.0)
        g = g.point(lambda p: p * 0.8)
        b = b.point(lambda p: p * 0.6)
        img = Image.merge('RGB', (r, g, b))
    
    elif filter_name == "Teal & Orange":
        img = enhancer.enhance(1.3)
        r, g, b = img.split()
        r = r.point(lambda p: p * 1.1)
        g = g.point(lambda p: p * 0.9)
        b = b.point(lambda p: p * 1.2)
        img = Image.merge('RGB', (r, g, b))
        img = ImageEnhance.Contrast(img).enhance(1.1)
    
    elif filter_name == "Desaturated / Minimalist":
        img = enhancer.enhance(0.5)
        img = ImageEnhance.Contrast(img).enhance(1.1)
    
    elif filter_name == "Pink Tint / Rosy Glow":
        overlay = Image.new('RGB', img.size, (255, 192, 203))
        img = Image.blend(img, overlay, alpha=0.15)
        img = ImageEnhance.Brightness(img).enhance(1.05)
    
    elif filter_name == "Cool Tone / Blue Tint":
        overlay = Image.new('RGB', img.size, (0, 0, 255))
        img = Image.blend(img, overlay, alpha=0.1)
        img = ImageEnhance.Contrast(img).enhance(1.1)
    
    elif filter_name == "Moody / Dark":
        img = ImageEnhance.Contrast(img).enhance(1.5)
        img = ImageEnhance.Brightness(img).enhance(0.7)
    
    elif filter_name == "Creamy / Soft Blur":
        img = img.filter(ImageFilter.BoxBlur(1))
        img = ImageEnhance.Contrast(img).enhance(0.9)

    elif filter_name == "Fujifilm / Kodak":
        img = ImageEnhance.Color(img).enhance(1.2)
        r, g, b = img.split()
        # Simulate film color shifts
        r = r.point(lambda p: p * 1.05)
        g = g.point(lambda p: p * 1.1)
        b = b.point(lambda p: p * 0.95)
        img = Image.merge('RGB', (r, g, b))
        img = ImageEnhance.Contrast(img).enhance(1.1)
    
    elif filter_name == "Dazzle / Sparkle":
        img = ImageEnhance.Brightness(img).enhance(1.2)
        img = ImageEnhance.Contrast(img).enhance(1.2)
        # Add high-frequency noise for a sparkling effect
        noise = np.random.normal(0, 30, img.size[::-1] + (3,))
        noise_img = Image.fromarray(np.uint8(np.clip(noise, 0, 255)))
        img = Image.blend(img, noise_img, alpha=0.15)

    elif filter_name == "HDR / Clarity":
        # Simulate local contrast enhancement
        sharpened = img.filter(ImageFilter.SHARPEN)
        img = Image.blend(img, sharpened, alpha=0.5)
        img = ImageEnhance.Contrast(img).enhance(1.5)
        
    return img

def apply_manual_tweaks(image, brightness, contrast, saturation, filter_name):
    """
    Applies brightness, contrast, saturation, and a filter to an image.
    """
    img = image.copy()
    
    # Apply manual adjustments
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Color(img).enhance(saturation)

    # Apply filter
    if filter_name != "None":
        img = apply_filter(img, filter_name)

    return img

# ---
# AI Logic
# ---

@st.cache_resource
def load_controlnet_pipeline():
    """
    Loads and caches the Stable Diffusion ControlNet pipeline.
    """
    try:
        hf_token = st.secrets["HF_TOKEN"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11e_sd15_shuffle",   # Corrected model ID
            torch_dtype=dtype,
            token=hf_token,
            use_safetensors=True
        )

        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=dtype,
            token=hf_token
        ).to(device)

        # Optional memory helpers
        pipe.enable_attention_slicing()
        if device == "cuda":
            pipe.enable_xformers_memory_efficient_attention()

        return pipe
    except Exception as e:
        st.error(f"Failed to load pipeline: {e}")
        return None

def generate_prompt(add_doodle, add_sticker, add_text, custom_text):
    """
    Dynamically generates a prompt based on user selections.
    """
    prompt = "Edit the target image to match the aesthetic of the reference image. "
    if add_doodle:
        prompt += "Add doodles. "
    if add_sticker:
        prompt += "Include stickers. "
    if add_text and custom_text:
        prompt += f"Add the text: '{custom_text}' in a matching style. "
    prompt += "Preserve the subject of the target image."
    return prompt

def generate_edit_with_controlnet(pipe, reference_image, target_image, prompt):
    """
    Calls the ControlNet pipeline to generate a stylized image.
    """
    if pipe is None:
        st.error("AI pipeline is not loaded. Cannot generate image.")
        return None
    
    try:
        reference_image = reference_image.resize((512, 512))
        target_image = target_image.resize((512, 512))
        
        output = pipe(
            prompt=prompt,
            image=target_image,
            control_image=reference_image,
            controlnet_conditioning_scale=0.8,  # New parameter for style strength
            num_inference_steps=30,
            strength=0.75,
            guidance_scale=7.5
        ).images[0]
        
        return output
    except Exception as e:
        st.error(f"AI generation failed. Error: {e}")
        return None

# ---
# Streamlit App Layout
# ---

st.title("🎨 Image Stylizer App")
st.markdown("Edit a target image to match the aesthetic of a reference image using **ControlNet** and **Stable Diffusion**.")

if 'generated_image' not in st.session_state:
    st.session_state.generated_image = None

st.header("1. Image Upload")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Reference Image (Style)")
    reference_file = st.file_uploader("Upload an image with the desired aesthetic.", type=["png", "jpg", "jpeg"])
    if reference_file:
        reference_image = Image.open(reference_file).convert("RGB")
        st.image(reference_image, caption="Reference Image", use_column_width=True)

with col2:
    st.subheader("Target Image (Subject)")
    target_file = st.file_uploader("Upload the image you want to edit.", type=["png", "jpg", "jpeg"])
    if target_file:
        target_image = Image.open(target_file).convert("RGB")
        st.image(target_image, caption="Target Image", use_column_width=True)

st.markdown("---")
st.header("2. Advanced Aesthetic Options")
col1, col2, col3 = st.columns(3)
add_doodle = col1.checkbox("Add doodles")
add_sticker = col2.checkbox("Add stickers")
add_text = col3.checkbox("Add text")

custom_text = ""
if add_text:
    custom_text = st.text_input("Enter the text to add:", "Hello, World!")

# ---
# AI-Powered Edit Button
# ---
st.markdown("---")
st.header("3. Generate AI Edit")
if st.button("🚀 Generate Stylized Image"):
    if reference_file and target_file:
        with st.spinner("Generating your stylized image... This may take a moment."):
            pipe = load_controlnet_pipeline()
            prompt = generate_prompt(add_doodle, add_sticker, add_text, custom_text)
            
            generated_image = generate_edit_with_controlnet(pipe, reference_image, target_image, prompt)
            
            st.session_state.generated_image = generated_image

            if st.session_state.generated_image:
                st.success("Image generated successfully!")
            else:
                st.error("Failed to generate image. Please check the logs.")
    else:
        st.warning("Please upload both a reference and a target image first.")

# ---
# Manual Tweaks Section
# ---
if st.session_state.generated_image:
    st.markdown("---")
    st.header("4. Manual Tweaks")
    
    col1, col2 = st.columns(2)
    with col1:
        brightness = st.slider("Brightness", 0.5, 1.5, 1.0, 0.05)
        contrast = st.slider("Contrast", 0.5, 1.5, 1.0, 0.05)
        saturation = st.slider("Saturation", 0.5, 1.5, 1.0, 0.05)
    
    with col2:
        filter_options = [
            "None", "Warm / Golden Hour", "Paris / Pastel / Soft", "Vivid / Vibrant", 
            "Retro / Film / Grainy", "Sepia / Brown", "Teal & Orange", 
            "Desaturated / Minimalist", "Pink Tint / Rosy Glow", "Cool Tone / Blue Tint", 
            "Moody / Dark", "Creamy / Soft Blur"
        ]
        selected_filter = st.selectbox("Select a Filter:", filter_options)
        
    tweaked_image = apply_manual_tweaks(
        st.session_state.generated_image, 
        brightness, 
        contrast, 
        saturation, 
        selected_filter
    )
    
    st.markdown("---")
    st.header("5. Final Output")
    st.image(tweaked_image, caption="Final Stylized Image", use_column_width=True)
    
    # Download button
    buf = io.BytesIO()
    tweaked_image.save(buf, format="PNG")
    st.download_button(
        label="Download Final Image",
        data=buf.getvalue(),
        file_name="stylized_image.png",
        mime="image/png"
    )