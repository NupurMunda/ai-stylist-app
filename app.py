import streamlit as st
import torch
import time
import requests
import io
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import replicate

# Set page configuration
st.set_page_config(
    page_title="Image Stylizer with ControlNet",
    page_icon="🎨",
    layout="wide"
)

# A small diagnostic block to check the environment
st.write({
    "cuda_available": torch.cuda.is_available(),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "torch_version": torch.__version__
})

# ---
# Replicate API Functions
# ---

@st.cache_resource
def get_replicate_client():
    """Returns a cached Replicate client."""
    token = st.secrets.get("REPLICATE_API_TOKEN")
    if not token:
        st.error("Replicate API token not found. Please add `REPLICATE_API_TOKEN` to your Streamlit secrets.")
        st.stop()
    return replicate.Client(api_token=token)

def replicate_upload_image(client, pil_img: Image.Image) -> str:
    """Uploads a PIL image to Replicate's temporary CDN."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    uploaded = client.files.upload(buf, filename="image.png")
    return uploaded.url

def resize_max_side(pil_img: Image.Image, max_side=1024) -> Image.Image:
    """Resizes an image so its longest side is no more than max_side."""
    w, h = pil_img.size
    scale = max(w, h) / max_side
    if scale <= 1: 
        return pil_img
    return pil_img.resize((int(w/scale), int(h/scale)), Image.LANCZOS)

def run_controlnet_generation(prompt, ref_img, negative_prompt, conditioning_scale, num_steps, guidance_scale, controlnet_type):
    """
    Calls the ControlNet model on Replicate with the correct schema.
    """
    client = get_replicate_client()

    ref_img = resize_max_side(ref_img)
    ref_url = replicate_upload_image(client, ref_img)

    # You must replace this with a valid model slug and version
    MODEL_SLUG = "fofr/sdxl-multi-controlnet-lora:89eb212b3d1366a83e949c12a4b45dfe6b6b313b594cb8268e864931ac9ffb16"

    inputs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "apply_watermark": False,
        "controlnet_1": controlnet_type,
        "controlnet_1_image": ref_url,
        "controlnet_1_conditioning_scale": conditioning_scale,
        "num_inference_steps": num_steps,
        "guidance_scale": guidance_scale,
    }

    st.write({"model_slug": MODEL_SLUG, "inputs": inputs})
    pred = client.predictions.create(model=MODEL_SLUG, input=inputs)

    status = st.empty()
    wait = 1
    while pred.status not in ("succeeded", "failed", "canceled"):
        status.write(f"Status: {pred.status}")
        time.sleep(wait)
        pred = client.predictions.get(pred.id)
        wait = min(wait * 1.5, 5)

    if pred.status != "succeeded":
        raise RuntimeError(f"Replicate failed with status: {pred.status}. Error: {getattr(pred, 'error', None) or pred.output}")
    
    # Correctly handle Replicate's different output types
    output = pred.output
    if isinstance(output, list):
        first = output[0]
    else:
        first = output
    
    if hasattr(first, "url"):
        out_url = first.url
    else:
        out_url = str(first)

    img_bytes = requests.get(out_url).content
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")

# ---
# UI Elements and Helper Functions
# ---

def apply_filter(image, filter_name):
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
        r = r.point(lambda p: p * 1.05)
        g = g.point(lambda p: p * 1.1)
        b = b.point(lambda p: p * 0.95)
        img = Image.merge('RGB', (r, g, b))
        img = ImageEnhance.Contrast(img).enhance(1.1)
    
    elif filter_name == "Dazzle / Sparkle":
        img = ImageEnhance.Brightness(img).enhance(1.2)
        img = ImageEnhance.Contrast(img).enhance(1.2)
        noise = np.random.normal(0, 30, img.size[::-1] + (3,))
        noise_img = Image.fromarray(np.uint8(np.clip(noise, 0, 255)))
        img = Image.blend(img, noise_img, alpha=0.15)

    elif filter_name == "HDR / Clarity":
        sharpened = img.filter(ImageFilter.SHARPEN)
        img = Image.blend(img, sharpened, alpha=0.5)
        img = ImageEnhance.Contrast(img).enhance(1.5)
        
    return img

def apply_manual_tweaks(image, brightness, contrast, saturation, filter_name):
    """
    Applies brightness, contrast, saturation, and a filter to an image.
    """
    img = image.copy()
    
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Color(img).enhance(saturation)

    if filter_name != "None":
        img = apply_filter(img, filter_name)

    return img

def generate_prompt(add_doodle, add_sticker, add_text, custom_text):
    """
    Dynamically generates a prompt based on user selections.
    """
    prompt = "Edit the image to match the aesthetic of the reference image. "
    if add_doodle:
        prompt += "Add doodles. "
    if add_sticker:
        prompt += "Include stickers. "
    if add_text and custom_text:
        prompt += f"Add the text: '{custom_text}' in a matching style. "
    prompt += "Preserve the subject of the image."
    return prompt

# ---
# Streamlit App Layout
# ---

st.title("🎨 Image Stylizer App")
st.markdown("Use a **reference image** to guide the style of a generated image.")

if 'generated_image' not in st.session_state:
    st.session_state.generated_image = None

st.header("1. Image Upload")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Reference Image (Style)")
    st.markdown("This image provides the aesthetic style, layout, and subject for the AI to follow.")
    reference_file = st.file_uploader("Upload a reference image.", type=["png", "jpg", "jpeg"])
    if reference_file:
        reference_image = Image.open(reference_file).convert("RGB")
        st.image(reference_image, caption="Reference Image", use_container_width=True)

st.markdown("---")
st.header("2. Prompt and Aesthetic Options")
st.markdown("This prompt will guide the AI in stylizing the reference image.")

col1, col2, col3 = st.columns(3)
add_doodle = col1.checkbox("Add doodles")
add_sticker = col2.checkbox("Add stickers")
add_text = col3.checkbox("Add text")

custom_text = ""
if add_text:
    custom_text = st.text_input("Enter the text to add:", "Hello, World!")
    
negative_prompt = st.text_input(
    "Negative Prompt (Optional):", 
    value="blurry, ugly, poor quality, bad colors"
)

with st.expander("Advanced Settings"):
    col1, col2 = st.columns(2)
    num_inference_steps = col1.slider("Inference Steps", 10, 50, 28, 1)
    guidance_scale = col2.slider("Guidance Scale", 1.0, 15.0, 7.0, 0.5)

    col3, col4 = st.columns(2)
    conditioning_scale = col3.slider("ControlNet Conditioning Scale", 0.0, 2.0, 0.7, 0.1)

    controlnet_type_options = ["shuffle", "soft_edge_hed", "canny", "depth"]
    selected_controlnet_type = st.selectbox(
        "Select ControlNet Type:",
        options=controlnet_type_options,
        index=0
    )


st.markdown("---")
st.header("3. Generate AI Edit")
if st.button("🚀 Generate Stylized Image"):
    if reference_file:
        with st.spinner("Generating on GPU (Replicate)... This may take a moment."):
            prompt = generate_prompt(add_doodle, add_sticker, add_text, custom_text)
            
            try:
                result_img = run_controlnet_generation(
                    prompt, 
                    reference_image, 
                    negative_prompt, 
                    conditioning_scale,
                    num_inference_steps,
                    guidance_scale,
                    selected_controlnet_type
                )
                
                if result_img is not None:
                    st.session_state.generated_image = result_img
                    st.success("Image generated successfully!")
                else:
                    st.error("Generation failed. Check the logs for details.")
            except Exception as e:
                st.error(f"Remote inference failed: {e}")
    else:
        st.warning("Please upload a reference image first.")

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
            "Moody / Dark", "Creamy / Soft Blur", "Fujifilm / Kodak", "Dazzle / Sparkle", "HDR / Clarity"
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
    st.image(tweaked_image, caption="Final Stylized Image", use_container_width=True)
    
    buf = io.BytesIO()
    tweaked_image.save(buf, format="PNG")
    st.download_button(
        label="Download Final Image",
        data=buf.getvalue(),
        file_name="stylized_image.png",
        mime="image/png"
    )