# AI Image Editor: Learn Your Style

A personalized AI-powered image editing app that learns your aesthetic from reference photos and enhances future images to match your style. Built with **Streamlit**, **CLIP embeddings**, and sticker-based overlays, it evolves over time based on user feedback and style preference.

---

## 🚀 Features

### 1. Style Learning from Reference Images

* Upload at least 3 aesthetic reference images the first time.
* The app extracts CLIP-based style embeddings and saves your visual taste.
* In future sessions, reuse the learned style or upload new reference images.

### 2. Aesthetic-Based Editing

* When you upload a new photo:

  * The AI calculates its similarity to your style.
  * Applies warm filters, doodle outlines, and sticker themes accordingly.

### 3. Continuous Learning with Feedback

* After the AI edits your photo, you can give a thumbs-up or thumbs-down.
* Positive feedback helps the AI refine your style further using the new photo.

### 4. Sticker Management

* Upload your own sticker images.
* Backgrounds are automatically removed using AI.
* Stickers are saved for future reuse in your edits.

### 5. AI-Generated Stickers (Coming Soon)

* Generate aesthetic stickers using AI with prompts like:

  * "hand-drawn kawaii sun with transparent background"
  * "minimal aesthetic flower outline"

---

## 🛠️ Tech Stack

* **Streamlit** for interactive UI
* **CLIP** for style embedding and similarity
* **OpenCV** + **Pillow** for filter/outline effects
* **rembg** for background removal
* **pickle** for local storage of style vectors
* (Optional) **DALL·E API / ImageGen AI** for sticker generation

---

## 📂 Folder Structure

```
📁 ai-image-editor/
├── app.py                 # Main Streamlit app
├── style_vector.pkl       # Saved CLIP vector of your aesthetic
├── /stickers              # Custom and AI-generated stickers
├── /reference_images      # Style reference uploads
├── /processed_images      # Output edits
├── requirements.txt
└── README.md
```

---

## 🔧 Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/NupurMunda/ai-image-editor.git
cd ai-image-editor
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the app**

```bash
streamlit run app.py
```

---

## 🌐 Deployment

You can deploy the app using:

* [Streamlit Community Cloud](https://streamlit.io/cloud)
* [Render](https://render.com/)
* [Railway](https://railway.app/)
* Or host locally using Streamlit

---

## 📌 Future Plans

* Integrate OpenAI or Hugging Face models for sticker generation
* Add login & cloud sync for style preferences
* Auto-adjust brightness/contrast based on mood
* Mobile UI optimization

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).

---
