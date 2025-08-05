# AI Stylist – Learning-Based Aesthetic Photo Editor

This is a machine learning-powered web application built with Streamlit that learns your personal aesthetic from reference images and automatically applies similar edits to new photos.

The app intelligently applies:
- Warm tone filters
- Doodle-style outlines
- Sticker overlays (auto or user-uploaded)
- Adaptive styling based on aesthetic similarity using CLIP embeddings

The more you use it, the better it becomes at matching your unique editing style.

---

## Features

- Learns aesthetic style from a set of 3 or more reference images
- Uses CLIP to compute aesthetic similarity between new images and your learned style
- Applies variable warm filters, outlines, and stickers based on style similarity
- Supports preloaded and user-uploaded stickers
- Automatically removes background from uploaded stickers (`rembg`)
- Continuously improves based on your feedback
- Allows downloading the final styled image
- Optional support for AI-generated stickers (e.g., via OpenAI DALL·E)

---

## How to Run Locally

### 1. Clone the repository

```bash
git clone git@github.com:NupurMunda/ai-stylist-app.git
cd ai-stylist-app
