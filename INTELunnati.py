#pip install torch torchvision open-clip-torch gradio faiss-cpu datasets pillow requests

import torch
import open_clip
import gradio as gr
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
import numpy as np

# Load the dataset (fashion product images dataset from hugging face )
dataset = load_dataset("ceyda/fashion-products-small", split="train")

# Load CLIP model with correct unpacking and QuickGELU
model = open_clip.create_model("ViT-B-32-quickgelu", pretrained="openai")

# Corrected image transform function
preprocess = open_clip.image_transform(model.visual.image_size, is_train=False)

# Load tokenizer
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Function to compute image embeddings
def get_image_embedding(image):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features / image_features.norm(dim=-1, keepdim=True)

# Function to compute text embeddings
def get_text_embedding(text):
    text_inputs = tokenizer([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    return text_features / text_features.norm(dim=-1, keepdim=True)

# Precompute embeddings for all images in the dataset
image_embeddings = []
image_paths = []
for item in dataset.select(range(1000)):  # Limit to 1000 images for speed
    image = item["image"]
    image_embeddings.append(get_image_embedding(image))
    image_paths.append(image)


image_embeddings = torch.cat(image_embeddings, dim=0)

# Function to search for similar images based on text
def search_similar_image(query_text):
    text_embedding = get_text_embedding(query_text)
    similarities = (image_embeddings @ text_embedding.T).squeeze(1).cpu().numpy()
    
    # Get top 20 matches
    best_match_idxs = np.argsort(similarities)[-20:][::-1]

    return [image_paths[i] for i in best_match_idxs]

# Function to search for similar images based on an uploaded image
def search_similar_by_image(uploaded_image):
    query_embedding = get_image_embedding(uploaded_image)
    similarities = (image_embeddings @ query_embedding.T).squeeze(1).cpu().numpy()

    # Get top 20 matches
    best_match_idxs = np.argsort(similarities)[-20:][::-1]

    return [image_paths[i] for i in best_match_idxs]

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🛍️ Visual Search for Fashion Products")
    gr.Markdown("Search using **text** or **upload an image** to find similar items.")

    with gr.Row():
        query_input = gr.Textbox(label="Search by Text", placeholder="e.g., red sneakers")
        search_button = gr.Button("Search by Text")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload an Image")
        image_search_button = gr.Button("Search by Image")

    output_gallery = gr.Gallery(label="Similar Items", columns=4, height=500)

    search_button.click(search_similar_image, inputs=[query_input], outputs=[output_gallery])
    image_search_button.click(search_similar_by_image, inputs=[image_input], outputs=[output_gallery])

demo.launch(share=True)