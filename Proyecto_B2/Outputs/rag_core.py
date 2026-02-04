import torch
import faiss
import pandas as pd
from transformers import CLIPProcessor, CLIPModel

print("✅ rag_core.py CARGADO – versión DEFENSIVA")

device = "cuda" if torch.cuda.is_available() else "cpu"

# 🔹 Modelo
clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(device)
clip_model.eval()

clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

# 🔹 Dataset
df = pd.read_csv("df_metadata.csv")

# 🔹 FAISS
text_index = faiss.read_index("text_index.faiss")

# 🔹 PROTECCIÓN CLIP
def extract_clip_embedding(output):
    if isinstance(output, torch.Tensor):
        return output
    elif hasattr(output, "pooler_output"):
        return output.pooler_output
    else:
        raise ValueError("Salida CLIP no reconocida")

# 🔹 BÚSQUEDA TEXTO → PRODUCTO
def text_to_product_search(query, k=5):
    inputs = clip_processor(
        text=[query],
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    with torch.no_grad():
        raw_output = clip_model.get_text_features(**inputs)

    text_emb = extract_clip_embedding(raw_output)

    # Normalización
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    query_embedding = text_emb.cpu().numpy().astype("float32")

    scores, indices = text_index.search(query_embedding, k)
    return scores[0], indices[0]

from PIL import Image
import numpy as np

# 🔹 Cargar índice de imágenes
image_index = faiss.read_index("image_index.faiss")

# 🔹 BÚSQUEDA IMAGEN → PRODUCTO
def image_to_product_search(image: Image.Image, k=5):
    """
    image: PIL.Image
    """
    inputs = clip_processor(
        images=image,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        raw_output = clip_model.get_image_features(**inputs)

    image_emb = extract_clip_embedding(raw_output)

    # Normalización
    image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)

    query_embedding = image_emb.cpu().numpy().astype("float32")

    scores, indices = image_index.search(query_embedding, k)
    return scores[0], indices[0]
