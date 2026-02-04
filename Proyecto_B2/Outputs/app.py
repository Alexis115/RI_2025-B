import streamlit as st
from PIL import Image
from rag_core import (
    text_to_product_search,
    image_to_product_search,
    df
)

st.set_page_config(page_title="Sistema RAG", layout="centered")
st.title("🤖 Sistema RAG - Productos")

st.markdown("### 🔍 Búsqueda por texto")
query = st.text_input("¿Qué estás buscando?")

st.markdown("### 🖼️ Búsqueda por imagen")
uploaded_image = st.file_uploader(
    "Sube una imagen",
    type=["jpg", "jpeg", "png"]
)

search_clicked = st.button("Buscar")

if search_clicked:
    if query:
        scores, indices = text_to_product_search(query)

    elif uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Imagen cargada", width=250)
        scores, indices = image_to_product_search(image)

    else:
        st.warning("Ingresa un texto o sube una imagen.")
        st.stop()

    st.subheader("Resultados")

    for score, idx in zip(scores, indices):
        product = df.loc[idx]

        st.markdown(f"### {product['name']}")
        st.write(product["primaryCategories"])
        st.image(product["image_url"], width=200)
        st.write(f"Score: {round(float(score), 4)}")
        st.markdown("---")

