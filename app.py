import os
import io
import torch
import numpy as np
import pandas as pd
from PIL import Image
from flask import Flask, request, render_template, send_from_directory
import torch.nn.functional as F
from open_clip import create_model_and_transforms
import open_clip
from sklearn.decomposition import PCA

device = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_FOLDER = "static/images/train2014"

# Load the embeddings
df = pd.read_pickle('image_embeddings.pickle')
embeddings_matrix = np.stack(df['embedding'].values, axis=0)

# Load the CLIP model (make sure model name and tokenizer name are consistent)
model, preprocess_train, preprocess_val = create_model_and_transforms('ViT-B-32', pretrained='openai')
model = model.to(device)
model.eval()

# Get the tokenizer function properly
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Set up PCA embeddings
pca = PCA(n_components=64)
pca_embeddings = pca.fit_transform(embeddings_matrix)

app = Flask(__name__)

def get_image_embedding_from_file(file):
    image = Image.open(file).convert("RGB")
    image_tensor = preprocess_val(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(image_tensor)
        emb = F.normalize(emb, p=2, dim=-1)
    return emb.cpu().numpy()

def get_text_embedding(query_text):
    text_tokens = tokenizer([query_text]).to(device)
    with torch.no_grad():
        text_emb = model.encode_text(text_tokens)
        text_emb = F.normalize(text_emb, p=2, dim=-1)
    return text_emb.cpu().numpy()

def search_similar(embedding, use_pca=False, k_pca=64, top_k=5):
    if not use_pca:
        sims = embeddings_matrix @ embedding.reshape(-1)
    else:
        reduced_embedding = pca.transform(embedding.reshape(1, -1))[:, :k_pca]
        sims = pca_embeddings[:, :k_pca] @ reduced_embedding.reshape(-1)

    top_inds = np.argsort(-sims)[:top_k]
    results = df.iloc[top_inds]
    scores = sims[top_inds]
    return list(zip(results['file_name'].values, scores))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text_query = request.form.get('text_query', '')
        lam = float(request.form.get('lambda', '0.5'))
        use_pca = 'use_pca' in request.form
        k_pca = int(request.form.get('k_pca', '64'))

        image_emb = None
        text_emb = None

        # Check if an image was uploaded
        if 'image_query' in request.files and request.files['image_query'].filename != '':
            img_file = request.files['image_query']
            image_emb = get_image_embedding_from_file(img_file)

        # Check if text was provided
        if text_query.strip():
            text_emb = get_text_embedding(text_query.strip())

        # Determine final query embedding
        if image_emb is not None and text_emb is not None:
            # Hybrid
            query = lam * text_emb + (1 - lam) * image_emb
        elif image_emb is not None:
            query = image_emb
        elif text_emb is not None:
            query = text_emb
        else:
            # No query provided
            return render_template('index.html', error="Please provide at least a text query or an image.")

        results = search_similar(query, use_pca=use_pca, k_pca=k_pca, top_k=5)
        result_images = [(os.path.join('static', 'images', 'train2014', fname), score) for fname, score in results]

        return render_template('results.html', 
                               results=result_images, 
                               text_query=text_query, 
                               use_pca=use_pca, 
                               k_pca=k_pca, 
                               lam=lam)
    return render_template('index.html')

@app.route('/static/images/train2014/<filename>')
def send_image(filename):
    return send_from_directory(os.path.join('static', 'images', 'train2014'), filename)

if __name__ == '__main__':
    app.run(debug=True)
