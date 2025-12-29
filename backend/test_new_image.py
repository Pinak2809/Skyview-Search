# backend/test_new_image.py
import torch
import numpy as np
import faiss
from open_clip import create_model_and_transforms, get_tokenizer
from PIL import Image
from db import init_db, ImageRecord
import sys

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model
model, _, preprocess = create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.to(device)
model.eval()

# Load FAISS index
INDEX_PATH = "../data/processed/skyview/index.faiss"
MAPPING_PATH = "../data/processed/skyview/faiss_mapping.npy"
index = faiss.read_index(INDEX_PATH)
mapping = np.load(MAPPING_PATH, allow_pickle=True).tolist()

Session = init_db()

def embed_image(image_path):
    """Embed a single image."""
    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(x)
    emb = emb.cpu().numpy().astype('float32')[0]
    emb /= np.linalg.norm(emb)
    return emb

def find_similar_images(image_path, k=5):
    """Find images similar to the given image."""
    print(f"\nSearching for images similar to: {image_path}")
    print("=" * 60)
    
    # Embed the query image
    query_emb = embed_image(image_path)
    
    # Search FAISS
    D, I = index.search(query_emb.reshape(1, -1), k)
    
    # Get results
    session = Session()
    results = []
    
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        uuid = mapping[idx]
        rec = session.query(ImageRecord).filter_by(uuid=uuid).first()
        if rec:
            results.append({
                "uuid": uuid,
                "score": float(score),
                "category": rec.category,
                "caption": rec.caption,
                "filepath": rec.filepath
            })
    
    session.close()
    return results

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_new_image.py <image_path>")
        print("Example: python test_new_image.py C:\\Users\\p.Ganatra\\Pictures\\my_photo.jpg")
        return
    
    image_path = sys.argv[1]
    
    results = find_similar_images(image_path, k=5)
    
    for i, r in enumerate(results, 1):
        caption = r['caption'][:50] if r['caption'] else "No caption"
        print(f"  {i}. {r['score']:.3f} | {r['category']:12} | {caption}")

if __name__ == "__main__":
    main()