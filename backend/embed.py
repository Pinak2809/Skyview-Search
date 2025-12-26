# backend/embed.py
import torch
import numpy as np
import faiss
from open_clip import create_model_and_transforms, get_tokenizer
from PIL import Image
from db import init_db, ImageRecord
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model, _, preprocess = create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = get_tokenizer('ViT-B-32')
model.to(device)
model.eval()

DIM = 512

INDEX_PATH = "../data/processed/skyview/index.faiss"
MAPPING_PATH = "../data/processed/skyview/faiss_mapping.npy"


def image_to_embedding(path):
    img = Image.open(path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(x)
    emb = emb.cpu().numpy().astype('float32')[0]
    emb /= np.linalg.norm(emb)
    return emb


def main():
    Session = init_db()
    session = Session()
    
    rows = session.query(ImageRecord).filter(ImageRecord.embedding_id == None).all()
    total = len(rows)
    print(f"Images to embed: {total}")
    
    if total == 0:
        print("Nothing to do.")
        return
    
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
        mapping = np.load(MAPPING_PATH, allow_pickle=True).tolist()
        next_idx = len(mapping)
        print(f"Loaded existing index with {next_idx} vectors")
    else:
        index = faiss.IndexFlatIP(DIM)
        mapping = []
        next_idx = 0
        print("Created new FAISS index")
    
    BATCH_COMMIT = 100
    
    for i, r in enumerate(rows):
        try:
            emb = image_to_embedding(r.filepath)
            index.add(np.expand_dims(emb, axis=0))
            mapping.append(r.uuid)
            r.embedding_id = next_idx
            next_idx += 1
            
            if (i + 1) % BATCH_COMMIT == 0:
                session.commit()
                print(f"Progress: {i + 1}/{total} ({100*(i+1)/total:.1f}%)")
                
        except Exception as e:
            print(f"Error {r.uuid}: {e}")
    
    session.commit()
    faiss.write_index(index, INDEX_PATH)
    np.save(MAPPING_PATH, mapping)
    print(f"Done. FAISS index saved with {len(mapping)} vectors.")


if __name__ == "__main__":
    main()