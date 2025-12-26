# backend/search_util.py
import numpy as np, faiss
from open_clip import create_model_and_transforms, get_tokenizer
import torch
from db import init_db, ImageRecord

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = get_tokenizer('ViT-B-32')
model.to(device)
model.eval()

INDEX_PATH = "../data/processed/skyview/index.faiss"
MAPPING_PATH = "../data/processed/skyview/faiss_mapping.npy"

index = faiss.read_index(INDEX_PATH)
mapping = np.load(MAPPING_PATH, allow_pickle=True).tolist()

def text_to_embedding(text):
    tokens = tokenizer(text)
    import torch
    with torch.no_grad():
        txt_emb = model.encode_text(tokens.to(device))
    emb = txt_emb.cpu().numpy().astype('float32')[0]
    emb /= np.linalg.norm(emb)
    return emb

def search_text(text, k=5):
    v = text_to_embedding(text).reshape(1,-1)
    D, I = index.search(v, k)
    res = []
    from db import init_db
    Session = init_db()
    session = Session() 
    for score, idx in zip(D[0], I[0]):
        if idx < 0: continue
        uuid = mapping[idx]
        rec = session.query(ImageRecord).filter_by(uuid=uuid).first()
        res.append({"uuid": uuid, "score": float(score), "caption": rec.caption, "filepath": rec.filepath})
    return res
