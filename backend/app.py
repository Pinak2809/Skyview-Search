# backend/app.py
from fastapi import FastAPI, Query, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from PIL import Image
import io
import os
import uuid
import torch
import numpy as np
import faiss

from db import init_db, ImageRecord
from search_util import search_text

app = FastAPI(
    title="Skyview Search API",
    description="Semantic image search for aerial landscape imagery",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database session
Session = init_db()

# Paths
DATA_DIR = Path("../data/processed/skyview")
INDEX_PATH = DATA_DIR / "index.faiss"
MAPPING_PATH = DATA_DIR / "faiss_mapping.npy"
UPLOAD_DIR = DATA_DIR / "Uploaded"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Skyview Search API"}


@app.get("/search")
def search(
    q: str = Query(..., description="Search query text"),
    k: int = Query(5, ge=1, le=50, description="Number of results to return")
):
    """
    Search for images matching the query text.
    Returns top-k results with UUID, score, caption, category, and filepath.
    """
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    results = search_text(q, k=k)
    
    # Add category to results
    session = Session()
    for r in results:
        rec = session.query(ImageRecord).filter_by(uuid=r["uuid"]).first()
        if rec:
            r["category"] = rec.category
    session.close()
    
    return {
        "query": q,
        "k": k,
        "count": len(results),
        "results": results
    }


@app.get("/image/{uuid}")
def get_image(uuid: str):
    """Return full image by UUID."""
    session = Session()
    rec = session.query(ImageRecord).filter_by(uuid=uuid).first()
    session.close()
    
    if not rec:
        raise HTTPException(status_code=404, detail="Image not found")
    
    if not os.path.exists(rec.filepath):
        raise HTTPException(status_code=404, detail="Image file not found on disk")
    
    return FileResponse(rec.filepath, media_type="image/jpeg")


@app.get("/thumbnail/{uuid}")
def get_thumbnail(uuid: str, size: int = Query(128, ge=32, le=512)):
    """Return resized thumbnail by UUID."""
    session = Session()
    rec = session.query(ImageRecord).filter_by(uuid=uuid).first()
    session.close()
    
    if not rec:
        raise HTTPException(status_code=404, detail="Image not found")
    
    if not os.path.exists(rec.filepath):
        raise HTTPException(status_code=404, detail="Image file not found on disk")
    
    # Create thumbnail
    img = Image.open(rec.filepath)
    img.thumbnail((size, size))
    
    # Save to bytes
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    
    return StreamingResponse(buf, media_type="image/jpeg")


@app.get("/info/{uuid}")
def get_info(uuid: str):
    """Get metadata for an image by UUID."""
    session = Session()
    rec = session.query(ImageRecord).filter_by(uuid=uuid).first()
    session.close()
    
    if not rec:
        raise HTTPException(status_code=404, detail="Image not found")
    
    return {
        "uuid": rec.uuid,
        "category": rec.category,
        "caption": rec.caption,
        "filepath": rec.filepath,
        "created_at": rec.created_at.isoformat() if rec.created_at else None
    }


@app.get("/stats")
def get_stats():
    """Get database statistics."""
    session = Session()
    
    total = session.query(ImageRecord).count()
    captioned = session.query(ImageRecord).filter(ImageRecord.caption != None).count()
    embedded = session.query(ImageRecord).filter(ImageRecord.embedding_id != None).count()
    
    # Category counts
    from sqlalchemy import func
    categories = session.query(
        ImageRecord.category, 
        func.count(ImageRecord.uuid)
    ).group_by(ImageRecord.category).all()
    
    session.close()
    
    return {
        "total_images": total,
        "captioned": captioned,
        "embedded": embedded,
        "categories": {cat: count for cat, count in categories}
    }


@app.get("/categories")
def get_categories():
    """List all categories."""
    session = Session()
    from sqlalchemy import func
    categories = session.query(
        ImageRecord.category,
        func.count(ImageRecord.uuid)
    ).group_by(ImageRecord.category).all()
    session.close()
    
    return {"categories": [{"name": cat, "count": count} for cat, count in categories]}


@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    category: str = Form(default="Uploaded")
):
    """
    Upload a new image to the database.
    Generates caption and embedding automatically.
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((256, 256), Image.LANCZOS)
        
        # Generate UUID and save
        img_uuid = str(uuid.uuid4())
        filepath = UPLOAD_DIR / f"{img_uuid}.jpg"
        img.save(filepath, quality=95)
        
        # Generate caption
        caption = generate_caption(img)
        
        # Generate embedding and add to FAISS
        embedding_id = add_to_faiss(img, img_uuid)
        
        # Add to database
        session = Session()
        rec = ImageRecord(
            uuid=img_uuid,
            filepath=str(filepath.resolve()),
            category=category,
            caption=caption,
            embedding_id=embedding_id
        )
        session.add(rec)
        session.commit()
        session.close()
        
        return {
            "status": "success",
            "uuid": img_uuid,
            "caption": caption,
            "category": category
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.delete("/image/{uuid}")
def delete_image(uuid: str):
    """Delete an image from the database (does not remove from FAISS)."""
    session = Session()
    rec = session.query(ImageRecord).filter_by(uuid=uuid).first()
    
    if not rec:
        session.close()
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Delete file if exists
    if rec.filepath and os.path.exists(rec.filepath):
        try:
            os.remove(rec.filepath)
        except:
            pass
    
    # Delete from database
    session.delete(rec)
    session.commit()
    session.close()
    
    return {"status": "deleted", "uuid": uuid}


# ============================================================
# Helper functions for upload
# ============================================================

_caption_model = None
_caption_proc = None
_clip_model = None
_clip_preprocess = None

def get_caption_model():
    """Lazy load caption model."""
    global _caption_model, _caption_proc
    if _caption_model is None:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _caption_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _caption_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device)
    return _caption_model, _caption_proc


def get_clip_model():
    """Lazy load CLIP model."""
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        from open_clip import create_model_and_transforms
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model, _, _clip_preprocess = create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k'
        )
        _clip_model.to(device)
        _clip_model.eval()
    return _clip_model, _clip_preprocess


def generate_caption(img: Image.Image) -> str:
    """Generate caption for an image."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, proc = get_caption_model()
    
    inputs = proc(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=50)
    return proc.decode(out[0], skip_special_tokens=True)


def add_to_faiss(img: Image.Image, img_uuid: str) -> int:
    """Add image embedding to FAISS index."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = get_clip_model()
    
    # Generate embedding
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(x)
    emb = emb.cpu().numpy().astype('float32')[0]
    emb /= np.linalg.norm(emb)
    
    # Load index and mapping
    index = faiss.read_index(str(INDEX_PATH))
    mapping = np.load(str(MAPPING_PATH), allow_pickle=True).tolist()
    
    # Add to index
    embedding_id = len(mapping)
    index.add(np.expand_dims(emb, axis=0))
    mapping.append(img_uuid)
    
    # Save
    faiss.write_index(index, str(INDEX_PATH))
    np.save(str(MAPPING_PATH), mapping)
    
    return embedding_id
