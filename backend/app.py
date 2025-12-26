# backend/app.py
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from PIL import Image
import io
import os

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
    
    return FileResponse(
        buf,
        media_type="image/jpeg",
        headers={"Content-Disposition": f"inline; filename={uuid}_thumb.jpg"}
    )


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